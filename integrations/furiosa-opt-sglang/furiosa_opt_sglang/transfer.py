"""Bridge engine copy completion into SGLang's native transfer events."""

from __future__ import annotations

from collections.abc import Callable
from queue import Queue
from threading import Event, Lock, Thread
from typing import Any

from sglang.srt.mem_cache.l2_transfer import (
    L2TransferEngine,
    TransferCompletion,
    device_module,
    make_timing_event_pair,
)
from sglang.srt.mem_cache.pool_host.base import shared_host_layout_domains

from furiosa_opt_engine.cache_transfer import settle as settle_transfers


class _TransferResult:
    def __init__(self) -> None:
        self._lock = Lock()
        self._error: BaseException | None = None

    def fail(self, error: BaseException) -> None:
        with self._lock:
            if self._error is None:
                self._error = error

    def raise_if_failed(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError("Furiosa cache DMA failed") from error


class _CompletionEvent:
    """SGLang event view with its native event available to device allocators."""

    def __init__(self, event: Any, operation: _TransferOperation) -> None:
        self.native_event = event
        self._operation = operation

    def query(self) -> bool:
        self._operation.result.raise_if_failed()
        return self.native_event.query()

    def synchronize(self) -> None:
        failure = None
        try:
            self._operation.synchronize()
        except BaseException as error:
            failure = error
        try:
            self.native_event.synchronize()
        except BaseException as error:
            failure = _combine_errors(failure, error)
        if failure is not None:
            raise failure

    def elapsed_time(self, end_event: _CompletionEvent) -> float:
        self._operation.result.raise_if_failed()
        return self.native_event.elapsed_time(end_event.native_event)


class _TransferLane:
    """Keep one SGLang stream's submissions and completion events in call order."""

    def __init__(self, name: str) -> None:
        self._submissions: Queue[Any] = Queue()
        self._completions: Queue[Any] = Queue()
        self._shutdown_lock = Lock()
        self._shutdown = False
        self._submit_thread = Thread(
            target=self._submit_loop,
            name=f"furiosa-sglang-{name}-submit",
            daemon=True,
        )
        self._completion_thread = Thread(
            target=self._completion_loop,
            name=f"furiosa-sglang-{name}-complete",
            daemon=True,
        )
        self._submit_thread.start()
        try:
            self._completion_thread.start()
        except BaseException:
            self._submissions.put(None)
            self._submit_thread.join()
            raise

    def submit(self, operation: _TransferOperation) -> None:
        self._submissions.put(operation)

    def shutdown(self) -> None:
        with self._shutdown_lock:
            if self._shutdown:
                return
            self._shutdown = True
            self._submissions.put(None)
            self._submit_thread.join()
            self._completions.put(None)
            self._completion_thread.join()

    def _submit_loop(self) -> None:
        while (operation := self._submissions.get()) is not None:
            try:
                operation.submit()
            except BaseException as error:
                operation.fail(error)
            self._completions.put(operation)

    def _completion_loop(self) -> None:
        while (operation := self._completions.get()) is not None:
            try:
                operation.complete()
            except BaseException as error:
                operation.fail(error)
            finally:
                operation.mark_completed()


class _TransferOperation:
    def __init__(
        self,
        engine: L2TransferEngine,
        transfers: list[Any],
        stream: Any,
        ready_event: Any,
        start_event: Any,
        finish_event: Any,
        *,
        loading: bool,
        transfer_layer_id_max: int,
        on_layer_done: Callable[[int], None] | None,
        on_finish: Callable[[_TransferOperation], None],
    ) -> None:
        self.engine = engine
        self.transfers = transfers
        self.stream = stream
        self.ready_event = ready_event
        self.start_event = start_event
        self.finish_event = finish_event
        self.loading = loading
        self.transfer_layer_id_max = transfer_layer_id_max
        self.on_layer_done = on_layer_done
        self._on_finish = on_finish
        self.result = _TransferResult()
        self._completed = Event()
        self._domains = []
        self._prepared = []
        self._copies = []
        self._failure = None

    def synchronize(self) -> None:
        self._completed.wait()
        self.result.raise_if_failed()

    def fail(self, error: BaseException) -> None:
        self._failure = _combine_errors(self._failure, error)

    def mark_completed(self) -> None:
        if self._failure is not None:
            self.result.fail(self._failure)
        self._completed.set()
        self._on_finish(self)

    def submit(self) -> None:
        opened_batches = []

        try:
            self._domains = shared_host_layout_domains(
                transfer.host_pool for transfer in self.transfers
            )
            domains = self._domains
            self._domains = []
            for domain in domains:
                domain.acquire_layout()
                self._domains.append(domain)
            self.ready_event.synchronize()

            seen_hosts = set()
            for transfer in self.transfers:
                host = transfer.host_pool
                if id(host) in seen_hosts:
                    continue
                seen_hosts.add(id(host))
                begin_batch = getattr(host, "begin_async_copy_batch", None)
                if begin_batch is not None:
                    begin_batch()
                    opened_batches.append(host)

            with device_module.stream(self.stream):
                self._prepared = self.engine._prepare_transfers(self.transfers)
                self.start_event.record()
                if self.loading:
                    self._load_layers(self._prepared)
                else:
                    self._store_pages(self._prepared)
        except BaseException as error:
            self.fail(error)
            try:
                self.stream.synchronize()
            except BaseException as drain_error:
                self.fail(drain_error)
        finally:
            for host in opened_batches:
                try:
                    self._copies.extend(host.finish_async_copy_batch())
                except BaseException as error:
                    self.fail(error)

    def complete(self) -> None:
        copy_error = settle_transfers(self._copies)
        if copy_error is not None:
            self.fail(copy_error)

        if self._failure is None and self.loading and self.on_layer_done is not None:
            try:
                with device_module.stream(self.stream):
                    for layer_id in range(self.transfer_layer_id_max):
                        self.on_layer_done(layer_id)
            except BaseException as error:
                self.fail(error)

        try:
            with device_module.stream(self.stream):
                self.engine._record_stream(self.transfers + self._prepared, self.stream)
        except BaseException as error:
            self.fail(error)

        try:
            with device_module.stream(self.stream):
                self.finish_event.record()
        except BaseException as error:
            self.fail(error)
        finally:
            for domain in reversed(self._domains):
                try:
                    domain.release_layout(
                        self.finish_event,
                        (
                            id(self.engine),
                            "host_to_device" if self.loading else "device_to_host",
                        ),
                    )
                except BaseException as error:
                    self.fail(error)

    def _load_layers(self, transfers: list[Any]) -> None:
        primary = transfers[0] if transfers else None
        for layer_id in range(self.transfer_layer_id_max):
            for transfer in transfers:
                local_layer_id = (
                    transfer.layer_mapper(layer_id)
                    if transfer.layer_mapper is not None
                    else layer_id
                )
                if local_layer_id is None or (
                    transfer is not primary
                    and transfer.layer_mapper is None
                    and layer_id >= transfer.host_pool.layer_num
                ):
                    continue
                transfer.host_pool.load_to_device_per_layer_physical(
                    transfer.device_pool,
                    transfer.host_indices,
                    transfer.device_indices,
                    local_layer_id,
                    self.engine.io_backend,
                    is_draft=transfer.is_draft,
                )

    def _store_pages(self, transfers: list[Any]) -> None:
        for transfer in transfers:
            transfer.host_pool.backup_from_device_all_layer_physical(
                transfer.device_pool,
                transfer.host_indices,
                transfer.device_indices,
                self.engine.io_backend,
            )


class AsyncL2TransferEngine:
    """Run Furiosa DMA asynchronously and record SGLang events after completion."""

    def __init__(self, engine: L2TransferEngine) -> None:
        self._engine = engine
        self.io_backend = engine.io_backend
        self.device_to_host_stream = engine.device_to_host_stream
        self.host_to_device_stream = engine.host_to_device_stream
        self._lock = Lock()
        self._operations: set[_TransferOperation] = set()
        self._device_to_host_lane = _TransferLane("device-to-host")
        try:
            self._host_to_device_lane = _TransferLane("host-to-device")
        except BaseException:
            self._device_to_host_lane.shutdown()
            raise
        self._closed = False

    def submit_device_to_host(self, transfers: list[Any]) -> TransferCompletion:
        return self._submit(
            transfers,
            lane=self._device_to_host_lane,
            stream=self.device_to_host_stream,
            start_event=None,
            loading=False,
        )

    def submit_host_to_device(
        self,
        transfers: list[Any],
        *,
        transfer_layer_id_max: int,
        start_event=None,
        on_layer_done: Callable[[int], None] | None = None,
    ) -> TransferCompletion:
        if transfers and transfer_layer_id_max <= 0:
            raise ValueError("a host-to-device transfer needs at least one layer")
        return self._submit(
            transfers,
            lane=self._host_to_device_lane,
            stream=self.host_to_device_stream,
            start_event=start_event,
            loading=True,
            transfer_layer_id_max=transfer_layer_id_max,
            on_layer_done=on_layer_done,
        )

    def _submit(
        self,
        transfers: list[Any],
        *,
        lane: _TransferLane,
        stream: Any,
        start_event: Any,
        loading: bool,
        transfer_layer_id_max: int = 0,
        on_layer_done: Callable[[int], None] | None = None,
    ) -> TransferCompletion:
        with self._lock:
            if self._closed:
                raise RuntimeError("SGLang Furiosa transfer engine is shut down")
            start_event = self._engine._start_event(start_event)
            start, finish, timing_enabled = make_timing_event_pair()
            ready = device_module.Event()
            with device_module.stream(stream):
                # Preserve SGLang's producer and load-fence ordering before Furiosa DMA.
                start_event.wait(stream)
                ready.record()
            operation = None
            try:
                operation = _TransferOperation(
                    self._engine,
                    transfers,
                    stream,
                    ready,
                    start,
                    finish,
                    loading=loading,
                    transfer_layer_id_max=transfer_layer_id_max,
                    on_layer_done=on_layer_done,
                    on_finish=self._forget,
                )
                self._operations.add(operation)
                lane.submit(operation)
            except BaseException:
                if operation is not None:
                    self._operations.discard(operation)
                raise
        return TransferCompletion(
            _CompletionEvent(start, operation),
            _CompletionEvent(finish, operation),
            timing_enabled,
        )

    def _forget(self, operation: _TransferOperation) -> None:
        with self._lock:
            self._operations.discard(operation)

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            operations = tuple(self._operations)
        self._device_to_host_lane.shutdown()
        self._host_to_device_lane.shutdown()
        first_error = None
        for operation in operations:
            try:
                operation.synchronize()
            except BaseException as error:
                first_error = _combine_errors(first_error, error)
        if first_error is not None:
            raise first_error


def _combine_errors(first: BaseException | None, second: BaseException) -> BaseException:
    if first is None:
        return second
    return BaseExceptionGroup("multiple cache transfer failures", [first, second])
