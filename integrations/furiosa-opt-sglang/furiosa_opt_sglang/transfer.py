"""Bridge engine copy completion into SGLang's native transfer events."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock, Thread, current_thread
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
        on_layer_done,
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
        self.result = _TransferResult()
        self._on_finish = on_finish
        self._thread = Thread(
            target=self._run,
            name="furiosa-sglang-cache-copy",
            daemon=True,
        )

    def synchronize(self) -> None:
        if self._thread is not current_thread():
            self._thread.join()
        self.result.raise_if_failed()

    def _run(self) -> None:
        domains = []
        acquired_domains = []
        opened_batches = []
        prepared = []
        copies = []
        failure = None

        try:
            domains = shared_host_layout_domains(transfer.host_pool for transfer in self.transfers)
            for domain in domains:
                domain.acquire_layout()
                acquired_domains.append(domain)
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
                prepared = self.engine._prepare_transfers(self.transfers)
                self.start_event.record()
                if self.loading:
                    self._load_layers(prepared)
                else:
                    self._store_pages(prepared)
        except BaseException as error:
            failure = error
            try:
                self.stream.synchronize()
            except BaseException as drain_error:
                failure = _combine_errors(failure, drain_error)
        finally:
            for host in opened_batches:
                try:
                    copies.extend(host.finish_async_copy_batch())
                except BaseException as error:
                    failure = _combine_errors(failure, error)
            copy_error = settle_transfers(copies)
            if copy_error is not None:
                failure = _combine_errors(failure, copy_error)

            if failure is None and self.loading and self.on_layer_done is not None:
                try:
                    with device_module.stream(self.stream):
                        for layer_id in range(self.transfer_layer_id_max):
                            self.on_layer_done(layer_id)
                except BaseException as error:
                    failure = error

            try:
                with device_module.stream(self.stream):
                    self.engine._record_stream(self.transfers + prepared, self.stream)
            except BaseException as error:
                failure = _combine_errors(failure, error)

            if failure is not None:
                # Copies are drained before the failure becomes visible to SGLang.
                self.result.fail(failure)
            try:
                with device_module.stream(self.stream):
                    self.finish_event.record()
            except BaseException as error:
                self.result.fail(_combine_errors(failure, error))

            for domain in reversed(acquired_domains):
                try:
                    domain.release_layout(
                        self.finish_event,
                        (id(self.engine), "host_to_device" if self.loading else "device_to_host"),
                    )
                except BaseException as error:
                    self.result.fail(_combine_errors(failure, error))

            self._on_finish(self)

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
        self._closed = False

    def submit_device_to_host(self, transfers: list[Any]) -> TransferCompletion:
        return self._submit(
            transfers,
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
        on_layer_done=None,
    ) -> TransferCompletion:
        return self._submit(
            transfers,
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
        stream: Any,
        start_event: Any,
        loading: bool,
        transfer_layer_id_max: int = 0,
        on_layer_done=None,
    ) -> TransferCompletion:
        start_event = self._engine._start_event(start_event)
        start, finish, timing_enabled = make_timing_event_pair()
        ready = device_module.Event()
        with device_module.stream(stream):
            # Preserve SGLang's producer and load-fence ordering before Furiosa DMA.
            start_event.wait(stream)
            ready.record()

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
        with self._lock:
            if self._closed:
                raise RuntimeError("SGLang Furiosa transfer engine is shut down")
            self._operations.add(operation)
            try:
                operation._thread.start()
            except BaseException:
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
    return RuntimeError(f"{first}; draining submitted cache copies also failed: {second}")
