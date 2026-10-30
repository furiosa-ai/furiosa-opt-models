"""Engine objects serve any thread, since a framework loads its model on one thread and runs its
requests on another."""

from concurrent.futures import ThreadPoolExecutor


def test_object_serves_and_drops_on_another_thread() -> None:
    import furiosa_opt_engine as engine

    created = [engine.HostTensor([1, 2, 3])]

    def use() -> object:
        # The only reference leaves the list here, so the object also drops on this thread.
        return created.pop().numpy().tolist()

    with ThreadPoolExecutor(max_workers=1) as thread:
        assert thread.submit(use).result() == [1, 2, 3]
