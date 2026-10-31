"""The KV a transfer backend receives in a token pool's host image belongs to one row allocation
and loads once."""


def test_received_kv_belongs_to_one_row_allocation() -> None:
    """KV a row received and no load took does not outlive the row: the next request allocated
    the row holds none."""
    from furiosa_opt_sglang.cache.image import Received
    from furiosa_opt_sglang.paging import Requests

    requests, received = Requests(4, 64, {}), Received()
    requests.register_on_alloc_rows(received.allocate)
    allocated = requests.alloc_rows(1)
    assert allocated is not None
    received.receive(allocated[0], [1, 2])
    requests.free_rows(allocated)
    assert requests.alloc_rows(1) == allocated
    assert received.take(allocated) == []


def test_received_kv_loads_once() -> None:
    """A load takes the pages each row received, a row's own, once."""
    from furiosa_opt_sglang.cache.image import Received
    from furiosa_opt_sglang.paging import Requests

    requests, received = Requests(4, 64, {}), Received()
    requests.register_on_alloc_rows(received.allocate)
    allocated = requests.alloc_rows(3)
    assert allocated is not None
    first, second, third = allocated
    received.receive(first, [1, 2])
    received.receive(third, [4])
    assert received.take([second, third, first]) == [1, 2, 4]
    assert received.take(allocated) == []
