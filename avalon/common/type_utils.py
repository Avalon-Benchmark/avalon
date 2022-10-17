from typing import Optional
from typing import TypeVar

T = TypeVar("T")


def assert_not_none(arg: Optional[T]) -> T:
    assert arg is not None
    return arg
