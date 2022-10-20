from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Generator
from typing import Iterable
from typing import Optional
from typing import Protocol
from typing import TypeVar
from typing import Union

import pytest
from _pytest.fixtures import Config
from _pytest.python import Function

from avalon.contrib.utils import TEMP_DIR
from avalon.contrib.utils import create_temp_file_path
from avalon.contrib.utils import temp_dir

if TYPE_CHECKING:
    # noinspection PyProtectedMember
    from _pytest.fixtures import _Scope

try:
    # noinspection PyProtectedMember
    from _pytest.fixtures import _FixtureFunction
except Exception:
    raise

T = TypeVar("T")


def fixture(
    fixture_function: Optional[_FixtureFunction] = None,
    *,
    scope: "Union[_Scope, Callable[[str, Config], _Scope]]" = "function",
    params: Optional[Iterable[object]] = None,
    autouse: bool = False,
    ids: Optional[
        Union[
            Iterable[Union[None, str, float, int, bool]],
            Callable[[Any], Optional[object]],
        ]
    ] = None,
) -> Any:
    def decorator(function: _FixtureFunction) -> Any:
        true_name = function.__name__[:-1]
        return pytest.fixture(function, name=true_name, scope=scope, params=params, autouse=autouse, ids=ids)

    if fixture_function is not None and callable(fixture_function):
        return decorator(fixture_function)

    return decorator


def use(*args: Callable[..., Any]) -> Any:
    true_names = [x.__name__[:-1] for x in args]
    return pytest.mark.usefixtures(*true_names)


def integration_test(function: Callable[..., None]) -> Any:
    return pytest.mark.integration_test(function)


def slow_integration_test(function: Callable[..., None]) -> Any:
    return pytest.mark.slow_integration_test(function)


class RequestFixture(Protocol[T]):
    """Yes, there is a class called FixtureRequest, but the types are quite bad for it"""

    node: Function
    param: T


@fixture
def temp_file_path_() -> Generator[Path, None, None]:
    with create_temp_file_path() as output:
        yield output


@fixture
def temp_path_() -> Generator[Path, None, None]:
    with temp_dir(TEMP_DIR) as output:
        yield output
