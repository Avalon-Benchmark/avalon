from typing import Sequence
from typing import Tuple

from common.imports import tqdm
from common.log_utils import is_tqdm_impossible


# annoyingly, this largest_request_size variable is only required for this particular call
# and only to work around a bit of a bug in tqdm in notebooks. Without this, the console
# versions can size themselves correctly, but the notebook one cannot (it does not know
# how large the progress bar should be). This figures out, ahead of time, how many
# videos we expect to see, although some requests may have fewer.
def create_progress_bars(num_bars: int, largest_request_size: int) -> Tuple[tqdm, ...]:
    return tuple(
        [
            tqdm(
                total=largest_request_size,
                disable=is_tqdm_impossible(),
                dynamic_ncols=True,
            )
            for _ in range(0, num_bars)
        ]
    )


def destroy_progress_bars(progress_bars: Sequence[tqdm]):
    for progress_bar in progress_bars:
        destroy_progress_bar(progress_bar)
        del progress_bar


def destroy_progress_bar(progress_bar: tqdm):
    progress_bar.__exit__(None, None, None)
    progress_bar.close()
    if hasattr(progress_bar, "container"):
        try:
            progress_bar.container.close()
        except AttributeError:
            progress_bar.container.visible = False


_CURRENT_PROGRESS_BARS: Tuple[tqdm, ...] = tuple()


def create_progress_bars_for_child_processes(num_bars: int, largest_request_size: int):
    global _CURRENT_PROGRESS_BARS
    _CURRENT_PROGRESS_BARS = create_progress_bars(num_bars, largest_request_size)
    for progress in _CURRENT_PROGRESS_BARS:
        progress.__enter__()


def get_child_progress_bar(index: int) -> tqdm:
    global _CURRENT_PROGRESS_BARS
    return _CURRENT_PROGRESS_BARS[index]


def destroy_progress_bars_for_child_processes():
    global _CURRENT_PROGRESS_BARS
    destroy_progress_bars(_CURRENT_PROGRESS_BARS)
    _CURRENT_PROGRESS_BARS = tuple()
