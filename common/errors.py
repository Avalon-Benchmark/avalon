from typing import Any


class SwitchError(Exception):
    def __init__(self, unmatched_case: Any) -> None:
        super().__init__(f"Failed to match: {unmatched_case}")
