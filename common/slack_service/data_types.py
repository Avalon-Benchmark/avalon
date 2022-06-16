from typing import Optional

import attr


@attr.s(auto_attribs=True, frozen=True)
class SlackResponse:
    message_timestamp: Optional[str] = None
