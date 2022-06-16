import json
import os
from typing import Any
from typing import Dict
from typing import Optional
from urllib.parse import quote
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter

# Following this https://www.peterbe.com/plog/best-practice-with-retries-with-requests for retries
# leaving this weird important so that requests will always pull the correct package
# noinspection PyUnresolvedReferences
from requests.packages.urllib3.util.retry import Retry

from common.slack_service.constants import SLACK_CHANNEL_BOT_TESTING
from common.slack_service.data_types import SlackResponse
from common.slack_service.utils import make_slack_response

SLACK_API_BASE_URL = "https://slack.com/api"


def requests_retry_session(
    retries: int = 10,
    backoff_factor: float = 0.5,
    status_forcelist: Any = None,
    session: Optional[requests.Session] = None,
) -> requests.Session:
    session = session or requests.Session()
    retry = Retry(
        total=retries, read=retries, connect=retries, backoff_factor=backoff_factor, status_forcelist=status_forcelist
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def get_slack_session() -> requests.Session:
    session = requests_retry_session()
    slack_api_token = os.environ.get("SLACK_API_TOKEN", "")
    session.headers.update(
        {"Authorization": f"Bearer {slack_api_token}", "Content-Type": "application/json; charset=utf-8"}
    )
    return session


def post_text_to_slack_channel(text: str, channel: str = SLACK_CHANNEL_BOT_TESTING) -> SlackResponse:
    return post_message_to_slack_channel({"text": text}, channel)


def post_message_to_slack_channel(message: Dict, channel: str = SLACK_CHANNEL_BOT_TESTING) -> SlackResponse:
    session = get_slack_session()
    data = {**message, "channel": channel, "link_names": True}
    headers = {"content-type": "application/json; charset=utf-8"}
    response = session.post(f"{SLACK_API_BASE_URL}/chat.postMessage", data=json.dumps(data), headers=headers)
    return make_slack_response(response)


def add_emoji_to_message(emoji_name: str, message_ts: str, channel: str) -> SlackResponse:
    session = get_slack_session()
    data = {
        "name": emoji_name,
        "channel": channel,
        "timestamp": message_ts,
    }
    headers = {"content-type": "application/x-www-form-urlencoded"}  # Slack api doesn't accept json...
    response = session.post(f"{SLACK_API_BASE_URL}/reactions.add", data=data, headers=headers)
    return make_slack_response(response)


def upload_snippet_to_slack_channel(
    content: str, filename: str, initial_comment: str, channel: str = SLACK_CHANNEL_BOT_TESTING
) -> SlackResponse:
    session = get_slack_session()
    data = {
        "content": content,
        "filename": filename,
        "channels": channel,
        "initial_comment": initial_comment,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    encoded_data = urlencode(data, quote_via=quote)  # type: ignore
    response = session.post(f"{SLACK_API_BASE_URL}/files.upload", data=encoded_data, headers=headers)
    return make_slack_response(response)
