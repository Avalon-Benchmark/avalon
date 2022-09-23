import os
from typing import Dict
from typing import Optional

import sentry_sdk


def _is_sentry_enabled():
    return "SENTRY_DSN" in os.environ


def setup_sentry(tags: Optional[Dict[str, str]] = None):
    sentry_dsn = os.environ.get("SENTRY_DSN", "")
    if sentry_dsn:
        sentry_sdk.init(sentry_dsn, traces_sample_rate=0.0, attach_stacktrace=True)  # type: ignore
        if tags:
            for key, value in tags.items():
                sentry_sdk.set_tag(key, value)


def complain(message: str, extra: Optional[Dict[str, str]] = None):
    if not _is_sentry_enabled():
        return

    setup_sentry(tags=extra)
    with sentry_sdk.push_scope() as scope:
        if extra is not None:
            for key, value in extra.items():
                scope.set_extra(key, value)
        sentry_sdk.capture_message(message, scope=scope)


def capture_exception(
    exception: BaseException, extra: Optional[Dict[str, str]] = None, extra_fingerprint: Optional[str] = None
):
    if not _is_sentry_enabled():
        raise exception

    setup_sentry(tags=extra)
    with sentry_sdk.push_scope() as scope:
        if extra is not None:
            for key, value in extra.items():
                scope.set_extra(key, value)
        if extra_fingerprint is not None:
            scope.fingerprint = ["{{ default }}", extra_fingerprint]
        sentry_sdk.capture_exception(exception, scope=scope)
