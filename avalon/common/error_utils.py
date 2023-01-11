import os
from typing import Dict
from typing import Optional
from typing import Sequence

import sentry_sdk
from sentry_sdk.integrations import Integration


def _is_sentry_enabled() -> bool:
    return "SENTRY_DSN" in os.environ


# Sentry only needs to be initialized once per process
_IS_SENTRY_SETUP: bool = False


def setup_sentry(
    percent_of_errors_to_capture: float = 1.0,
    percent_of_traces_to_capture: float = 0.0,
    is_attaching_stacktraces: bool = True,
    tags: Optional[Dict[str, str]] = None,
    integrations: Sequence[Integration] = tuple(),
) -> None:
    """setup sentry using standard SENTRY_DSN environment variable[^1]

    Arguments are mostly slightly more literate passthroughs for sentry_sdk.init,
    except tags, which will be added with setnry_sdk.set_tag if provided.

    [^1]: https://docs.sentry.io/platforms/python/guides/wsgi/configuration/options/#dsn
    """
    if not _is_sentry_enabled():
        return
    # NOTE: sentry-sdk 1.12.1 resolves this typing issue but conflicts with other dependencies
    sentry_sdk.init(  # type: ignore[abstract]
        sample_rate=percent_of_errors_to_capture,
        traces_sample_rate=percent_of_traces_to_capture,
        attach_stacktrace=is_attaching_stacktraces,
        integrations=integrations,
    )
    if tags is None:
        return
    for key, value in tags.items():
        sentry_sdk.set_tag(key, value)

    global _IS_SENTRY_SETUP
    _IS_SENTRY_SETUP = True


def capture_message(message: str, extra: Optional[Dict[str, str]] = None) -> None:
    if not _is_sentry_enabled():
        return

    if not _IS_SENTRY_SETUP:
        setup_sentry(tags=extra)

    with sentry_sdk.push_scope() as scope:
        if extra is not None:
            for key, value in extra.items():
                scope.set_extra(key, value)
        sentry_sdk.capture_message(message, scope=scope)


def capture_exception(
    exception: BaseException,
    extra: Optional[Dict[str, str]] = None,
    extra_fingerprint: Optional[str] = None,
    is_thrown_without_sentry: bool = True,
):
    if not _is_sentry_enabled():
        if not is_thrown_without_sentry:
            return
        raise exception

    if not _IS_SENTRY_SETUP:
        setup_sentry(tags=extra)

    with sentry_sdk.push_scope() as scope:
        if extra is not None:
            for key, value in extra.items():
                scope.set_extra(key, value)
        if extra_fingerprint is not None:
            scope.fingerprint = ["{{ default }}", extra_fingerprint]
        sentry_sdk.capture_exception(exception, scope=scope)
