"""
The only reason this whole file is here is so that we can set environment vars coorectly when
running via a notebook during a case where we want to use the resulting artifacts (ie, not just debugging)
"""

import os


def is_notebook() -> bool:
    """This specific snippet checks if the file is run in interactive mode."""
    import __main__ as main

    return not hasattr(main, "__file__")


_NOTEBOOK_SECRETS_TIME = None
if is_notebook():
    if os.path.exists("bashenv_secrets.sh"):
        _NOTEBOOK_SECRETS_TIME = os.path.getmtime("bashenv_secrets.sh")
        with open("bashenv_secrets.sh", "r") as infile:
            # TODO: this is pretty awful... but we can change when we move to better secrets
            var_data = infile.read().replace("export ", "", 1).split("\nexport ")
            for entry in var_data:
                key, value = entry.strip().split("=")
                if value.startswith('"'):
                    assert '"' not in value[1:-1], "Dont support escaped quotes, sorry"
                    value = value[1:-1]
                os.environ[key] = value


def is_secrets_file_updated_since_notebook_restart():
    assert is_notebook(), "Only makes sense in notebook"
    assert os.path.exists("bashenv_secrets.sh"), "Expected to find secrets file but did not, that's not good!"
    assert _NOTEBOOK_SECRETS_TIME is not None, "Expected file to have existed when you restarted"
    secrets_time = os.path.getmtime("bashenv_secrets.sh")
    return secrets_time != is_secrets_file_updated_since_notebook_restart
