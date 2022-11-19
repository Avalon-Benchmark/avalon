import os
import subprocess

import pytest


def test_pytest_environ():
    assert "PYTEST_CURRENT_TEST" in os.environ


@pytest.mark.skip(reason="it really doesn't seem like anyone else is running this...")
def test_code_contains_no_print_statement():
    """
    Please don't use print. User logger.whatever instead
    Reasoning: then:
        - it's actually logged appropriately
        - other people can turn the message on and off as they wish
        - it forces you to think through the right level and which it should be printed
    This directive does not apply for tests or scripts
    In scripts, simply put:
        ...   # script
    after the line with the print on it and it will be ignore by this linter
    """
    exclusions = " ".join([f"--exclude-dir={dirname}" for dirname in ["contrib", "quarantine", "env", "venv"]])
    ignore_test_files_and_notebooks = "grep -v -E '(tests?/|/test_|_test.py|.sync.py)'"
    ignore_lines_marked_with_script_comment = 'grep -v " # script"'
    result = subprocess.run(
        rf'grep -E -r {exclusions} --include="*.py" "(^|\s)print\\(" | {ignore_test_files_and_notebooks} | {ignore_lines_marked_with_script_comment}',
        shell=True,
        capture_output=True,
        text=True,
    )
    is_ok = result.returncode == 1
    assert is_ok, (
        "Some files appear to have print statements. "
        "Please replace with calls to logger.whatever, or end the line with # script, as appropriate. "
        f"Offending lines:\n{result.stdout}"
    )
