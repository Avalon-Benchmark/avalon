import os
import subprocess


def test_pytest_environ():
    assert "PYTEST_CURRENT_TEST" in os.environ


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
    result = subprocess.run(
        'grep -E -r --exclude-dir=contrib --exclude-dir=quarantine --include="*.py" "print\\(" | grep -v -E "(/test_|_test.py)" | grep -v "  # script"',
        shell=True,
    )
    assert (
        result.returncode == 1
    ), "Some files appear to have print statements. Please replace with calls to logger.whatever, or end the line with # script, as appropriate"
