#!/usr/bin/env bash
set -eo pipefail

# move up until we get to the correct directory
while [ ! -f ./common/mypy.ini ]; do cd ..; done

# activate the correct env for mypy if we're running from pycharm (tests don't use this)
if [ -f "$2/activate" ]; then
    source "$2/activate"
fi

mypy_targets="$1"

#actually run mypy
(MYPYPATH=. mypy --show-absolute-path --config-file ./common/mypy.ini $mypy_targets | grep -v 'Function is missing a return type annotation' | grep -v 'Use \"-> None\" if function does not return a value' | grep -E "(error:|note:)" | cat ) || echo "Done."
