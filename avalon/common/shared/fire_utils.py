import sys

from fire import helptext


def make_fire_more_friendly(line_length: int = 200) -> None:
    # make fire stop truncating so many things
    helptext.LINE_LENGTH = line_length

    # this makes fire do what I would expect if you pass --help into the base command
    if len(sys.argv) == 2 and sys.argv[-1] in ("--help", "-h"):
        sys.argv.pop()
