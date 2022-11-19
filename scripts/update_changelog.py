import re
from collections import defaultdict
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

from fire import Fire

from avalon.common.log_utils import configure_local_logger
from avalon.common.shared.fire_utils import make_fire_more_friendly
from avalon.common.utils import AVALON_PACKAGE_DIR
from avalon.contrib.utils import run_local_command

CHANGELOG_PATH = (Path(AVALON_PACKAGE_DIR) / "../CHANGELOG.md").resolve()
PACKAGE_NAME = "avalon"


class UpdateChangelog:
    def _get_commit_messages_since_last_changelog_update(self, since_commit: Optional[str]) -> List[str]:
        if since_commit is not None:
            get_latest_changelog_update = f"echo '{since_commit}'"
        else:
            just_hash = '--pretty=format:"%h"'
            get_latest_changelog_update = f"git log -1 {just_hash} {CHANGELOG_PATH}"

        delimiter = "##########"
        full_commit_text_with_delimiter = f'--pretty=format:"%B{delimiter}"'
        get_commit_messages_since = f"xargs -I _commit_ git log {full_commit_text_with_delimiter } _commit_^..HEAD"
        delimited_output: str = run_local_command(
            f"{get_latest_changelog_update} | {get_commit_messages_since}"
        ).stdout.decode("UTF-8")
        return delimited_output.split(delimiter)

    def _changelog_regexs(self, text_for_log_group_name: str) -> Dict[str, re.Pattern[str]]:
        changelog_actions: Dict[str, str] = {
            "Added": r"add(s|ed)?",
            "Changed": r"change(s|d)?",
            "Deprecated": r"deprecate(d|s)?",
            "Removed": r"remove(s|d)?",
            "Fixed": r"fix(es|ed)?",
            "Security": r"security",
        }
        flags = re.IGNORECASE | re.MULTILINE
        regexs = {}
        for section_name, pattern in changelog_actions.items():
            prefix = r"^(-|\*| )*"
            indented_wrap_without_bullet = r"\n +[^ -*]"
            extract_text = f"(?P<{text_for_log_group_name}>(.|{indented_wrap_without_bullet})*)"
            full_line_search = re.compile(f"{prefix}{pattern}:? *{extract_text} *$", flags=flags)
            regexs[section_name] = full_line_search
        return regexs

    def _parse_changes_from_messages(self, messages: List[str]) -> Dict[str, List[str]]:
        change_prefix = r"^(#|-|\*| )*"
        flags = re.IGNORECASE | re.MULTILINE
        find_changes_section = re.compile(f"{change_prefix}{PACKAGE_NAME} changes:?$", flags=flags)

        text_for_log_group_name = "text_for_log"
        changelog_patterns = self._changelog_regexs(text_for_log_group_name)
        changes: Dict[str, List[str]] = {k: [] for k in changelog_patterns}

        for commit in reversed(messages):
            find_changes = re.split(find_changes_section, commit.strip(), maxsplit=1)
            if len(find_changes) < 3:
                continue
            avalon_changes = find_changes[2].split("\n\n\n")[0]
            for section_name, full_line_search in changelog_patterns.items():
                for match in re.finditer(full_line_search, avalon_changes):
                    change = match.group(text_for_log_group_name)
                    change = re.sub(r"\n +", "\n  ", change)
                    changes[section_name].append(change)
        return changes

    def _update_changelog_in_place(self, changes: Dict[str, List[str]], changelog_file: Path):
        changelog_actions = ["Added", "Changed", "Deprecated", "Removed", "Fixed", "Security"]
        flags = re.IGNORECASE | re.MULTILINE
        with open(changelog_file, "r+") as changelog:
            main_header, unreleased, *releases = re.split(r"^## ", changelog.read(), flags=flags)
            if not unreleased.startswith("[Unreleased]"):
                releases = [unreleased, *releases]
                unreleased = "[Unreleased]\n"
            unreleased_header, unreleased_section = unreleased.split("\n", maxsplit=1)
            sections = defaultdict(lambda: [])
            unreleased_overview, *section_texts = unreleased_section.split("### ")
            for section_text in section_texts:
                name, rest = section_text.split("\n", maxsplit=1)
                if name not in changelog_actions:
                    continue
                sections[name] = [line.strip() for line in f"\n{rest}".split("\n- ") if line.strip() != ""]
            for section_name, line_items in changes.items():
                if len(line_items) == 0:
                    continue
                sections[section_name] += [line for line in line_items if line not in sections[section_name]]

            rebuilt_sections = "\n\n".join(
                [
                    "\n".join([f"### {name}", *(f"- {item}" for item in line_items)])
                    for name, line_items in sections.items()
                ]
            )
            rebuilt_unreleased = "\n\n".join(
                [
                    subsection
                    for subsection in [
                        unreleased_header,
                        unreleased_overview.strip(),
                        rebuilt_sections,
                    ]
                    if subsection != ""
                ]
            )

            top_level_sections: List[str] = [main_header, rebuilt_unreleased, *releases]
            reconstructed_updated_changelog = "\n\n\n## ".join([section.strip() for section in top_level_sections])
            changelog.seek(0)
            changelog.write(reconstructed_updated_changelog + "\n")

    def from_commit_log(self, since_commit: Optional[str] = None):
        """
        Update the avalon CHANGELOG.md based on commits since it's last edit for manual review and commit.

        The parser recognizes a block starting with a line like "Avalon Changes" (full pattern r"^(#| )*avalon changes:?$"),
        then parses out line items into idiomatic "keep a changelog" sections:
        Added, Changed, Deprecated, Removed, Fixed, Security (https://keepachangelog.com/en/1.0.0/#how).

        The updater only updates the "Unreleased" section, creating one if not present.
        It does not distinguish between merge and regular commits, and will avoid adding duplicate line-items.

        The line item parser accepts any line, starting with a case-insensitive keyword variant (patterns like r"add(s|ed)?" for "add | adds | added),
        and allows leading whitespace or markdown-like bullets - & *, and will include newlines followed by indents for wrapping ("\\n  +[^-* ]").

        Note: numbered lists aren't yet supported.

        Examples:
        ```
        # example 1
        avalon changes:
        * fixed broken thing
        * adds some api
        * update binaries. Long explanation that wraps...
          indents are required for wrapping and will carry over to changelog

        # example 2
        ### Avalon Changes
        This line is ignored as it doesn't have a keyword
        SECURITY: updated vulnerable dependencies.
          This line will be included in the security line item with indentation preserved


        changes on lines following a double-newline (like this one) will be ignored,
          as "\\n\\n" indicates the end of the change block.

        # example 3
        changes in this commit will also be ignored, as they do not begin with a line like "avalon changes:", "# AVALON CHANGES" etc
        fixed something somewhere else
        ```

        :param since_commit: Override the CHANGELOG.md edit detection and collect all changes since the given commit.
        """

        messages_since_last_update = self._get_commit_messages_since_last_changelog_update(since_commit)
        changes = self._parse_changes_from_messages(messages_since_last_update)

        if len(changes) == 0:
            print(f"No changes to log")
            return

        self._update_changelog_in_place(changes, CHANGELOG_PATH)
        print(f"CHANGELOG.md updated! Please review the changes before committing them.")

    def from_literal_message(self, message: str):
        """
        Update CHANGELOG.md with raw message text. See `from_commit_log` for formatting details.
        Message must still contain "avalon changes" sub-header.

        :param message: Message in the form parsable by `from_commit_log`
        """
        changes = self._parse_changes_from_messages([message])

        if len(changes) == 0:
            print(f"No changes to log")
            return

        self._update_changelog_in_place(changes, CHANGELOG_PATH)
        print(f"CHANGELOG.md updated! Please review the changes before committing them.")


if __name__ == "__main__":
    make_fire_more_friendly()
    configure_local_logger()
    Fire(UpdateChangelog)
