"""Finds text that identifies a person or a machine."""

import re
from pathlib import Path

HOME_PATH = r"/(?:Users|home)/[^/\s]+|[A-Za-z]:\\Users\\[^\\\s]+"
# The lookahead skips SSH remotes (user@host:owner/repo and ssh://user@host/owner/repo). It also
# stops the domain from backtracking to a shorter one: without that, the last letter of the
# domain would be dropped and the remote would match anyway.
EMAIL = r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w-]|\.\w|[:/][\w~/.])"


def find_private_looking_text(text: str) -> list[str]:
    return re.findall(f"{HOME_PATH}|{EMAIL}", text)


def find_leaks_in_files(root, files, exempt=()):
    leaks = {}
    for name in files:
        if name in exempt:
            continue
        data = (Path(root) / name).read_bytes()
        if b"\x00" in data[:8000]:
            continue
        found = find_private_looking_text(data.decode(errors="ignore"))
        if found:
            leaks[name] = found
    return leaks
