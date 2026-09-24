"""Reads package versions from uv.lock and from the prose that states them."""

import re
import tomllib


def locked_versions(lock_text):
    return {package["name"]: package["version"] for package in tomllib.loads(lock_text)["package"]}


def claude_md_versions(text):
    """`name version` pairs from the "Versions locked <date>:" sentence, which may wrap over lines."""
    sentence = re.search(r"Versions locked [\d-]+:(.*?)\.(?:\s*\n\s*-|\s*\n\n|\s*\Z)", text, re.DOTALL)
    if not sentence:
        raise ValueError("no 'Versions locked' sentence found in CLAUDE.md")
    return dict(re.findall(r"([A-Za-z][\w-]*) (\d+\.\d+\.\d+)", sentence.group(1)))


def drift(stated, locked):
    found = []
    for name, version in stated.items():
        if name not in locked:
            found.append(f"{name}: stated {version}, not in uv.lock")
        elif locked[name] != version:
            found.append(f"{name}: stated {version}, uv.lock has {locked[name]}")
    return found
