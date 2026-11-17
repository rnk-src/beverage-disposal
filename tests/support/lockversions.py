"""Reads package versions from uv.lock and reports drift from stated versions."""

import tomllib


def locked_versions(lock_text):
    return {package["name"]: package["version"] for package in tomllib.loads(lock_text)["package"]}


def drift(stated, locked):
    found = []
    for name, version in stated.items():
        if name not in locked:
            found.append(f"{name}: stated {version}, not in uv.lock")
        elif locked[name] != version:
            found.append(f"{name}: stated {version}, uv.lock has {locked[name]}")
    return found
