"""Which top-level entries git would publish."""


def unexpected_top_level(files, allowed):
    top_level = {name.split("/")[0] for name in files}
    return sorted(top_level - set(allowed))


def explain_unexpected(names):
    return (
        f"git would publish top-level entries that are not on the allowlist: {names}. "
        "If one is a new project file, add it to the allowlist. If it is private notes, your "
        "local exclude rules are missing on this clone: recreate them, then check again."
    )
