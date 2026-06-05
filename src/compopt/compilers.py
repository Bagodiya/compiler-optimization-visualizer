"""Figuring out which compilers we can actually use on this machine."""

import shutil

# the ones we know how to drive for now
KNOWN_COMPILERS = ["gcc", "clang"]


def find_compilers() -> list[str]:
    """Return the compilers from KNOWN_COMPILERS that are on PATH.

    Uses shutil.which so we only report compilers we can really run.
    Order follows KNOWN_COMPILERS, gcc first.
    """
    found = []
    for name in KNOWN_COMPILERS:
        if shutil.which(name) is not None:
            found.append(name)
    return found
