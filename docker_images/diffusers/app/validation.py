import re


STR_TO_BOOL = re.compile(r"^\s*true|yes|1\s*$", re.IGNORECASE)


def str_to_bool(s):
    return STR_TO_BOOL.match(str(s))
