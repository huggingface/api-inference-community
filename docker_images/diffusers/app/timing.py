import logging
from functools import wraps
from time import time


logger = logging.getLogger(__name__)


def timing(f):
    @wraps(f)
    def inner(*args, **kwargs):
        start = time()
        try:
            ret = f(*args, **kwargs)
        finally:
            end = time()
            logger.debug("Func: %r took: %.2f sec to execute", f.__name__, end - start)
        return ret

    return inner
