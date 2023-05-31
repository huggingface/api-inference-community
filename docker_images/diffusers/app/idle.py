import asyncio
import contextlib
import logging
import os
import signal
import time


LOG = logging.getLogger(__name__)

LAST_START = None
LAST_END = None

UNLOAD_IDLE = os.getenv("UNLOAD_IDLE", "").lower() in ("1", "true")
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", 15))


async def live_check_loop():
    global LAST_START, LAST_END

    pid = os.getpid()

    LOG.debug("Starting live check loop")

    while True:
        await asyncio.sleep(IDLE_TIMEOUT)
        LOG.debug("Checking whether we should unload anything from gpu")

        last_start = LAST_START
        last_end = LAST_END

        LOG.debug("Checking pid %d activity", pid)
        if not last_start:
            continue
        if not last_end or last_start >= last_end:
            LOG.debug("Request likely being processed for pid %d", pid)
            continue
        now = time.time()
        last_request_age = now - last_end
        LOG.debug("Pid %d, last request age %s", pid, last_request_age)
        if last_request_age < IDLE_TIMEOUT:
            LOG.debug("Model recently active")
        else:
            LOG.debug("Inactive for too long. Leaving live check loop")
            break
    LOG.debug("Aborting this worker")
    os.kill(pid, signal.SIGTERM)


@contextlib.contextmanager
def request_witnesses():
    global LAST_START, LAST_END
    # Simple assignment, concurrency safe, no need for any lock
    LAST_START = time.time()
    try:
        yield
    finally:
        LAST_END = time.time()
