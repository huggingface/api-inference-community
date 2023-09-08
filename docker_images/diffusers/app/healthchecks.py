"""
This file allows users to spawn some side service helping with giving a better view on the main ASGI app status.
The issue with the status route of the main application is that it gets unresponsive as soon as all workers get busy.
Thus, you cannot really use the said route as a healthcheck to decide whether your app is healthy or not.
Instead this module allows you to distinguish between a dead service (not able to even tcp connect to app port)
and a busy one (able to connect but not to process a trivial http request in time) as both states should result in
different actions (restarting the service vs scaling it). It also exposes some data to be
consumed as custom metrics, for example to be used in autoscaling decisions.
"""

import asyncio
import functools
import logging
import os
from collections import namedtuple
from typing import Optional

import aiohttp
import psutil
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route


logger = logging.getLogger(__name__)


METRICS = ""
STATUS_OK = 0
STATUS_BUSY = 1
STATUS_ERROR = 2


def metrics():
    logging.debug("Requesting metrics")
    return METRICS


async def metrics_route(_request: Request) -> Response:
    return Response(content=metrics())


routes = [
    Route("/{whatever:path}", metrics_route),
]

app = Starlette(routes=routes)


def reset_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="healthchecks - %(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )


@app.on_event("startup")
async def startup_event():
    reset_logging()
    # Link between `api-inference-community` and framework code.
    asyncio.create_task(compute_metrics_loop(), name="compute_metrics")


@functools.lru_cache()
def get_listening_port():
    logger.debug("Get listening port")
    main_app_port = os.environ.get("MAIN_APP_PORT", "80")
    try:
        main_app_port = int(main_app_port)
    except ValueError:
        logger.warning(
            "Main app port cannot be converted to an int, skipping and defaulting to 80"
        )
        main_app_port = 80
    return main_app_port


async def find_app_process(
    listening_port: int,
) -> Optional[namedtuple("addr", ["ip", "port"])]:  # noqa
    connections = psutil.net_connections()
    app_laddr = None
    for c in connections:
        if c.laddr.port != listening_port:
            logger.debug("Skipping listening connection bound to excluded port %s", c)
            continue
        if c.status == psutil.CONN_LISTEN:
            logger.debug("Found LISTEN conn %s", c)
            candidate = c.pid
            try:
                p = psutil.Process(candidate)
            except psutil.NoSuchProcess:
                continue
            if p.name() == "gunicorn":
                logger.debug("Found gunicorn process %s", p)
                app_laddr = c.laddr
                break

    return app_laddr


def count_current_conns(app_port: int) -> str:
    estab = []
    conns = psutil.net_connections()

    # logger.debug("Connections %s", conns)

    for c in conns:
        if c.status != psutil.CONN_ESTABLISHED:
            continue
        if c.laddr.port == app_port:
            estab.append(c)
    current_conns = len(estab)
    logger.debug("Established connections %d", current_conns)

    curr_conns_str = """# HELP inference_app_established_conns Established connection count for a given app.
# TYPE inference_app_established_conns gauge
inference_app_established_conns{{port="{:d}"}} {:d}
""".format(
        app_port, current_conns
    )
    return curr_conns_str


async def status_with_timeout(
    listening_port: int, app_laddr: Optional[namedtuple("addr", ["ip", "port"])]  # noqa
) -> str:
    logger.debug("Checking application status")

    status = STATUS_OK

    if not app_laddr:
        status = STATUS_ERROR
    else:
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=0.5)
            ) as session:
                url = "http://{}:{:d}/".format(app_laddr.ip, app_laddr.port)
                async with session.get(url) as resp:
                    status_code = resp.status
                    status_text = await resp.text()
                logger.debug("Status code %s and text %s", status_code, status_text)
                if status_code != 200 or status_text != '{"ok":"ok"}':
                    status = STATUS_ERROR
        except asyncio.TimeoutError:
            logger.debug("Asgi app seems busy, unable to reach it before timeout")
            status = STATUS_BUSY
        except Exception as e:
            logger.exception(e)
            status = STATUS_ERROR

    status_str = """# HELP inference_app_status Application health status (0: ok, 1: busy, 2: error).
# TYPE inference_app_status gauge
inference_app_status{{port="{:d}"}} {:d}
""".format(
        listening_port, status
    )

    return status_str


async def single_metrics_compute():
    global METRICS
    listening_port = get_listening_port()
    app_laddr = await find_app_process(listening_port)
    current_conns = count_current_conns(listening_port)
    status = await status_with_timeout(listening_port, app_laddr)

    # Assignment is atomic, we should be safe without locking
    METRICS = current_conns + status

    # Persist metrics to the local ephemeral as well
    metrics_file = os.environ.get("METRICS_FILE")
    if metrics_file:
        with open(metrics_file) as f:
            f.write(METRICS)


@functools.lru_cache()
def get_polling_sleep():
    logger.debug("Get polling sleep interval")
    sleep_value = os.environ.get("METRICS_POLLING_INTERVAL", 10)
    try:
        sleep_value = float(sleep_value)
    except ValueError:
        logger.warning(
            "Unable to cast METRICS_POLLING_INTERVAL env value %s to float. Defaulting to 10.",
            sleep_value,
        )
        sleep_value = 10.0
    return sleep_value


@functools.lru_cache()
def get_initial_delay():
    logger.debug("Get polling initial delay")
    sleep_value = os.environ.get("METRICS_INITIAL_DELAY", 30)
    try:
        sleep_value = float(sleep_value)
    except ValueError:
        logger.warning(
            "Unable to cast METRICS_INITIAL_DELAY env value %s to float. "
            "Defaulting to 30.",
            sleep_value,
        )
        sleep_value = 30.0
    return sleep_value


async def compute_metrics_loop():
    initial_delay = get_initial_delay()

    await asyncio.sleep(initial_delay)

    polling_sleep = get_polling_sleep()
    while True:
        await asyncio.sleep(polling_sleep)
        try:
            await single_metrics_compute()
        except Exception as e:
            logger.error("Something wrong occurred while computing metrics")
            logger.exception(e)


if __name__ == "__main__":
    reset_logging()
    try:
        single_metrics_compute()
        logger.info("Metrics %s", metrics())
    except Exception as exc:
        logging.exception(exc)
        raise
