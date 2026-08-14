import asyncio
import logging

from colorlog import ColoredFormatter

from core.bootstrap import ServiceGraph
from core.config_loader import ConfigError, ConfigLoader


def setup_logging() -> logging.Logger:
    _colors = {
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    }

    _formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)-8s] %(blue)s%(name)s%(reset)s: %(message)s",
        datefmt="%H:%M:%S",
        reset=True,
        log_colors=_colors,
    )

    _handler = logging.StreamHandler()
    _handler.setFormatter(_formatter)
    logging.basicConfig(level=logging.INFO, handlers=[_handler], force=True)

    return logging.getLogger(__name__)


async def main() -> None:
    log = setup_logging()

    services = None

    async def _cleanup_after_startup_failure() -> None:
        if services is None:
            return
        try:
            await services.stop()
        except BaseException as cleanup_exc:
            log.warning(
                "启动失败后的资源清理异常 category=%s",
                type(cleanup_exc).__name__,
            )

    try:
        services = ServiceGraph(ConfigLoader())
        await services.build()
        await services.start()
    except ConfigError as exc:
        await _cleanup_after_startup_failure()
        log.critical(str(exc))
        return
    except Exception:
        await _cleanup_after_startup_failure()
        log.critical("启动失败", exc_info=True)
        return

    try:
        await asyncio.Event().wait()
    finally:
        await services.stop()


if __name__ == "__main__":
    asyncio.run(main())
