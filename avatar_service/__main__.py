import logging
import os

from aiohttp import web

from .config import Settings
from .server import build_app


def main() -> None:
    logging.basicConfig(level=os.environ.get("AVATAR_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("aioice").setLevel(logging.WARNING)
    s = Settings()
    web.run_app(build_app(s), host=s.host, port=s.port, access_log=None)


if __name__ == "__main__":
    main()
