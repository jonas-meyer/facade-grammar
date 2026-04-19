"""Entry point for ``uv run sam-service`` / ``python -m sam_service``.

``import torch`` must happen before ``LitServer(...)`` so LitServe's MPS
detection sees it (checks ``"torch" in sys.modules``).
"""

import litserve as ls
import torch  # noqa: F401

from sam_service.api import SamAPI
from sam_service.config import ServiceConfig


def main() -> None:
    cfg = ServiceConfig()
    server = ls.LitServer(
        SamAPI(),
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        workers_per_device=cfg.workers_per_device,
        timeout=cfg.request_timeout,
    )
    server.run(host=cfg.host, port=cfg.port, generate_client_file=False)


if __name__ == "__main__":
    main()
