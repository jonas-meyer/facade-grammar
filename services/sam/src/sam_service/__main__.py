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
    api = SamAPI(max_batch_size=cfg.max_batch_size, batch_timeout=cfg.batch_timeout)
    server = ls.LitServer(api, accelerator="auto", timeout=cfg.request_timeout)
    server.run(host=cfg.host, port=cfg.port, generate_client_file=False)


if __name__ == "__main__":
    main()
