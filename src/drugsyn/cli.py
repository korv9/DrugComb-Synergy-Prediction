"""Command-line entry point: ``drugsyn <stage> [--config path]``."""

from __future__ import annotations

import argparse
import logging
import time

from . import (
    build,
    cells,
    download,
    drugs,
    figures,
    ingest,
    model,
    monotherapy,
    report,
    targets,
    warehouse,
)
from .config import Paths, load_config

STAGES = {
    "download": download.run,
    "ingest": ingest.run,
    "monotherapy": monotherapy.run,
    "drugs": drugs.run,
    "cells": cells.run,
    "targets": targets.run,
    "build": build.run,
    "warehouse": warehouse.run,
    "train": model.run,
    "figures": figures.run,
    "report": report.run,
}
PIPELINES = {
    "data": ["download", "ingest", "monotherapy", "drugs", "cells", "targets", "build",
             "warehouse"],
    "all": list(STAGES),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="drugsyn", description=__doc__)
    parser.add_argument("stage", choices=[*STAGES, *PIPELINES])
    parser.add_argument("--config", default=None, help="YAML file overriding configs/default.yaml")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    paths = Paths.from_config(cfg)
    for stage in PIPELINES.get(args.stage, [args.stage]):
        t0 = time.time()
        logging.getLogger("drugsyn").info("=== %s ===", stage)
        STAGES[stage](cfg, paths)
        logging.getLogger("drugsyn").info("=== %s done in %.1fs ===", stage, time.time() - t0)


if __name__ == "__main__":
    main()
