from __future__ import annotations

import os
import subprocess
import sys

FLOWS = [
    ("flows.scrape_round:daily_scrape_round", "daily-scrape-round", "0 7 * * *"),
    ("flows.reembed_index:nightly_reembed", "nightly-reembed", "30 2 * * *"),
    ("flows.recalibrate:weekly_recalibrate", "weekly-recalibrate", "0 4 * * 1"),
    ("flows.drift_report:drift_report", "drift-report", "0 6 * * *"),
]


def main() -> int:
    work_pool = os.environ.get("PREFECT_WORK_POOL", "default-agent-pool")
    for entrypoint, name, cron in FLOWS:
        cmd = [
            sys.executable, "-m", "prefect", "deploy",
            "--name", name, "--pool", work_pool,
            "--cron", cron, entrypoint,
        ]
        print(" ".join(cmd))
        subprocess.check_call(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
