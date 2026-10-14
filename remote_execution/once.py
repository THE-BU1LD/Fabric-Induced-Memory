"""Fail closed on reruns or any earlier dispatch of this campaign workflow."""

import json
import os
import subprocess


def check(current, runs, attempt):
    if str(attempt) != "1":
        raise RuntimeError(
            "reruns forbidden; retain and adjudicate the original attempt"
        )
    if any(int(r["id"]) < int(current) for r in runs):
        raise RuntimeError(
            "earlier dispatch exists; campaign must not be silently repeated"
        )


if __name__ == "__main__":
    repo = os.environ["GITHUB_REPOSITORY"]
    current = os.environ["GITHUB_RUN_ID"]
    attempt = os.environ["GITHUB_RUN_ATTEMPT"]
    # Exact current workflow ID; pagination prevents missing older attempts.
    run = json.loads(
        subprocess.check_output(["gh", "api", f"repos/{repo}/actions/runs/{current}"])
    )
    pages = json.loads(
        subprocess.check_output(
            [
                "gh",
                "api",
                "--paginate",
                "--slurp",
                f"repos/{repo}/actions/workflows/{run['workflow_id']}/runs?per_page=100",
            ]
        )
    )
    check(current, [r for page in pages for r in page["workflow_runs"]], attempt)
    print("PASS_FIRST_DISPATCH_ONLY")
