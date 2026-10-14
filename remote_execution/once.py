"""Fail closed on reruns or any earlier dispatch of this campaign workflow."""

import json
import os
import subprocess
from pathlib import Path


def check(current, runs, attempt, adjudicated=()):
    if str(attempt) != "1":
        raise RuntimeError(
            "reruns forbidden; retain and adjudicate the original attempt"
        )
    if any(
        int(r["id"]) < int(current) and int(r["id"]) not in adjudicated for r in runs
    ):
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
    accepted = []
    path = Path(__file__).with_name("PREEXECUTION_ADJUDICATION.json")
    if path.exists():
        for item in json.loads(path.read_text())["runs"]:
            previous = json.loads(
                subprocess.check_output(
                    ["gh", "api", f"repos/{repo}/actions/runs/{item['id']}"]
                )
            )
            jobs = json.loads(
                subprocess.check_output(
                    ["gh", "api", f"repos/{repo}/actions/runs/{item['id']}/jobs"]
                )
            )
            guarded = [
                step
                for job in jobs["jobs"]
                for step in job["steps"]
                if step["name"]
                == "V2 raw replay, then V3 tests, campaign and raw verification"
            ]
            if (
                previous["head_sha"] != item["head_sha"]
                or previous["status"] != "completed"
                or not guarded
                or any(s["conclusion"] != "skipped" for s in guarded)
            ):
                raise RuntimeError("prior attempt not proven pre-execution; no retry")
            accepted.append(item["id"])
    check(
        current, [r for page in pages for r in page["workflow_runs"]], attempt, accepted
    )
    print("PASS_FIRST_DISPATCH_ONLY")
