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


def tests_only(item):
    pipeline = item.get("retained_pipeline_receipt", {})
    steps = pipeline.get("steps", [])
    return (
        pipeline.get("status") == "failed"
        and pipeline.get("failed_step") == "static_contract_audit"
        and [s["name"] for s in steps]
        == ["resource_admission_before_tests", "tests", "static_contract_audit"]
        and [s["exit_code"] for s in steps] == [0, 0, 1]
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
                or not (
                    all(s["conclusion"] == "skipped" for s in guarded)
                    or (
                        tests_only(item)
                        and all(s["conclusion"] == "failure" for s in guarded)
                    )
                )
            ):
                raise RuntimeError("prior attempt not proven pre-execution; no retry")
            accepted.append(item["id"])
    check(
        current, [r for page in pages for r in page["workflow_runs"]], attempt, accepted
    )
    print("PASS_FIRST_DISPATCH_ONLY")
