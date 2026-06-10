from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs" / "automation"
ARCHIVE_DIR = ROOT / "archives"
ALERT_DIR = ROOT / "alerts"

PIPELINE_STEPS = [
    ("update_public_match_data", [sys.executable, "scripts/web_ingest.py"]),
    ("update_video_metadata", [sys.executable, "scripts/video_source_ingest.py"]),
    ("clean_model_data", [sys.executable, "scripts/clean_model_data.py"]),
    ("build_feature_store", [sys.executable, "scripts/build_feature_store.py"]),
    ("validate_model_data", [sys.executable, "scripts/validate_model_data.py"]),
    ("build_tournament_predictions", [sys.executable, "scripts/build_tournament_predictions.py"]),
]

OUTPUT_PATHS = [
    "data/raw",
    "data/cleaned",
    "data/derived",
    "data/data_sources.csv",
    "data/ingestion_summary.csv",
    "data/video_ingestion_summary.csv",
    "data/cleaning_summary.csv",
    "data/feature_engineering_summary.csv",
    "data/trust_report.csv",
    "data/validation_report.csv",
]

SUMMARY_FILES = [
    "data/ingestion_summary.csv",
    "data/video_ingestion_summary.csv",
    "data/cleaning_summary.csv",
    "data/feature_engineering_summary.csv",
    "data/trust_report.csv",
    "data/validation_report.csv",
    "data/derived/prediction_summary.csv",
]


@dataclass
class StepResult:
    name: str
    command: list[str]
    return_code: int
    started_at: str
    ended_at: str
    duration_seconds: float
    stdout_tail: str
    stderr_tail: str


@dataclass
class RunState:
    run_id: str
    attempt: int
    started_at: str
    ended_at: str | None = None
    status: str = "running"
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    data_sources_updated: list[dict[str, Any]] = field(default_factory=list)
    pipeline_output_summary: dict[str, Any] = field(default_factory=dict)
    steps: list[StepResult] = field(default_factory=list)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_line(log_file: Path, message: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"{now_iso()} {message}\n")


def tail(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def copy_path(source: Path, destination: Path) -> None:
    if source.is_dir():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    elif source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def snapshot_outputs(run_id: str) -> Path:
    snapshot_dir = ARCHIVE_DIR / run_id / "previous_success"
    for relative in OUTPUT_PATHS:
        source = ROOT / relative
        if source.exists():
            copy_path(source, snapshot_dir / relative)
    return snapshot_dir


def restore_outputs(snapshot_dir: Path) -> None:
    if not snapshot_dir.exists():
        return
    for relative in OUTPUT_PATHS:
        target = ROOT / relative
        snapshot = snapshot_dir / relative
        remove_path(target)
        if snapshot.exists():
            copy_path(snapshot, target)


def archive_success(run_id: str) -> None:
    archive_dir = ARCHIVE_DIR / run_id / "successful_output"
    for relative in OUTPUT_PATHS:
        source = ROOT / relative
        if source.exists():
            copy_path(source, archive_dir / relative)


def read_csv_records(relative_path: str) -> list[dict[str, Any]]:
    path = ROOT / relative_path
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    return frame.fillna("").to_dict(orient="records")


def collect_data_sources_updated() -> list[dict[str, Any]]:
    sources = read_csv_records("data/data_sources.csv")
    if not sources:
        return []
    return [
        {
            "dataset": row.get("dataset", ""),
            "source_name": row.get("source_name", ""),
            "local_path": row.get("local_path", ""),
            "rows_saved": row.get("rows_saved", ""),
            "retrieved_on": row.get("retrieved_on", ""),
        }
        for row in sources
    ]


def collect_pipeline_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for relative in SUMMARY_FILES:
        records = read_csv_records(relative)
        if records:
            summary[relative] = records

    probability_path = ROOT / "data/derived/tournament_round_probabilities.csv"
    if probability_path.exists():
        probabilities = pd.read_csv(probability_path)
        top_columns = ["team", "semi_final_probability", "final_probability", "winner_probability"]
        present = [column for column in top_columns if column in probabilities.columns]
        summary["top_predictions"] = probabilities[present].head(8).fillna("").to_dict(orient="records")
    return summary


def validate_reports(state: RunState) -> None:
    validation_path = ROOT / "data/validation_report.csv"
    trust_path = ROOT / "data/trust_report.csv"

    for path, label, status_column in [
        (validation_path, "validation", "status"),
        (trust_path, "trust", "status"),
    ]:
        if not path.exists():
            raise RuntimeError(f"{label} report was not created: {path}")
        report = pd.read_csv(path)
        if status_column not in report.columns:
            raise RuntimeError(f"{label} report is missing required column: {status_column}")
        statuses = report[status_column].astype(str).str.lower()
        fail_count = int((statuses == "fail").sum())
        warn_count = int((statuses == "warn").sum())
        if fail_count:
            failed_rows = report[statuses == "fail"].to_dict(orient="records")
            raise RuntimeError(f"{label} report has {fail_count} failing checks: {failed_rows}")
        if warn_count:
            state.warnings.append(f"{label} report has {warn_count} warning checks")


def run_step(name: str, command: list[str], log_file: Path) -> StepResult:
    started = datetime.now()
    write_line(log_file, f"START step={name} command={' '.join(command)}")
    process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    ended = datetime.now()

    if process.stdout:
        write_line(log_file, f"STDOUT step={name}\n{process.stdout.rstrip()}")
    if process.stderr:
        write_line(log_file, f"STDERR step={name}\n{process.stderr.rstrip()}")
    write_line(log_file, f"END step={name} return_code={process.returncode}")

    result = StepResult(
        name=name,
        command=command,
        return_code=process.returncode,
        started_at=started.astimezone().isoformat(timespec="seconds"),
        ended_at=ended.astimezone().isoformat(timespec="seconds"),
        duration_seconds=round((ended - started).total_seconds(), 3),
        stdout_tail=tail(process.stdout),
        stderr_tail=tail(process.stderr),
    )
    if process.returncode != 0:
        raise RuntimeError(f"Step failed: {name} returned {process.returncode}\n{tail(process.stderr or process.stdout)}")
    return result


def write_run_summary(state: RunState, run_dir: Path) -> None:
    state.ended_at = state.ended_at or now_iso()
    payload = {
        "run_id": state.run_id,
        "attempt": state.attempt,
        "started_at": state.started_at,
        "ended_at": state.ended_at,
        "status": state.status,
        "warnings": state.warnings,
        "errors": state.errors,
        "data_sources_updated": state.data_sources_updated,
        "pipeline_output_summary": state.pipeline_output_summary,
        "steps": [step.__dict__ for step in state.steps],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_path = LOG_DIR / "latest_run.json"
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def send_alert(run_id: str, errors: list[str], warnings: list[str], summary_path: Path) -> None:
    ALERT_DIR.mkdir(parents=True, exist_ok=True)
    alert_payload = {
        "run_id": run_id,
        "status": "failed",
        "sent_at": now_iso(),
        "errors": errors,
        "warnings": warnings,
        "summary_path": str(summary_path),
    }
    alert_json = ALERT_DIR / f"automation_alert_{run_id}.json"
    alert_txt = ALERT_DIR / f"automation_alert_{run_id}.txt"
    alert_json.write_text(json.dumps(alert_payload, indent=2), encoding="utf-8")
    alert_txt.write_text(
        "World Cup automation failed after retry.\n\n"
        f"Run ID: {run_id}\n"
        f"Summary: {summary_path}\n\n"
        "Errors:\n"
        + "\n".join(f"- {error}" for error in errors)
        + "\n\nWarnings:\n"
        + "\n".join(f"- {warning}" for warning in warnings),
        encoding="utf-8",
    )

    webhook_url = os.environ.get("AUTOMATION_ALERT_WEBHOOK_URL", "").strip()
    if webhook_url:
        request = Request(
            webhook_url,
            data=json.dumps(alert_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(request, timeout=20).read()
        except (OSError, URLError) as exc:
            alert_payload["webhook_error"] = str(exc)
            alert_json.write_text(json.dumps(alert_payload, indent=2), encoding="utf-8")


def run_attempt(run_id: str, attempt: int, snapshot_dir: Path) -> RunState:
    run_dir = LOG_DIR / run_id / f"attempt_{attempt}"
    log_file = run_dir / "run.log"
    state = RunState(run_id=run_id, attempt=attempt, started_at=now_iso())
    write_line(log_file, f"RUN START run_id={run_id} attempt={attempt}")

    try:
        for name, command in PIPELINE_STEPS:
            state.steps.append(run_step(name, command, log_file))
        validate_reports(state)
        state.data_sources_updated = collect_data_sources_updated()
        state.pipeline_output_summary = collect_pipeline_summary()
        state.status = "success"
        state.ended_at = now_iso()
        write_line(log_file, f"RUN SUCCESS run_id={run_id} attempt={attempt}")
        archive_success(run_id)
    except Exception as exc:  # noqa: BLE001 - automation must capture full failure context
        state.status = "failed"
        state.errors.append(str(exc))
        state.errors.append(traceback.format_exc())
        state.pipeline_output_summary = collect_pipeline_summary()
        state.ended_at = now_iso()
        write_line(log_file, f"RUN FAILED run_id={run_id} attempt={attempt} error={exc}")
        if attempt == 2:
            restore_outputs(snapshot_dir)
            write_line(log_file, f"RESTORED previous successful outputs from {snapshot_dir}")
    finally:
        write_run_summary(state, run_dir)

    return state


def main() -> int:
    run_id = make_run_id()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_dir = snapshot_outputs(run_id)
    final_state = run_attempt(run_id, 1, snapshot_dir)
    if final_state.status != "success":
        final_state = run_attempt(run_id, 2, snapshot_dir)

    final_summary = LOG_DIR / run_id / f"attempt_{final_state.attempt}" / "run_summary.json"
    if final_state.status != "success":
        send_alert(run_id, final_state.errors, final_state.warnings, final_summary)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
