
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import polars as pl
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUDIT_FILE = PROJECT_ROOT / "reports" / "workflow_audit.jsonl"
SIMULATION_AUDIT_FILE = PROJECT_ROOT / "reports" / "simulation_audit.jsonl"
VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transformed"
    / "selected_v2"
    / "validation.parquet"
)

SIMULATION_AUDIT_LOCK = Lock()

app = FastAPI(
    title="Agentic Cyber Defense API",
    version="1.0.0",
    description=(
        "Dashboard API for the existing Transformer "
        "and MCP defense workflow."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

_workflow = None
_workflow_lock = Lock()


class AnalyzeRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description=(
            "One network flow containing the model's "
            "required numeric features."
        ),
    )


class SimulationRequest(BaseModel):
    predicted_attack: str = Field(
        ...,
        min_length=1,
        max_length=64,
    )
    response_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
    )
    outcome: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )
    real_action_executed: bool = False


def get_workflow():
    global _workflow

    if _workflow is None:
        with _workflow_lock:
            if _workflow is None:
                from src.agents.cyber_defense_workflow import (
                    CyberDefenseWorkflow,
                )

                _workflow = CyberDefenseWorkflow()

    return _workflow


def make_json_safe(value: Any):
    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]

    if isinstance(value, np.generic):
        return make_json_safe(value.item())

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, float) and not math.isfinite(value):
        return None

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "service": "Agentic Cyber Defense API",
        "model_workflow": "existing CyberDefenseWorkflow",
        "response_actions_enabled": False,
    }


@app.get("/api/audit")
def get_audit(limit: int = 100):
    if limit < 1 or limit > 500:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 500.",
        )

    records = []

    audit_files = [
        AUDIT_FILE,
        SIMULATION_AUDIT_FILE,
    ]

    # TEMPORARY DEBUG OUTPUT: inspect the paths used by this API.
    print(
        "AUDIT FILES:",
        [str(path.resolve()) for path in audit_files],
    )
    print(
        "EXISTS:",
        [path.exists() for path in audit_files],
    )

    try:
        for audit_file in audit_files:
            if not audit_file.exists():
                continue

            with audit_file.open("r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue

                    try:
                        record = json.loads(line)
                        if isinstance(record, dict):
                            records.append(record)
                    except json.JSONDecodeError:
                        # Skip incomplete or malformed audit lines.
                        continue

    except OSError:
        raise HTTPException(
            status_code=500,
            detail="Unable to read audit records.",
        )

    # ISO-8601 UTC timestamps sort chronologically when normalized.
    records.sort(
        key=lambda record: str(
            record.get("timestamp_utc", "")
        ),
        reverse=True,
    )

    records = records[:limit]

    return {
        "count": len(records),
        "records": make_json_safe(records),
    }


@app.post("/api/analyze")
def analyze_flow(request: AnalyzeRequest):
    if not request.features:
        raise HTTPException(
            status_code=422,
            detail="The features object cannot be empty.",
        )

    for name, value in request.features.items():
        if not math.isfinite(value):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Feature '{name}' must contain "
                    "a finite numeric value."
                ),
            )

    try:
        result = get_workflow().analyze(request.features)
        return make_json_safe(result)

    except (ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )

    except Exception:
        # Keep internal paths and stack traces out of API responses.
        raise HTTPException(
            status_code=500,
            detail=(
                "Workflow analysis failed. Check the backend "
                "terminal for details."
            ),
        )


@app.post("/api/simulate-response")
def simulate_response(request: SimulationRequest):
    # Real response actions remain disabled.
    if request.real_action_executed:
        raise HTTPException(
            status_code=400,
            detail="Real response actions are disabled.",
        )

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event_type": "SIMULATED_RESPONSE",
        "predicted_attack": request.predicted_attack.strip().upper(),
        "response_name": request.response_name.strip(),
        "outcome": request.outcome.strip(),
        "simulation_status": "SIMULATED",
        "real_action_executed": False,
        "verification": {
            "verified": True,
            "checks": [
                "Simulation-only endpoint used",
                "No real response action executed",
            ],
        },
    }

    try:
        SIMULATION_AUDIT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with SIMULATION_AUDIT_LOCK:
            with SIMULATION_AUDIT_FILE.open(
                "a",
                encoding="utf-8",
            ) as file:
                file.write(
                    json.dumps(make_json_safe(record)) + "\n"
                )

    except OSError:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to save the simulation audit record."
            ),
        )

    return make_json_safe(record)


@app.get("/api/flows")
def get_validation_flows():
    if not VALIDATION_FILE.exists():
        raise HTTPException(
            status_code=500,
            detail="Validation dataset was not found.",
        )

    try:
        df = pl.read_parquet(VALIDATION_FILE)

        if "Attack_Type" not in df.columns:
            raise HTTPException(
                status_code=500,
                detail="Attack_Type column is missing.",
            )

        samples = []

        labels = (
            df.get_column("Attack_Type")
            .unique()
            .sort()
            .to_list()
        )

        for label in labels:
            group = df.filter(
                pl.col("Attack_Type") == label
            ).head(2)

            for row in group.to_dicts():
                features = {
                    key: float(value)
                    for key, value in row.items()
                    if key != "Attack_Type"
                }

                if not all(
                    math.isfinite(value)
                    for value in features.values()
                ):
                    continue

                samples.append({
                    "id": len(samples) + 1,
                    "actual_label": label,
                    "features": features,
                })

        return {
            "count": len(samples),
            "feature_count": 58,
            "samples": samples,
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to load validation samples. "
                "Check the backend terminal."
            ),
        )
