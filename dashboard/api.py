
import json
import logging
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

from src.simulation.attack_simulator import AttackSimulator


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUDIT_FILE = PROJECT_ROOT / "reports" / "workflow_audit.jsonl"
SIMULATION_AUDIT_FILE = (
    PROJECT_ROOT / "reports" / "simulation_audit.jsonl"
)
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


class AttackSimulationRequest(BaseModel):
    sample_id: int = Field(
        ...,
        ge=1,
        description="ID of a sample returned by /api/flows.",
    )
    event_count: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Number of simulated events.",
    )


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


def load_validation_samples():
    """Load a small, consistent set of samples from each class."""
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

        labels = (
            df.get_column("Attack_Type")
            .unique()
            .sort()
            .to_list()
        )

        samples = []

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
                    "actual_label": label,
                    "features": features,
                })

        return samples

    except HTTPException:
        raise

    except Exception:
        logger.exception("Unable to load validation samples.")
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to load validation samples. "
                "Check the backend terminal."
            ),
        )


def append_simulation_audit(record: dict):
    """Append one record to the simulation audit log."""
    try:
        SIMULATION_AUDIT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        serialized_record = json.dumps(make_json_safe(record))

        with SIMULATION_AUDIT_LOCK:
            with SIMULATION_AUDIT_FILE.open(
                "a",
                encoding="utf-8",
            ) as file:
                file.write(serialized_record + "\n")

    except OSError:
        logger.exception("Unable to write simulation audit record.")
        raise HTTPException(
            status_code=500,
            detail="Unable to save the simulation audit record.",
        )


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "service": "Agentic Cyber Defense API",
        "model_workflow": "existing CyberDefenseWorkflow",
        "response_actions_enabled": False,
        "simulation_only": True,
    }


@app.get("/api/audit")
def get_audit(limit: int = 500):
    if limit < 1 or limit > 500:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 500.",
        )

    records = []
    file_stats = []

    audit_files = [
        ("workflow", AUDIT_FILE),
        ("simulation", SIMULATION_AUDIT_FILE),
    ]

    try:
        for source, audit_file in audit_files:
            valid_count = 0
            invalid_count = 0

            if not audit_file.exists():
                file_stats.append({
                    "source": source,
                    "path": str(audit_file.resolve()),
                    "exists": False,
                    "valid_records": 0,
                    "invalid_records": 0,
                })
                continue

            with audit_file.open("r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue

                    try:
                        record = json.loads(line)

                        if isinstance(record, dict):
                            record.setdefault("audit_source", source)
                            records.append(record)
                            valid_count += 1
                        else:
                            invalid_count += 1

                    except json.JSONDecodeError:
                        invalid_count += 1

            file_stats.append({
                "source": source,
                "path": str(audit_file.resolve()),
                "exists": True,
                "valid_records": valid_count,
                "invalid_records": invalid_count,
            })

    except OSError:
        raise HTTPException(
            status_code=500,
            detail="Unable to read audit records.",
        )

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
        "file_stats": file_stats,
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
        logger.exception("Workflow analysis failed.")
        raise HTTPException(
            status_code=500,
            detail=(
                "Workflow analysis failed. Check the backend "
                "terminal for details."
            ),
        )


@app.post("/api/simulate-response")
def simulate_response(request: SimulationRequest):
    # Explicitly reject requests claiming a real action was executed.
    if request.real_action_executed:
        raise HTTPException(
            status_code=400,
            detail="Real response actions are disabled.",
        )

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event_type": "MANUAL_SIMULATED_RESPONSE",
        "predicted_attack": request.predicted_attack,
        "response_name": request.response_name,
        "outcome": request.outcome,
        "execution_mode": "SIMULATION_ONLY",
        "real_action_executed": False,
    }

    append_simulation_audit(record)
    return make_json_safe(record)


@app.get("/api/flows")
def get_validation_flows():
    samples = load_validation_samples()

    response_samples = [
        {
            "id": index,
            "actual_label": sample["actual_label"],
            "features": sample["features"],
        }
        for index, sample in enumerate(samples, start=1)
    ]

    feature_count = (
        len(response_samples[0]["features"])
        if response_samples
        else 0
    )

    return {
        "count": len(response_samples),
        "feature_count": feature_count,
        "samples": response_samples,
    }


@app.post("/api/attack-simulation/run")
def run_attack_simulation(request: AttackSimulationRequest):
    samples = load_validation_samples()

    if request.sample_id > len(samples):
        raise HTTPException(
            status_code=404,
            detail="Simulation sample ID was not found.",
        )

    sample = samples[request.sample_id - 1]

    try:
        workflow_result = get_workflow().analyze(
            sample["features"]
        )

        analysis = workflow_result.get("analysis", {})
        detection = analysis.get("detection", {})
        plan = workflow_result.get("plan", {})
        safety = workflow_result.get("safety", {})

        predicted_label = str(
            detection.get("predicted_attack", "UNKNOWN")
        )

        workflow_decision = str(
            plan.get("decision", "INVESTIGATE")
        )

        # Preserve the real-response safety policy.
        action_permitted = (
            safety.get("action_permitted") is True
        )

        # This enables only illustrative simulator behavior.
        # It does not change action_permitted or the workflow policy.
        simulation_only = True

        simulator_result = AttackSimulator().run(
            attack_label=str(sample["actual_label"]),
            predicted_label=predicted_label,
            event_count=request.event_count,
            workflow_decision=workflow_decision,
            action_permitted=action_permitted,
            simulation_only=simulation_only,
        )

        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event_type": "ATTACK_SIMULATION_RUN",
            "sample_id": request.sample_id,
            "actual_dataset_label": sample["actual_label"],
            "predicted_label": predicted_label,
            "confidence": detection.get("confidence"),
            "workflow_decision": workflow_decision,
            "action_permitted": action_permitted,
            "simulation_only": simulation_only,
            "simulation": simulator_result,
            "real_action_executed": False,
        }

        append_simulation_audit(record)

        return make_json_safe({
            "workflow": workflow_result,
            "simulation": simulator_result,
            "audit_record": record,
        })

    except HTTPException:
        raise

    except (ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )

    except Exception:
        logger.exception(
            "Attack simulation failed for sample_id=%s",
            request.sample_id,
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "Attack simulation failed. Check the backend "
                "terminal for details."
            ),
        )
