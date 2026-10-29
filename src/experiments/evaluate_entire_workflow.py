"""
End-to-end workflow evaluation for the Cyber Threat Defence project.

Run from the repository root:
    python -m src.experiments.evaluate_entire_workflow

This script:
- Reuses saved held-out detector metrics; it does NOT retrain or reevaluate the model.
- Runs one real workflow scenario per attack class from selected_v2 validation data.
- Separately checks planner/policy handling for simulated MCP statuses.
- Writes a JSON report with detection metrics and scenario-level workflow metrics kept separate.

Important: one sample per class is a smoke/scenario evaluation, not a statistically
representative end-to-end benchmark. Response execution remains disabled.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DETECTOR_REPORT = ROOT / "reports" / "heldout_test_comparison_v1.json"
VALIDATION_DATA = (
    ROOT / "data" / "processed" / "transformed" / "selected_v2" / "validation.parquet"
)
OUTPUT_REPORT = ROOT / "reports" / "end_to_end_evaluation.json"

EXPECTED_CLASSES = [
    "BENIGN",
    "BOT",
    "BRUTE_FORCE",
    "DDoS",
    "DoS",
    "PORT_SCAN",
    "WEB_ATTACK",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_saved_detector_metrics() -> dict[str, Any]:
    if not DETECTOR_REPORT.exists():
        return {
            "status": "missing",
            "source": str(DETECTOR_REPORT.relative_to(ROOT)),
            "note": (
                "Saved detector report was not found. Detector metrics were not "
                "recomputed, and no values have been fabricated."
            ),
        }

    saved = load_json(DETECTOR_REPORT)
    return {
        "status": "loaded",
        "source": str(DETECTOR_REPORT.relative_to(ROOT)),
        "experiment": saved.get("experiment"),
        "dataset": saved.get("dataset"),
        "evaluation": saved.get("evaluation"),
        "test_records": saved.get("test_records"),
        "features": saved.get("features"),
        "thresholds": saved.get("thresholds"),
        "production": saved.get("production"),
        "experiment_v1": saved.get("experiment_v1"),
        "decision": saved.get("decision"),
        "note": (
            "These are saved held-out detector metrics. They are not workflow-level "
            "scores and were not recalculated by this script."
        ),
    }


def select_validation_scenarios() -> list[tuple[str, dict[str, Any]]]:
    if not VALIDATION_DATA.exists():
        raise FileNotFoundError(
            f"Validation data not found: {VALIDATION_DATA.relative_to(ROOT)}"
        )

    frame = pd.read_parquet(VALIDATION_DATA)
    if "Attack_Type" not in frame.columns:
        raise ValueError("Validation parquet does not contain the Attack_Type column.")

    scenarios: list[tuple[str, dict[str, Any]]] = []
    for label in EXPECTED_CLASSES:
        matches = frame.loc[frame["Attack_Type"].astype(str) == label]
        if matches.empty:
            scenarios.append((label, {}))
            continue
        row = matches.iloc[0].drop(labels=["Attack_Type"]).to_dict()
        # Convert NumPy scalar types to ordinary Python values for downstream tools.
        normalized = {
            str(key): (value.item() if hasattr(value, "item") else value)
            for key, value in row.items()
        }
        scenarios.append((label, normalized))
    return scenarios


def expected_decision_for_label(label: str) -> str:
    return "MONITOR" if label == "BENIGN" else "INVESTIGATE"


def run_workflow_scenarios() -> dict[str, Any]:
    from src.agents.cyber_defense_workflow import CyberDefenseWorkflow

    scenarios = select_validation_scenarios()
    workflow = CyberDefenseWorkflow()
    results: list[dict[str, Any]] = []

    for actual_label, flow in scenarios:
        if not flow:
            results.append({
                "actual_label": actual_label,
                "status": "skipped",
                "reason": "No validation row was available for this class.",
            })
            continue

        started = time.perf_counter()
        try:
            result = workflow.analyze(flow)
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            analysis = result.get("analysis") or {}
            detection = analysis.get("detection") or {}
            plan = result.get("plan") or {}
            safety = result.get("safety") or {}
            verification = result.get("verification") or {}
            audit = result.get("audit") or {}
            mcp = analysis.get("mcp_knowledge") or {}

            predicted = detection.get("predicted_attack", "UNKNOWN")
            expected_decision = expected_decision_for_label(actual_label)
            actual_decision = plan.get("decision", "UNKNOWN")
            detection_correct = predicted == actual_label
            decision_correct = actual_decision == expected_decision
            safety_compliant = (
                safety.get("approved") is True
                and safety.get("action_permitted") is False
                and result.get("action_taken") is False
            )
            verification_success = verification.get("verified") is True
            audit_recorded = audit.get("recorded") is True

            # A scenario is an end-to-end pass only if the key stages behave as
            # expected. This is deliberately not called classifier accuracy.
            workflow_success = all([
                detection_correct,
                decision_correct,
                safety_compliant,
                verification_success,
                audit_recorded,
            ])

            results.append({
                "actual_label": actual_label,
                "status": "completed",
                "predicted_label": predicted,
                "confidence": detection.get("confidence"),
                "detection_correct": detection_correct,
                "expected_decision": expected_decision,
                "actual_decision": actual_decision,
                "decision_correct": decision_correct,
                "mcp_status": mcp.get("status", "unknown"),
                "safety_compliant": safety_compliant,
                "verification_success": verification_success,
                "audit_recorded": audit_recorded,
                "action_taken": result.get("action_taken", False),
                "latency_ms": round(elapsed_ms, 3),
                "workflow_success": workflow_success,
            })
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            results.append({
                "actual_label": actual_label,
                "status": "failed",
                "latency_ms": round(elapsed_ms, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

    completed = [row for row in results if row.get("status") == "completed"]
    latencies = [row["latency_ms"] for row in completed if "latency_ms" in row]
    mcp_counts = Counter(str(row.get("mcp_status", "unknown")) for row in completed)

    def rate(field: str) -> float | None:
        if not completed:
            return None
        return round(
            sum(row.get(field) is True for row in completed) / len(completed), 4
        )

    return {
        "evaluation_type": "scenario-level smoke evaluation",
        "validation_dataset": str(VALIDATION_DATA.relative_to(ROOT)),
        "sampling_method": "First available validation row for each expected class",
        "warning": (
            "At most one sample per class. These scenario results are not a "
            "statistically representative benchmark and must not be presented as "
            "full-dataset workflow accuracy."
        ),
        "scenarios_requested": len(results),
        "scenarios_completed": len(completed),
        "scenarios_skipped_or_failed": len(results) - len(completed),
        "workflow_success_rate": rate("workflow_success"),
        "detection_match_rate_on_scenarios": rate("detection_correct"),
        "decision_correctness_rate": rate("decision_correct"),
        "safety_policy_compliance_rate": rate("safety_compliant"),
        "verification_success_rate": rate("verification_success"),
        "audit_recorded_rate": rate("audit_recorded"),
        "mcp_status_counts": dict(mcp_counts),
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 3) if latencies else None,
            "median": round(statistics.median(latencies), 3) if latencies else None,
            "max": round(max(latencies), 3) if latencies else None,
        },
        "response_execution_enabled": False,
        "scenarios": results,
    }


def run_component_resilience_checks() -> dict[str, Any]:
    """Exercise planner/policy branches using simulated MCP outcomes, not live MCP."""
    from src.planner.decision_planner import DecisionPlanner
    from src.safety.response_policy import ResponsePolicy

    planner = DecisionPlanner()
    policy = ResponsePolicy()
    cases = [
        ("found", {"status": "found", "action_taken": False}),
        ("not_found", {"status": "not_found", "action_taken": False}),
        ("error", {"status": "error", "message": "simulated test error", "action_taken": False}),
    ]
    results = []

    for case_name, mcp_knowledge in cases:
        analysis = {
            "detection": {
                "predicted_attack": "DDoS",
                "confidence": 0.9,
                "benign_prediction": False,
            },
            "investigation": {"status": "completed"},
            "mcp_knowledge": mcp_knowledge,
        }
        try:
            plan = planner.plan(analysis)
            safety = policy.evaluate(plan)
            compliant = (
                safety.get("approved") is True
                and safety.get("action_permitted") is False
                and plan.get("action_taken") is False
            )
            results.append({
                "simulated_mcp_case": case_name,
                "decision": plan.get("decision"),
                "safety_approved": safety.get("approved"),
                "action_permitted": safety.get("action_permitted"),
                "action_taken": plan.get("action_taken"),
                "policy_compliant": compliant,
                "status": "completed",
            })
        except Exception as exc:
            results.append({
                "simulated_mcp_case": case_name,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

    completed = [item for item in results if item.get("status") == "completed"]
    compliance_rate = (
        round(sum(item.get("policy_compliant") is True for item in completed) / len(completed), 4)
        if completed else None
    )
    return {
        "evaluation_type": "component resilience checks",
        "cases": len(results),
        "completed": len(completed),
        "policy_compliance_rate": compliance_rate,
        "warning": (
            "MCP outcomes in this section are simulated inputs. They do not measure "
            "live MCP availability or real-world threat-intelligence coverage."
        ),
        "results": results,
    }


def main() -> int:
    generated_at = datetime.now(timezone.utc).isoformat()
    print("=" * 72)
    print("CYBER THREAT DEFENCE — END-TO-END WORKFLOW EVALUATION")
    print("=" * 72)

    detector_metrics = load_saved_detector_metrics()
    if detector_metrics["status"] == "loaded":
        production = detector_metrics.get("production") or {}
        print("\n1. SAVED HELD-OUT DETECTOR METRICS")
        print(f"   Dataset: {detector_metrics.get('dataset')}")
        print(f"   Test records: {detector_metrics.get('test_records')}")
        for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
            value = production.get(key)
            print(f"   {key}: {value if value is not None else 'not available'}")
        print("   (Saved values reused; no model retraining or reevaluation.)")
    else:
        print("\n1. SAVED HELD-OUT DETECTOR METRICS")
        print("   Saved detector report not found; metrics omitted.")

    try:
        workflow_metrics = run_workflow_scenarios()
    except Exception as exc:
        workflow_metrics = {
            "status": "failed_to_run",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    try:
        resilience_metrics = run_component_resilience_checks()
    except Exception as exc:
        resilience_metrics = {
            "status": "failed_to_run",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    report = {
        "report_title": "Cyber Threat Defence — End-to-End Workflow Evaluation",
        "generated_at_utc": generated_at,
        "project_root": str(ROOT),
        "detector_metrics": detector_metrics,
        "workflow_scenario_metrics": workflow_metrics,
        "component_resilience_checks": resilience_metrics,
        "interpretation": {
            "detector_metrics": (
                "Held-out classification metrics loaded from the saved detector report."
            ),
            "workflow_scenario_metrics": (
                "Small, real-workflow scenario checks; rates are not full-dataset "
                "classifier metrics."
            ),
            "component_resilience_checks": (
                "Planner/policy checks with simulated MCP statuses, not live service rates."
            ),
            "response_execution": (
                "Disabled. The workflow may recommend MONITOR, INVESTIGATE, or "
                "MANUAL_REVIEW, but this evaluation does not execute mitigation."
            ),
        },
    }

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_REPORT.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, default=str)

    print("\n2. REAL WORKFLOW SCENARIO METRICS")
    if workflow_metrics.get("status") == "failed_to_run":
        print(f"   Evaluation could not run: {workflow_metrics.get('error')}")
    else:
        for key in (
            "scenarios_completed",
            "workflow_success_rate",
            "detection_match_rate_on_scenarios",
            "decision_correctness_rate",
            "safety_policy_compliance_rate",
            "verification_success_rate",
            "audit_recorded_rate",
        ):
            print(f"   {key}: {workflow_metrics.get(key)}")
        print(f"   latency_ms: {workflow_metrics.get('latency_ms')}")
        print("   Note: scenario-level smoke evaluation, not a full-dataset benchmark.")

    print("\n3. PLANNER/POLICY RESILIENCE (SIMULATED MCP OUTCOMES)")
    if resilience_metrics.get("status") == "failed_to_run":
        print(f"   Checks could not run: {resilience_metrics.get('error')}")
    else:
        print(f"   Completed cases: {resilience_metrics.get('completed')}/{resilience_metrics.get('cases')}")
        print(f"   Policy compliance rate: {resilience_metrics.get('policy_compliance_rate')}")

    print("\n4. REPORT")
    print(f"   Saved: {OUTPUT_REPORT.relative_to(ROOT)}")
    print("   Response execution enabled: False")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
