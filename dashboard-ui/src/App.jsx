import { useCallback, useEffect, useState } from "react";
import "./App.css";
import AttackReplay from "./components/AttackReplay";

const API = "http://127.0.0.1:8000";

const NAV = [
  { id: "overview", label: "Overview", icon: "" },
  { id: "investigation", label: "Threat Investigation", icon: "" },
  { id: "workflow", label: "Agent Workflow", icon: "" },
  { id: "active-defense", label: "Active Defense", icon: "" },
  { id: "attack-simulation", label: "Attack Simulation", icon: "" },
  { id: "intelligence", label: "Threat Intelligence", icon: "" },
  { id: "audit", label: "Audit Records", icon: "" },
];

function pretty(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value);
}
function getAttackName(sample) {
  return sample?.actual_label ||
    sample?.label ||
    sample?.attack_type ||
    "Unknown";
}
function normalizeLabel(value) {
  const label = String(value ?? "")
    .trim()
    .toUpperCase()
    .replace(/[\s-]+/g, "_");

  const aliases = {
    DDOS: "DDOS",
    DOS: "DOS",
    BRUTE_FORCE: "BRUTE_FORCE",
    PORT_SCAN: "PORT_SCAN",
    WEB_ATTACK: "WEB_ATTACK",
    BENIGN: "BENIGN",
    BOT: "BOT",
  };

  return aliases[label] || label;
}
function formatTimestamp(value) {
  if (!value) return "—";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) return String(value);

  return date.toLocaleString();
}

function ResultCard({ title, children }) {
  return (
    <section className="panel">
      <h3>{title}</h3>
      {children}
    </section>
  );
}

function getDecisionClass(decision) {
  const normalized = String(decision).toUpperCase();

  if (normalized === "INVESTIGATE") return "decision-investigate";
  if (normalized === "MONITOR") return "decision-monitor";

  return "decision-default";
}

function isSimulationRecord(record) {
  return record?.event_type === "SIMULATED_RESPONSE";
}

function getRecordAttack(record) {
  const recordPlan = record?.plan || {};

  return (
    recordPlan.predicted_attack ||
    record?.predicted_attack ||
    record?.attack_type ||
    record?.event ||
    "Workflow event"
  );
}

function getRecordTimestamp(record) {
  return (
    record?.timestamp_utc ||
    record?.timestamp ||
    record?.created_at ||
    ""
  );
}

function getRecordConfidence(record) {
  const confidence = record?.plan?.confidence;

  return typeof confidence === "number"
    ? `${(confidence * 100).toFixed(2)}%`
    : "—";
}

function getRecordDecision(record) {
  if (isSimulationRecord(record)) return "SIMULATED";

  return (
    record?.plan?.decision ||
    record?.verification?.decision ||
    "—"
  );
}

function getRecordPriority(record) {
  return record?.plan?.priority || "—";
}

function getRecordVerification(record) {
  return record?.verification?.verified;
}

function getRecordAction(record) {
  return (
    record?.action_taken ??
    record?.plan?.action_taken ??
    record?.real_action_executed ??
    false
  );
}

function getRecordPermissionText(record) {
  if (isSimulationRecord(record)) {
    return record.real_action_executed === true
      ? "Real action reported"
      : "No real action executed";
  }

  if (record?.safety?.action_permitted === false) {
    return "Execution disabled";
  }

  if (record?.safety?.action_permitted === true) {
    return "Execution permitted";
  }

  if (record?.verification?.response_execution_enabled === false) {
    return "Execution disabled";
  }

  return "Permission not recorded";
}

export default function App() {
  const [page, setPage] = useState("overview");
  const [health, setHealth] = useState(null);
  const [audit, setAudit] = useState([]);
  const [auditError, setAuditError] = useState("");
  const [featuresText, setFeaturesText] = useState("");
  const [samples, setSamples] = useState([]);
  const [selectedSampleId, setSelectedSampleId] = useState("");
  const [samplesLoading, setSamplesLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  // Active Defense simulation state.
  const [simulationBusy, setSimulationBusy] = useState(false);
  const [simulationError, setSimulationError] = useState("");

  // Attack Simulation page state.
  const [attackSimulationBusy, setAttackSimulationBusy] = useState(false);
  const [attackSimulationError, setAttackSimulationError] = useState("");
  const [attackSimulationResult, setAttackSimulationResult] = useState(null);
  const [attackSimulationSampleId, setAttackSimulationSampleId] = useState("");
  const [attackEventCount, setAttackEventCount] = useState("100");

  const selectedSample = samples.find(
    (sample) => String(sample.id) === selectedSampleId
  );

  const loadHealth = useCallback(async () => {
    try {
      const response = await fetch(`${API}/api/health`);

      if (!response.ok) {
        throw new Error("Backend health check failed");
      }

      setHealth(await response.json());
    } catch {
      setHealth(null);
    }
  }, []);

  const loadAudit = useCallback(async () => {
    try {
      const response = await fetch(`${API}/api/audit?limit=500`);

      if (!response.ok) {
        throw new Error("Could not load audit records");
      }

      const data = await response.json();
      const records = Array.isArray(data.records) ? data.records : [];

      // Keep the frontend ordered by timestamp, newest first.
      records.sort((a, b) => {
        const timeA = new Date(getRecordTimestamp(a)).getTime() || 0;
        const timeB = new Date(getRecordTimestamp(b)).getTime() || 0;
        return timeB - timeA;
      });

      setAudit(records);
      setAuditError("");
    } catch {
      setAuditError(
        "Unable to reach the audit API. Check that FastAPI is running."
      );
    }
  }, []);
  
  const loadSamples = useCallback(async () => {
    setSamplesLoading(true);

    try {
      const response = await fetch(`${API}/api/flows`);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Unable to load validation samples.");
      }

      setSamples(
  (data.samples || []).map((sample) => ({
    ...sample,
    actual_label:
      sample.actual_label ||
      sample.label ||
      sample.attack_type ||
      "Unknown",
  }))
);
      setError("");
    } catch (err) {
      setError(err.message || "Unable to load validation samples.");
    } finally {
      setSamplesLoading(false);
    }
  }, []);

  useEffect(() => {
    loadHealth();
    loadAudit();
    loadSamples();
  }, [loadHealth, loadAudit, loadSamples]);

  async function analyze(event) {
    event.preventDefault();
    setBusy(true);
    setError("");
    setResult(null);
    setSimulationError("");

    try {
      const features = JSON.parse(featuresText);

      if (
        !features ||
        Array.isArray(features) ||
        typeof features !== "object"
      ) {
        throw new Error(
          "Enter one JSON object containing numeric model features."
        );
      }

      const response = await fetch(`${API}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Analysis failed");
      }

      setResult(data);
      await loadAudit();
      setPage("workflow");
    } catch (err) {
      setError(
        err instanceof SyntaxError
          ? "Invalid JSON. Paste a valid network-flow feature object."
          : err.message || "Threat analysis failed."
      );
    } finally {
      setBusy(false);
    }
  }

  async function simulateResponse(attack, recommendation) {
    setSimulationBusy(true);
    setSimulationError("");

    const outcome =
      attack === "BENIGN"
        ? "Monitoring recommendation recorded in the simulation."
        : "Proposed response reviewed in simulation; no real mitigation performed.";

    try {
      const response = await fetch(`${API}/api/simulate-response`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          predicted_attack: attack,
          response_name: recommendation.title,
          outcome,
          real_action_executed: false,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Unable to save the simulation.");
      }

      setResult((current) =>
        current
          ? {
              ...current,
              simulated_response: data,
            }
          : current
      );

      await loadAudit();
    } catch (err) {
      setSimulationError(
        err.message || "Unable to connect to the simulation API."
      );
    } finally {
      setSimulationBusy(false);
    }
  }

  
async function runAttackSimulation() {
  if (!attackSimulationSampleId) {
    setAttackSimulationError("Please select a dataset sample.");
    return;
  }

  const eventCount = Number(attackEventCount);

  if (
    !Number.isInteger(eventCount) ||
    eventCount < 1 ||
    eventCount > 1000
  ) {
    setAttackSimulationError(
      "Enter a whole number of simulated events between 1 and 1,000."
    );
    return;
  }

  const chosenSample = samples.find(
    (sample) =>
      String(sample.id) === String(attackSimulationSampleId)
  );

  if (!chosenSample) {
    setAttackSimulationError("Please select a valid dataset sample.");
    return;
  }

  setAttackSimulationBusy(true);
  setAttackSimulationError("");
  setAttackSimulationResult(null);

  try {
    const response = await fetch(
      `${API}/api/attack-simulation/run`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sample_id: Number(attackSimulationSampleId),
          event_count: eventCount,
        }),
      }
    );

    const data = await response.json();

    if (!response.ok) {
      throw new Error(
        data.detail || "Attack simulation failed."
      );
    }

    setAttackSimulationResult(data);
    await loadAudit();
  } catch (err) {
    setAttackSimulationError(
      err.message || "Unable to run attack simulation."
    );
  } finally {
    setAttackSimulationBusy(false);
  }
}


  const analysis = result?.analysis || {};
  const detection = analysis.detection || {};
  const investigation = analysis.investigation || {};
  const knowledge = analysis.mcp_knowledge || {};
  const plan = result?.plan || {};
  const safety = result?.safety || {};
  const verification = result?.verification || {};

 const auditCount = audit.length;

const simulatedAuditCount = audit.filter(
  (record) => record.audit_source === "simulation"
).length;

const workflowAuditCount = audit.filter(
  (record) => record.audit_source === "workflow"
).length;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            A<span>?</span>
          </div>

          <div>
            <strong>CYBER THREAT</strong>
            <small>DEFENSE</small>
          </div>
        </div>

        <div className="nav-caption">SECURITY OPERATIONS</div>

        <nav>
          {NAV.map((item) => (
            <button
              key={item.id}
              className={`nav-item ${page === item.id ? "active" : ""}`}
              onClick={() => setPage(item.id)}
            >
              {item.icon && <span className="nav-icon">{item.icon}</span>}
              {item.label}

              {item.id === "audit" && (
                <span className="nav-count">{auditCount}</span>
              )}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className={`status-indicator ${health ? "online" : "offline"}`}>
            <span className="status-dot" />

            <div>
              <strong>{health ? "API Connected" : "API Offline"}</strong>
              <small>127.0.0.1:8000</small>
            </div>
          </div>

          <div className="safe-note">
            <span>?</span>

            <div>
              <strong>Safe mode enabled</strong>
              <small>Response actions disabled</small>
            </div>
          </div>
        </div>
      </aside>
            <main className="main-content">
        <header className="topbar">
          <div>
            <div className="breadcrumb">Security Operations / {NAV.find((item) => item.id === page)?.label || "Overview"}</div>
            <h1>
              {NAV.find((item) => item.id === page)?.label || "Overview"}
            </h1>
          </div>

          <div className="topbar-actions">
            <span className={`api-pill ${health ? "success" : "danger"}`}>
              <span className="status-dot" />
              {health ? "System Online" : "System Offline"}
            </span>

            <button className="secondary-button" onClick={() => {
              loadHealth();
              loadAudit();
              loadSamples();
            }}>
              Refresh status
            </button>
          </div>
        </header>

        {error && (
          <div className="alert alert-error" role="alert">
            {error}
            <button onClick={() => setError("")} aria-label="Dismiss error">×</button>
          </div>
        )}

        {page === "overview" && (
  <div className="page-content">
    <div className="page-heading">
      <div>
        <div className="eyebrow">SECURITY OPERATIONS / OVERVIEW</div>
        <h1>Security Overview</h1>
        <p>
          Monitor threat detection, investigation workflows, and
          simulated defense activity.
        </p>
      </div>

      <button
        className="secondary-button"
        onClick={() => {
          loadHealth();
          loadAudit();
          loadSamples();
        }}
      >
        Refresh status
      </button>
    </div>

    <section className="welcome-banner">
      <div className="welcome-copy">
        <div className="eyebrow cyan">
          AUTONOMOUS CYBER DEFENSE
        </div>

        <h2>Detect. Investigate. Verify.</h2>

        <p>
          Analyze network-flow evidence, investigate suspicious
          activity, review recommended responses, and verify
          security decisions through a controlled workflow.
          Real mitigation actions remain disabled.
        </p>

        <div className="hero-actions">
          <button
            className="primary-button"
            onClick={() => setPage("investigation")}
          >
            Start investigation
            <span>→</span>
          </button>

          <button
            className="secondary-button"
            onClick={() => setPage("attack-simulation")}
          >
            Open attack simulation
          </button>
        </div>
      </div>

      <div className="radar" aria-label="Decorative security radar">
        <div className="radar-ring" />
        <div className="radar-ring ring-two" />
        <div className="radar-ring ring-three" />
        <div className="radar-cross cross-x" />
        <div className="radar-cross cross-y" />
        <div className="radar-sweep" />
        <div className="radar-center" />
      </div>
    </section>

    <div className="section-title">
      <h2>System status</h2>
      <span className="muted">Live backend data</span>
    </div>

    <div className="metric-grid">
      <div className="metric-card">
        <div className="metric-top">
          API STATUS
          <span className="metric-icon cyan-icon">⌁</span>
        </div>

        <div className="metric-value">
          {health ? "Online" : "Offline"}
        </div>

        <div className="metric-foot">
          Backend health endpoint
        </div>
      </div>

      <div className="metric-card">
        <div className="metric-top">
          AUDIT RECORDS
          <span className="metric-icon purple-icon">▤</span>
        </div>

        <div className="metric-value">{auditCount}</div>

        <div className="metric-foot">
          Records returned by the audit API
        </div>
      </div>

      <div className="metric-card">
        <div className="metric-top">
          MODEL WORKFLOW
          <span className="metric-icon cyan-icon">◎</span>
        </div>

        <div className="metric-value small-value">
          {result ? "Completed" : "Ready"}
        </div>

        <div className="metric-foot">
          {result
            ? "Latest analysis result available"
            : "Run an investigation to begin"}
        </div>
      </div>

      <div className="metric-card">
        <div className="metric-top">
          RESPONSE ACTIONS
          <span className="metric-icon amber-icon">⌑</span>
        </div>

        <div className="metric-value small-value">
          Disabled
        </div>

        <div className="metric-foot">
          Simulation-only response policy
        </div>
      </div>
    </div>

    <div className="content-grid">
      <section className="panel">
        <div className="section-title">
          <h2>Defense workflow</h2>
          <span className="muted">05 stages</span>
        </div>

        <div className="pipeline">
          {[
            ["01", "Telemetry collection", "→"],
            ["02", "Threat detection", "→"],
            ["03", "Investigation and context", "→"],
            ["04", "Response planning", "→"],
            ["05", "Verification and audit", "✓"],
          ].map(([number, title, marker]) => (
            <div className="pipeline-step" key={number}>
              <span className="pipeline-number">{number}</span>
              <span>{title}</span>
              <span className="pipeline-arrow">{marker}</span>
            </div>
          ))}
        </div>

        <button
          className="text-button"
          onClick={() => setPage("workflow")}
        >
          View workflow details →
        </button>
      </section>

      <section className="panel">
        <div className="section-title">
          <h2>Recent audit activity</h2>
          <span className="muted">
            {auditCount} records
          </span>
        </div>

        {auditError ? (
          <p className="empty-state">{auditError}</p>
        ) : audit.length === 0 ? (
          <p className="empty-state">
            No audit records are available yet. Run an analysis
            or simulation to check whether the backend records
            the event.
          </p>
        ) : (
          <div className="recent-list">
            {audit.slice(0, 5).map((record, index) => (
              <div
                className="recent-item"
                key={record.id || index}
              >
                <span className="activity-dot" />

                <div>
                  <strong>{getRecordAttack(record)}</strong>
                  <small>
                    {formatTimestamp(
                      getRecordTimestamp(record)
                    )}
                  </small>
                </div>

                <span className="activity-status">
                  {getRecordDecision(record)}
                </span>
              </div>
            ))}
          </div>
        )}

        <button
          className="text-button"
          onClick={() => setPage("audit")}
        >
          View all audit records →
        </button>
      </section>
    </div>

    <div className="notice">
      <span>ⓘ</span>
      <p>
        <strong>Safety and data integrity:</strong> dashboard
        recommendations and simulation counters do not establish
        that a real attack occurred or that a real mitigation
        action was executed.
      </p>
    </div>
  </div>
)}

        {page === "investigation" && (
          <div className="page-content">
            <section className="page-intro">
              <span className="eyebrow">NETWORK FLOW ANALYSIS</span>
              <h2>Threat Investigation</h2>
              <p>
                Submit a numeric feature object to the existing backend
                detector. The detector and its configured thresholds remain
                unchanged.
              </p>
            </section>

            <div className="two-column">
              <ResultCard title="Analyze network flow">
                <form onSubmit={analyze}>
                  <label className="field-label" htmlFor="sample-select">
                    Validation sample
                  </label>

                  <select
                    id="sample-select"
                    value={selectedSampleId}
                    onChange={(event) => {
                      const id = event.target.value;
                      setSelectedSampleId(id);

                      const sample = samples.find(
                        (item) => String(item.id) === id
                      );

                      if (sample?.features) {
                        setFeaturesText(
                          JSON.stringify(sample.features, null, 2)
                        );
                      }
                    }}
                  >
                    <option value="">
                      {samplesLoading ? "Loading samples..." : "Select a sample"}
                    </option>

                    {samples.map((sample) => (
  <option key={sample.id} value={sample.id}>
    {sample.id} — {getAttackName(sample)}
  </option>
))}
                  </select>

                  <label className="field-label" htmlFor="features-json">
                    Feature JSON
                  </label>

                  <textarea
                    id="features-json"
                    value={featuresText}
                    onChange={(event) => setFeaturesText(event.target.value)}
                    placeholder={'{\n  "Destination Port": 443,\n  "Flow Duration": 1000\n}'}
                    rows={12}
                    spellCheck={false}
                    required
                  />

                  <button
                    type="submit"
                    className="primary-button full-width"
                    disabled={busy}
                  >
                    {busy ? "Analyzing..." : "Run threat analysis"}
                  </button>
                </form>
              </ResultCard>

              <ResultCard title="Analysis guidance">
                <div className="guidance-list">
                  <div>
                    <span className="guidance-number">1</span>
                    <div>
                      <strong>Use the correct features</strong>
                      <p>
                        Submit numeric features that match the model's expected
                        feature names and order.
                      </p>
                    </div>
                  </div>

                  <div>
                    <span className="guidance-number">2</span>
                    <div>
                      <strong>Review the prediction</strong>
                      <p>
                        A predicted category is a model output, not proof that
                        malicious activity occurred.
                      </p>
                    </div>
                  </div>

                  <div>
                    <span className="guidance-number">3</span>
                    <div>
                      <strong>Verify before responding</strong>
                      <p>
                        Recommendations and simulation outcomes do not modify
                        network devices or block real traffic.
                      </p>
                    </div>
                  </div>
                </div>

                {selectedSample && (
                  <div className="sample-summary">
                    <strong>Selected sample</strong>
                    <span>ID: {selectedSample.id}</span>
                    <span>
  Label: {getAttackName(selectedSample)}
</span>
                  </div>
                )}
              </ResultCard>
            </div>

            {result && (
              <ResultCard title="Threat analysis result">
                <div className="metric-grid">
                  <div className="metric-card">
                    <span className="metric-label">PREDICTION</span>
                    <strong>
                      {detection.predicted_attack ||
                        detection.prediction ||
                        analysis.predicted_attack ||
                        "—"}
                    </strong>
                  </div>

                  <div className="metric-card">
                    <span className="metric-label">CONFIDENCE</span>
                    <strong>
                      {typeof detection.confidence === "number"
                        ? `${(detection.confidence * 100).toFixed(2)}%`
                        : "—"}
                    </strong>
                  </div>

                  <div className="metric-card">
                    <span className="metric-label">DECISION</span>
                    <strong>{plan.decision || "—"}</strong>
                  </div>

                  <div className="metric-card">
                    <span className="metric-label">VERIFICATION</span>
                    <strong>
                      {verification.verified === true
                        ? "Verified"
                        : verification.verified === false
                          ? "Not verified"
                          : "—"}
                    </strong>
                  </div>
                </div>

                <div className="detail-list">
                  <div className="detail-row">
                    <span>Priority</span>
                    <strong>{pretty(plan.priority)}</strong>
                  </div>

                  <div className="detail-row">
                    <span>Investigation</span>
                    <strong>{pretty(investigation)}</strong>
                  </div>

                  <div className="detail-row">
                    <span>Threat knowledge</span>
                    <strong>{pretty(knowledge)}</strong>
                  </div>

                  <div className="detail-row">
                    <span>Safety status</span>
                    <strong>{pretty(safety)}</strong>
                  </div>
                </div>

                <div className="notice notice-info">
                  <strong>Safe execution policy</strong>
                  <p>
                    This analysis displays a prediction and recommendation.
                    It does not authorize a real response action.
                  </p>
                </div>

                <button
                  className="secondary-button"
                  onClick={() => setPage("workflow")}
                >
                  View workflow details
                </button>
              </ResultCard>
            )}
          </div>
        )}
                {page === "workflow" && (
          <div className="page-content">
            <section className="page-intro">
              <span className="eyebrow">DECISION TRACE</span>
              <h2>Agent Workflow</h2>
              <p>
                Review the analysis, investigation, decision, safety checks,
                and verification results from the most recent run.
              </p>
            </section>

            {!result ? (
              <ResultCard title="No workflow result">
                <p className="muted">
                  Run a threat analysis first to view the workflow details.
                </p>
                <button
                  className="primary-button"
                  onClick={() => setPage("investigation")}
                >
                  Analyze a network flow
                </button>
              </ResultCard>
            ) : (
              <>
                <div className="timeline">
                  {[
                    {
                      title: "Threat detection",
                      description: "The detector analyzed the submitted features.",
                      value: detection.predicted_attack || detection.prediction || "Prediction recorded",
                    },
                    {
                      title: "Investigation",
                      description: "Investigation evidence and context were processed.",
                      value: investigation.status || "Review the investigation output",
                    },
                    {
                      title: "Decision planning",
                      description: "The workflow produced a response recommendation.",
                      value: plan.decision || "No decision recorded",
                    },
                    {
                      title: "Safety enforcement",
                      description: "The response policy restricts execution to safe options.",
                      value: safety.action_permitted === false
                        ? "Real action not permitted"
                        : "Review policy details",
                    },
                    {
                      title: "Verification",
                      description: "The workflow verification output is shown below.",
                      value: verification.verified === true
                        ? "Verified"
                        : verification.verified === false
                          ? "Verification failed"
                          : "Status unavailable",
                    },
                  ].map((step, index) => (
                    <div className="timeline-item" key={step.title}>
                      <div className="timeline-marker">{index + 1}</div>
                      <div className="timeline-content">
                        <h3>{step.title}</h3>
                        <p>{step.description}</p>
                        <strong>{pretty(step.value)}</strong>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="two-column">
                  <ResultCard title="Decision details">
                    <div className="detail-list">
                      {Object.entries(plan).length ? (
                        Object.entries(plan).map(([key, value]) => (
                          <div className="detail-row" key={key}>
                            <span>{key.replaceAll("_", " ")}</span>
                            <strong>{pretty(value)}</strong>
                          </div>
                        ))
                      ) : (
                        <p className="muted">No decision details returned.</p>
                      )}
                    </div>
                  </ResultCard>

                  <ResultCard title="Verification details">
                    <pre className="json-output">
                      {JSON.stringify(verification, null, 2)}
                    </pre>
                  </ResultCard>
                </div>

                <ResultCard title="Complete analysis output">
                  <pre className="json-output">
                    {JSON.stringify(result, null, 2)}
                  </pre>
                </ResultCard>

                <button
                  className="secondary-button"
                  onClick={() => setPage("investigation")}
                >
                  Analyze another flow
                </button>
              </>
            )}
          </div>
        )}

        {page === "active-defense" && (
          <div className="page-content">
            <section className="page-intro">
              <span className="eyebrow">POLICY-RESTRICTED RESPONSE</span>
              <h2>Active Defense</h2>
              <p>
                Review safe response recommendations and record simulated
                outcomes. No firewall, endpoint, account, or network settings
                are changed by this dashboard.
              </p>
            </section>

            <div className="notice notice-warning">
              <strong>Real response actions are disabled</strong>
              <p>
                The controls below record a simulated response only. They
                do not block traffic, disable accounts, or apply live security
                rules.
              </p>
            </div>

            <div className="metric-grid">
              <div className="metric-card">
                <span className="metric-label">POLICY MODE</span>
                <strong>Safe mode</strong>
                <small>Restricted response decisions</small>
              </div>

              <div className="metric-card">
                <span className="metric-label">EXECUTION</span>
                <strong>Disabled</strong>
                <small>No real mitigation</small>
              </div>

              <div className="metric-card">
                <span className="metric-label">AUDIT TRAIL</span>
                <strong>Enabled</strong>
                <small>Simulated events may be recorded</small>
              </div>
            </div>

            <ResultCard title="Recommended response options">
              <div className="response-options">
                {[
                  {
                    attack: "DDoS",
                    title: "Simulated rate limiting",
                    description: "Illustrates reducing the allowed event rate.",
                  },
                  {
                    attack: "DoS",
                    title: "Simulated traffic filtering",
                    description: "Illustrates filtering suspicious events.",
                  },
                  {
                    attack: "BRUTE_FORCE",
                    title: "Simulated login guard",
                    description: "Illustrates a login-protection response.",
                  },
                  {
                    attack: "PORT_SCAN",
                    title: "Simulated source restriction",
                    description: "Illustrates a source-blocking recommendation.",
                  },
                  {
                    attack: "BOT",
                    title: "Simulated traffic filtering",
                    description: "Illustrates a bot-traffic response.",
                  },
                  {
                    attack: "WEB_ATTACK",
                    title: "Manual review",
                    description: "Requests analyst review instead of automatic mitigation.",
                  },
                  {
                    attack: "BENIGN",
                    title: "Monitor only",
                    description: "Records a monitoring recommendation.",
                  },
                ].map((option) => (
                  <div className="response-option" key={option.attack}>
                    <div>
                      <span className="option-tag">{option.attack}</span>
                      <h3>{option.title}</h3>
                      <p>{option.description}</p>
                    </div>

                    <button
                      className="secondary-button"
                      disabled={simulationBusy}
                      onClick={() =>
                        simulateResponse(option.attack, option)
                      }
                    >
                      {simulationBusy ? "Recording..." : "Simulate response"}
                    </button>
                  </div>
                ))}
              </div>
            </ResultCard>

            {simulationError && (
              <div className="alert alert-error" role="alert">
                {simulationError}
              </div>
            )}

            {result?.simulated_response && (
              <ResultCard title="Latest simulated response">
                <pre className="json-output">
                  {JSON.stringify(result.simulated_response, null, 2)}
                </pre>
              </ResultCard>
            )}
          </div>
        )}

        
{page === "attack-simulation" && (
  <div className="page-content">
    <section className="page-intro">
      <span className="eyebrow">CONTROLLED SIMULATION</span>
      <h2>Attack Simulation</h2>
      <p>
        Select a labeled dataset sample and run the simulation
        to visualize detection, investigation, simulated response,
        and verification.
      </p>
    </section>

    <div className="notice notice-info">
      <strong>Dataset-based simulation only</strong>
      <p>
        This animation is illustrative. It does not send network
        packets, execute an attack, block real traffic, or change
        network settings.
      </p>
    </div>

    <ResultCard title="Simulation configuration">
      <div className="attack-sim-controls">
        <div className="attack-sim-field">
  <label htmlFor="attack-sample">
    Dataset sample
  </label>
          <select
            id="attack-sample"
            value={attackSimulationSampleId}
            onChange={(event) => {
              setAttackSimulationSampleId(event.target.value);
              setAttackSimulationResult(null);
              setAttackSimulationError("");
            }}
          >
            <option value="">
              {samplesLoading ? "Loading samples..." : "Select a sample"}
            </option>

            {samples.map((sample) => (
              <option key={sample.id} value={sample.id}>
                {sample.id} — {getAttackName(sample)}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="field-label" htmlFor="attack-count">
            Number of simulated events
          </label>

          <input
            id="attack-count"
            type="number"
            min="1"
            max="1000"
            step="1"
            value={attackEventCount}
            onChange={(event) =>
              setAttackEventCount(event.target.value)
            }
          />
        </div>
      </div>

      <button
        className="primary-button"
        disabled={
          attackSimulationBusy ||
          !attackSimulationSampleId ||
          !attackEventCount ||
          !Number.isInteger(Number(attackEventCount)) ||
          Number(attackEventCount) < 1 ||
          Number(attackEventCount) > 1000
        }
        onClick={runAttackSimulation}
      >
        {attackSimulationBusy
          ? "Running simulation..."
          : "Run attack simulation"}
      </button>

      {attackSimulationError && (
        <div className="alert alert-error" role="alert">
          {attackSimulationError}
        </div>
      )}
    </ResultCard>

    {attackSimulationResult && (
      <>
        <AttackReplay
  result={attackSimulationResult}
  eventCount={Number(attackEventCount)}
/>

        <div className="metric-grid">
          <div className="metric-card">
            <span className="metric-label">ACTUAL LABEL</span>
            <strong>
              {pretty(
                attackSimulationResult.simulation?.actual_dataset_label ??
                attackSimulationResult.actual
              )}
            </strong>
          </div>

          <div className="metric-card">
            <span className="metric-label">PREDICTED LABEL</span>
            <strong>
              {pretty(
                attackSimulationResult.simulation?.predicted_label ??
                attackSimulationResult.predicted
              )}
            </strong>
          </div>

          <div className="metric-card">
            <span className="metric-label">DETECTION MATCH</span>
            <strong>
              {attackSimulationResult.simulation?.detection_match === true
                ? "Match"
                : attackSimulationResult.simulation?.detection_match === false
                  ? "Mismatch"
                  : "—"}
            </strong>
          </div>

          <div className="metric-card">
            <span className="metric-label">SIMULATION STATUS</span>
            <strong>
              {pretty(
                attackSimulationResult.simulation?.simulation_status
              )}
            </strong>
          </div>
        </div>

        {attackSimulationResult.simulation?.detection_match === false && (
          <div className="notice notice-warning">
            <strong>Detection mismatch</strong>
            <p>
              The predicted label differs from the dataset label.
              The simulated response uses the reported prediction,
              so this must not be presented as a correct detection.
            </p>
          </div>
        )}

        <ResultCard title="Simulation timeline">
          <div className="timeline">
            {[
              [
                "Sample selected",
                `Sample ${attackSimulationSampleId} — ${
                  attackSimulationResult.simulation?.actual_dataset_label ||
                  "Unknown"
                }`,
              ],
              [
                "Prediction evaluated",
                `${
                  attackSimulationResult.simulation?.predicted_label ||
                  "Unknown"
                } — ${
                  attackSimulationResult.simulation?.detection_match === true
                    ? "Match"
                    : attackSimulationResult.simulation?.detection_match === false
                      ? "Mismatch"
                      : "Awaiting result"
                }`,
              ],
              [
                "Response simulated",
                `${
                  attackSimulationResult.simulation?.response?.rule_name ||
                  attackSimulationResult.simulation?.action ||
                  "Pending"
                }`,
              ],
              [
                "Verification performed",
                `${
                  attackSimulationResult.simulation?.verification?.verified === true
                    ? "Passed"
                    : attackSimulationResult.simulation?.verification?.verified === false
                      ? "Failed"
                      : "Awaiting result"
                } — no real action executed`,
              ],
            ].map(([title, description], index) => (
              <div className="timeline-item" key={title}>
                <div className="timeline-marker">{index + 1}</div>
                <div className="timeline-content">
  <h3>{title}</h3>
  <p>{description}</p>
</div>
              </div>
            ))}
          </div>
        </ResultCard>

        <ResultCard title="Simulated response details">
          <div className="detail-list">
            <div className="detail-row">
              <span>Action</span>
              <strong>
                {pretty(attackSimulationResult.simulation?.action)}
              </strong>
            </div>

            <div className="detail-row">
              <span>Response rule</span>
              <strong>
                {pretty(
                  attackSimulationResult.simulation?.response?.rule_name
                )}
              </strong>
            </div>

            <div className="detail-row">
              <span>Outcome</span>
              <strong>
                {pretty(
                  attackSimulationResult.simulation?.response?.outcome
                )}
              </strong>
            </div>

            <div className="detail-row">
              <span>Simulation rule active</span>
              <strong>
                {pretty(
                  attackSimulationResult.simulation?.response?.simulated_rule_active
                )}
              </strong>
            </div>

            <div className="detail-row">
              <span>Real action executed</span>
              <strong>
                {attackSimulationResult.simulation?.real_action_executed === true
                  ? "Yes"
                  : "No"}
              </strong>
            </div>
          </div>

          {attackSimulationResult.simulation?.response?.description && (
            <p className="muted">
              {attackSimulationResult.simulation.response.description}
            </p>
          )}
        </ResultCard>

        <ResultCard title="Simulated event counts">
          <div className="attack-sim-metrics">
            {[
              ["Total events", attackSimulationResult.simulation?.after?.events_total],
              ["Allowed", attackSimulationResult.simulation?.after?.events_allowed],
              ["Filtered", attackSimulationResult.simulation?.after?.events_filtered],
              ["Rate limited", attackSimulationResult.simulation?.after?.events_rate_limited],
              ["Login blocked", attackSimulationResult.simulation?.after?.events_login_blocked],
              ["Source blocked", attackSimulationResult.simulation?.after?.events_source_blocked],
              ["Manual review", attackSimulationResult.simulation?.after?.events_manual_review],
            ].map(([label, value]) => (
              <div className="attack-sim-metric" key={label}>
                <span>{label}</span>
                <strong>{pretty(value ?? 0)}</strong>
              </div>
            ))}
          </div>
        </ResultCard>

        <div className="notice notice-info">
          <strong>Simulation-only result</strong>
          <p>
            These counters represent simulator output, not confirmed
            changes to real network traffic.
          </p>
        </div>

        <ResultCard title="Verification results">
          <div className="detail-list">
            {Object.entries(
              attackSimulationResult.simulation?.verification || {}
            ).map(([key, value]) => (
              <div className="detail-row" key={key}>
                <span>{key.replaceAll("_", " ")}</span>
                <strong>{pretty(value)}</strong>
              </div>
            ))}
          </div>
        </ResultCard>

        <ResultCard title="Complete simulation output">
          <pre className="json-output">
            {JSON.stringify(attackSimulationResult, null, 2)}
          </pre>
        </ResultCard>

        <button
          className="secondary-button"
          onClick={() => {
            setAttackSimulationResult(null);
            setAttackSimulationError("");
          }}
        >
          Reset simulation result
        </button>
      </>
    )}
  </div>
)}


        {page === "intelligence" && (
          <div className="page-content">
            <section className="page-intro">
              <span className="eyebrow">THREAT CONTEXT</span>
              <h2>Threat Intelligence</h2>
              <p>
                Review threat knowledge returned by the latest investigation.
                This page displays available results without inventing
                intelligence that the backend did not return.
              </p>
            </section>

            {!result ? (
              <ResultCard title="No threat intelligence result">
                <p className="muted">
                  Run an investigation to view any threat knowledge returned
                  by the existing workflow.
                </p>

                <button
                  className="primary-button"
                  onClick={() => setPage("investigation")}
                >
                  Go to Threat Investigation
                </button>
              </ResultCard>
            ) : (
              <>
                <ResultCard title="Threat knowledge">
                  <pre className="json-output">
                    {JSON.stringify(knowledge, null, 2)}
                  </pre>
                </ResultCard>

                <ResultCard title="Investigation evidence">
                  <pre className="json-output">
                    {JSON.stringify(investigation, null, 2)}
                  </pre>
                </ResultCard>
              </>
            )}
          </div>
        )}

        {page === "audit" && (
          <div className="page-content">
            <section className="page-intro">
              <span className="eyebrow">TRACEABILITY</span>
              <h2>Audit Records</h2>
              <p>
                Review workflow and simulation records returned by the backend.
                Simulation entries are not evidence of real mitigation.
              </p>
            </section>

            <div className="metric-grid">
              <div className="metric-card">
                <span className="metric-label">TOTAL RECORDS</span>
                <strong>{auditCount}</strong>
              </div>

              <div className="metric-card">
                <span className="metric-label">WORKFLOW RECORDS</span>
                <strong>{workflowAuditCount}</strong>
              </div>

              <div className="metric-card">
                <span className="metric-label">SIMULATION RECORDS</span>
                <strong>{simulatedAuditCount}</strong>
              </div>

              <div className="metric-card">
                <span className="metric-label">REAL ACTIONS</span>
                <strong>Disabled</strong>
              </div>
            </div>

            {auditError && (
              <div className="alert alert-error" role="alert">
                {auditError}
              </div>
            )}

            <div className="table-panel">
              <div className="table-heading">
                <div>
                  <h3>Security activity</h3>
                  <p>Newest records appear first.</p>
                </div>

                <button
                  className="secondary-button"
                  onClick={loadAudit}
                >
                  Refresh records
                </button>
              </div>

              {audit.length === 0 ? (
                <div className="empty-state">
                  <h3>No audit records found</h3>
                  <p>
                    Run an analysis or simulation to generate records, if the
                    backend supports recording that event.
                  </p>
                </div>
              ) : (
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Timestamp</th>
                        <th>Event</th>
                        <th>Decision</th>
                        <th>Priority</th>
                        <th>Confidence</th>
                        <th>Verification</th>
                        <th>Execution</th>
                      </tr>
                    </thead>

                    <tbody>
                      {audit.map((record, index) => {
                        const verified = getRecordVerification(record);

                        return (
                          <tr key={record.id || `${getRecordTimestamp(record)}-${index}`}>
                            <td>{formatTimestamp(getRecordTimestamp(record))}</td>

                            <td>
                              <strong>{getRecordAttack(record)}</strong>
                              {isSimulationRecord(record) && (
                                <span className="table-subtext">
                                  Simulation only
                                </span>
                              )}
                            </td>

                            <td>
                              <span
                                className={`decision-badge ${getDecisionClass(
                                  getRecordDecision(record)
                                )}`}
                              >
                                {getRecordDecision(record)}
                              </span>
                            </td>

                            <td>{pretty(getRecordPriority(record))}</td>

                            <td>{getRecordConfidence(record)}</td>

                            <td>
                              {verified === true
                                ? "Verified"
                                : verified === false
                                  ? "Failed"
                                  : "—"}
                            </td>

                            <td>
                              <span
                                className={`execution-badge ${
                                  getRecordAction(record)
                                    ? "execution-warning"
                                    : "execution-safe"
                                }`}
                              >
                                {getRecordAction(record)
                                  ? "Action reported"
                                  : getRecordPermissionText(record)}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            <ResultCard title="Audit record details">
              <p className="muted">
                The complete records below are the data returned by the audit
                endpoint. Review the raw fields when investigating a mismatch
                between the displayed summary and the backend record.
              </p>

              <pre className="json-output">
                {JSON.stringify(audit, null, 2)}
              </pre>
            </ResultCard>
          </div>
        )}
      </main>
    </div>
  );
}