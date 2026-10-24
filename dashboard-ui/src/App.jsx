
import { useCallback, useEffect, useState } from "react";
import "./App.css";

const API = "http://127.0.0.1:8000";

const NAV = [
  { id: "overview", label: "Overview", icon: "◫" },
  { id: "investigation", label: "Threat Investigation", icon: "⌕" },
  { id: "workflow", label: "Agent Workflow", icon: "◎" },
  { id: "active-defense", label: "Active Defense", icon: "⛨" },
  { id: "intelligence", label: "Threat Intelligence", icon: "⌘" },
  { id: "audit", label: "Audit Records", icon: "▤" },
];

function pretty(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value);
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
      const response = await fetch(`${API}/api/audit?limit=100`);

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

      setSamples(data.samples || []);
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

  const analysis = result?.analysis || {};
  const detection = analysis.detection || {};
  const investigation = analysis.investigation || {};
  const knowledge = analysis.mcp_knowledge || {};
  const plan = result?.plan || {};
  const safety = result?.safety || {};
  const verification = result?.verification || {};

  const auditCount = audit.length;
  const simulatedAuditCount = audit.filter(isSimulationRecord).length;
  const workflowAuditCount = auditCount - simulatedAuditCount;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            A<span>◆</span>
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
              <span className="nav-icon">{item.icon}</span>
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
            <span>◈</span>

            <div>
              <strong>Safe mode enabled</strong>
              <small>Response actions disabled</small>
            </div>
          </div>

          <div className="sidebar-version">
            AGENTIC DEFENSE · MINI PROJECT
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumbs">
            Security Operations <span>/</span>{" "}
            {NAV.find((item) => item.id === page)?.label}
          </div>

          <div className="topbar-right">
            <span className="live-pill">
              <span className="status-dot" />
              LOCAL ENVIRONMENT
            </span>

            <div className="avatar">SOC</div>
          </div>
        </header>

        <div className="page-content">
          {/* OVERVIEW */}
          {page === "overview" && (
            <>
              <div className="page-heading">
                <div>
                  <div className="eyebrow">DEFENSE COMMAND CENTER</div>
                  <h1>Security Overview</h1>
                  <p>
                    Monitor your autonomous threat investigation workflow.
                  </p>
                </div>

                <button
                  className="secondary-button"
                  onClick={() => {
                    loadHealth();
                    loadAudit();
                  }}
                >
                  ↻ Refresh status
                </button>
              </div>

              <div className="welcome-banner">
                <div className="welcome-copy">
                  <div className="eyebrow cyan">AGENTIC CYBER DEFENSE</div>

                  <h2>Detect. Investigate. Verify.</h2>

                  <p>
                    Transformer-based network threat detection connected to
                    agent orchestration and MCP threat intelligence.
                  </p>

                  <button
                    className="primary-button"
                    onClick={() => setPage("investigation")}
                  >
                    Start investigation <span>→</span>
                  </button>
                </div>

                <div className="radar">
                  <div className="radar-ring ring-one" />
                  <div className="radar-ring ring-two" />
                  <div className="radar-ring ring-three" />
                  <div className="radar-cross cross-x" />
                  <div className="radar-cross cross-y" />
                  <div className="radar-sweep" />
                  <span className="radar-center" />
                </div>
              </div>

              <div className="section-title">
                <h2>System status</h2>
                <span className="muted">
                  Live API and persisted audit data
                </span>
              </div>

              <div className="metric-grid">
                <div className="metric-card">
                  <div className="metric-top">
                    <span>API STATUS</span>
                    <span className="metric-icon cyan-icon">⌁</span>
                  </div>

                  <div
                    className={`metric-value ${
                      health ? "success-text" : "danger-text"
                    }`}
                  >
                    {health ? "Online" : "Offline"}
                  </div>

                  <div className="metric-foot">
                    {health
                      ? "FastAPI is responding"
                      : "Start the backend server"}
                  </div>
                </div>

                <div className="metric-card">
                  <div className="metric-top">
                    <span>AUDIT RECORDS</span>
                    <span className="metric-icon purple-icon">▤</span>
                  </div>

                  <div className="metric-value">{auditCount}</div>
                  <div className="metric-foot">
                    {workflowAuditCount} workflow · {simulatedAuditCount} simulated
                  </div>
                </div>

                <div className="metric-card">
                  <div className="metric-top">
                    <span>MODEL WORKFLOW</span>
                    <span className="metric-icon cyan-icon">⌘</span>
                  </div>

                  <div className="metric-value small-value">
                    {health?.model_workflow ? "Configured" : "Unknown"}
                  </div>

                  <div className="metric-foot">Existing Python workflow</div>
                </div>

                <div className="metric-card">
                  <div className="metric-top">
                    <span>RESPONSE ACTIONS</span>
                    <span className="metric-icon amber-icon">⛨</span>
                  </div>

                  <div className="metric-value success-text">Disabled</div>

                  <div className="metric-foot">
                    Read-only demonstration mode
                  </div>
                </div>
              </div>

              <div className="content-grid">
                <ResultCard title="Defense workflow">
                  <div className="pipeline">
                    {[
                      "Network flow",
                      "Transformer detector",
                      "Threat investigation",
                      "MCP knowledge",
                      "Planner & safety",
                      "Verification & audit",
                    ].map((step, i) => (
                      <div className="pipeline-step" key={step}>
                        <span className="pipeline-number">
                          {String(i + 1).padStart(2, "0")}
                        </span>

                        <span>{step}</span>

                        {i < 5 && <span className="pipeline-arrow">→</span>}
                      </div>
                    ))}
                  </div>
                </ResultCard>

                <ResultCard title="Recent audit activity">
                  {audit.length ? (
                    <div className="recent-list">
                      {audit.slice(0, 4).map((record, i) => (
                        <div
                          className="recent-item"
                          key={`${getRecordTimestamp(record)}-${i}`}
                        >
                          <span className="activity-dot" />

                          <div>
                            <strong>
                              {getRecordAttack(record)}
                              {isSimulationRecord(record) &&
                                " · SIMULATED"}
                            </strong>

                            <small>
                              {formatTimestamp(getRecordTimestamp(record))}
                            </small>
                          </div>

                          <span className="activity-status">LOGGED</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="empty-state">
                      {auditError || "No audit records returned yet."}
                    </p>
                  )}

                  <button
                    className="text-button"
                    onClick={() => setPage("audit")}
                  >
                    View audit records →
                  </button>
                </ResultCard>
              </div>

              <div className="notice">
                <span>ⓘ</span>
                <p>
                  <strong>Demo data integrity:</strong> Dashboard metrics show
                  API status and records actually returned by the backend.
                  Detection performance statistics will be added from measured
                  evaluation results, not invented values.
                </p>
              </div>
            </>
          )}

          {/* THREAT INVESTIGATION */}
          {page === "investigation" && (
            <>
              <div className="page-heading">
                <div>
                  <div className="eyebrow">DETECTION CONSOLE</div>
                  <h1>Threat Investigation</h1>
                  <p>
                    Submit one network flow to the existing 58-feature
                    Transformer workflow.
                  </p>
                </div>

                <span className="read-only-badge">READ-ONLY MODE</span>
              </div>

              <div className="notice">
                <span>ⓘ</span>
                <p>
                  Input must be a JSON object containing all required numeric
                  model features in the checkpoint&apos;s expected schema. Do
                  not include the target label <code>Attack_Type</code>. This
                  form does not retrain or modify the model.
                </p>
              </div>

              <ResultCard title="Network flow input">
                <form onSubmit={analyze}>
                  <label className="field-label" htmlFor="sample-select">
                    VALIDATION DATASET SAMPLE
                  </label>

                  <select
                    id="sample-select"
                    className="sample-select"
                    value={selectedSampleId}
                    onChange={(event) => {
                      const sample = samples.find(
                        (item) => String(item.id) === event.target.value
                      );

                      setSelectedSampleId(event.target.value);

                      if (sample) {
                        setFeaturesText(
                          JSON.stringify(sample.features, null, 2)
                        );
                        setResult(null);
                        setError("");
                        setSimulationError("");
                      }
                    }}
                    disabled={samplesLoading || samples.length === 0}
                  >
                    <option value="">
                      {samplesLoading
                        ? "Loading samples..."
                        : "Choose a validation sample"}
                    </option>

                    {samples.map((sample) => (
                      <option key={sample.id} value={String(sample.id)}>
                        Sample {sample.id} — {sample.actual_label}
                      </option>
                    ))}
                  </select>

                  <p className="muted sample-help">
                    Choose a real validation row to populate its 58 model
                    features.
                  </p>

                  {selectedSample && (
                    <div className="sample-label">
                      <span>ACTUAL DATASET LABEL</span>
                      <strong>{pretty(selectedSample.actual_label)}</strong>
                      <small>
                        This is the validation label, not the model prediction.
                      </small>
                    </div>
                  )}

                  <label className="field-label" htmlFor="flow-json">
                    FEATURE JSON
                  </label>

                  <textarea
                    id="flow-json"
                    className="json-input"
                    value={featuresText}
                    onChange={(event) => setFeaturesText(event.target.value)}
                    placeholder={`Paste a single flow as JSON, for example:
{
  "feature_name": 0.0
}`}
                    spellCheck="false"
                    required
                  />

                  <div className="form-bottom">
                    <span className="muted">
                      Numeric feature values only · JSON format
                    </span>

                    <button className="primary-button" disabled={busy}>
                      {busy ? "Analyzing…" : "Run threat analysis →"}
                    </button>
                  </div>
                </form>

                {error && <div className="error-banner">{error}</div>}
              </ResultCard>

              <ResultCard title="Analysis output">
                {result ? (
                  <div className="result-summary">
                    <div className="result-chip">
                      <small>PREDICTED CLASS</small>
                      <strong>{pretty(detection.predicted_attack)}</strong>
                    </div>

                    <div className="result-chip">
                      <small>MODEL CONFIDENCE</small>
                      <strong>
                        {typeof detection.confidence === "number"
                          ? `${(detection.confidence * 100).toFixed(2)}%`
                          : "—"}
                      </strong>
                    </div>

                    <button
                      className="secondary-button"
                      onClick={() => setPage("workflow")}
                    >
                      View workflow details →
                    </button>
                  </div>
                ) : (
                  <p className="empty-state">
                    Submit a valid feature object to see the actual prediction
                    and downstream workflow results.
                  </p>
                )}
              </ResultCard>
            </>
          )}

          {/* AGENT WORKFLOW */}
          {page === "workflow" && (
            <>
              <div className="page-heading">
                <div>
                  <div className="eyebrow">AGENT ORCHESTRATION</div>
                  <h1>Agent Workflow</h1>
                  <p>
                    Trace the decisions returned by your existing Python
                    workflow.
                  </p>
                </div>

                <span className="read-only-badge">ACTIONS DISABLED</span>
              </div>

              <div className="workflow-timeline">
                {[
                  ["01", "Transformer Detection", "detection", detection],
                  [
                    "02",
                    "Threat Investigation",
                    "investigation",
                    investigation,
                  ],
                  ["03", "MCP Knowledge Lookup", "mcp_knowledge", knowledge],
                  ["04", "Decision Planner", "plan", plan],
                  ["05", "Safety Policy", "safety", safety],
                  ["06", "Verification", "verification", verification],
                ].map(([number, title, key, data], i) => (
                  <div className="timeline-row" key={key}>
                    <div className="timeline-marker">{number}</div>

                    <div className="timeline-body">
                      <div className="timeline-heading">
                        <h3>{title}</h3>

                        <span
                          className={`stage-badge ${
                            result ? "stage-done" : "stage-idle"
                          }`}
                        >
                          {result ? "RESULT AVAILABLE" : "AWAITING ANALYSIS"}
                        </span>
                      </div>

                      <pre>
                        {result
                          ? pretty(data)
                          : "Run an investigation to populate this stage with real workflow output."}
                      </pre>
                    </div>

                    {i < 5 && <div className="timeline-connector" />}
                  </div>
                ))}
              </div>

              {result && (
                <div className="notice">
                  <span>⛨</span>
                  <p>
                    <strong>Action safety:</strong> Workflow reports{" "}
                    <code>
                      action_taken: {String(result.action_taken ?? false)}
                    </code>
                    . The dashboard does not perform blocking, isolation, or
                    other network modifications.
                  </p>
                </div>
              )}
            </>
          )}

          {/* ACTIVE DEFENSE */}
          {page === "active-defense" && (
            <>
              <div className="page-heading">
                <div>
                  <div className="eyebrow">SIMULATED INCIDENT RESPONSE</div>
                  <h1>Active Defense &amp; Response</h1>
                  <p>
                    Review proposed security responses and safely simulate
                    their outcomes using your existing threat analysis
                    workflow.
                  </p>
                </div>

                <span className="read-only-badge">SIMULATION MODE</span>
              </div>

              <div className="notice">
                <span>ⓘ</span>
                <p>
                  <strong>Safe simulation only:</strong> This prototype does
                  not block traffic, isolate devices, change firewall rules,
                  or modify the network. A simulated result is not proof that
                  a real attack has been stopped.
                </p>
              </div>

              <ResultCard title="Incident assessment">
                {result ? (
                  <div className="result-summary">
                    <div className="result-chip">
                      <small>DETECTED ATTACK</small>
                      <strong>{pretty(detection.predicted_attack)}</strong>
                    </div>

                    <div className="result-chip">
                      <small>MODEL CONFIDENCE</small>
                      <strong>
                        {typeof detection.confidence === "number"
                          ? `${(detection.confidence * 100).toFixed(2)}%`
                          : "—"}
                      </strong>
                    </div>

                    <div className="result-chip">
                      <small>PLANNER DECISION</small>
                      <strong>{pretty(plan.decision)}</strong>
                    </div>

                    <button
                      className="secondary-button"
                      onClick={() => setPage("investigation")}
                    >
                      Analyze another flow →
                    </button>
                  </div>
                ) : (
                  <div className="empty-state">
                    <p>No analysis result is available yet.</p>
                    <button
                      className="primary-button"
                      onClick={() => setPage("investigation")}
                    >
                      Go to Threat Investigation →
                    </button>
                  </div>
                )}
              </ResultCard>

              <ResultCard title="Proposed response">
                {result ? (
                  (() => {
                    const attack = String(
                      detection.predicted_attack || "UNKNOWN"
                    ).toUpperCase();

                    const responses = {
                      DDOS: {
                        title: "DDoS traffic investigation",
                        description:
                          "Review traffic volume, source distribution, and service availability. Simulate a rate-limiting recommendation.",
                      },
                      DOS: {
                        title: "Denial-of-service investigation",
                        description:
                          "Review resource consumption and suspicious traffic patterns. Simulate a service-protection recommendation.",
                      },
                      PORT_SCAN: {
                        title: "Port-scan investigation",
                        description:
                          "Review connection attempts and targeted ports. Simulate a recommendation to review firewall rules.",
                      },
                      BRUTE_FORCE: {
                        title: "Brute-force investigation",
                        description:
                          "Review repeated authentication failures. Simulate an account-protection recommendation.",
                      },
                      WEB_ATTACK: {
                        title: "Web-attack investigation",
                        description:
                          "Review suspicious HTTP requests and application logs. Simulate an application-protection recommendation.",
                      },
                      BOT: {
                        title: "Bot-traffic investigation",
                        description:
                          "Review suspicious communication patterns and related events. Simulate a further-investigation recommendation.",
                      },
                      BENIGN: {
                        title: "Continue monitoring",
                        description:
                          "The detector classified this flow as benign. Simulate recording the assessment and continuing monitoring, without containment.",
                      },
                    };

                    const recommendation = responses[attack] || {
                      title: "Manual security review",
                      description:
                        "The predicted class has no configured response template. Simulate recording the event for manual investigation.",
                    };

                    return (
                      <div className="response-plan">
                        <div className="response-plan-header">
                          <span className="metric-icon amber-icon">⛨</span>

                          <div>
                            <h3>{recommendation.title}</h3>
                            <p>{recommendation.description}</p>
                          </div>
                        </div>

                        <div className="notice">
                          <span>✓</span>
                          <p>
                            <strong>Safety check:</strong>{" "}
                            {safety.action_permitted === false
                              ? "The workflow does not permit real response execution."
                              : "Real response execution remains disabled in this prototype."}
                          </p>
                        </div>

                        <button
                          className="primary-button"
                          disabled={simulationBusy}
                          onClick={() =>
                            simulateResponse(attack, recommendation)
                          }
                        >
                          {simulationBusy
                            ? "Saving simulation…"
                            : "▶ Simulate Response"}
                        </button>

                        {simulationError && (
                          <div className="error-banner">
                            {simulationError}
                          </div>
                        )}

                        {result.simulated_response && (
                          <div className="simulation-result">
                            <div className="intel-status">
                              <span className="status-dot online-dot" />
                              <strong>Simulation record saved</strong>
                            </div>

                            <div className="result-summary">
                              <div className="result-chip">
                                <small>SIMULATION STATUS</small>
                                <strong>
                                  {pretty(
                                    result.simulated_response.simulation_status ||
                                      result.simulated_response.status ||
                                      "SIMULATED"
                                  )}
                                </strong>
                              </div>

                              <div className="result-chip">
                                <small>REAL ACTION EXECUTED</small>
                                <strong>
                                  {result.simulated_response.real_action_executed
                                    ? "YES"
                                    : "NO"}
                                </strong>
                              </div>
                            </div>

                            <p>{result.simulated_response.outcome}</p>

                            <pre>{pretty(result.simulated_response)}</pre>
                          </div>
                        )}
                      </div>
                    );
                  })()
                ) : (
                  <p className="empty-state">
                    Run an analysis first to generate a response recommendation.
                  </p>
                )}
              </ResultCard>
            </>
          )}

          {/* THREAT INTELLIGENCE */}
          {page === "intelligence" && (
            <>
              <div className="page-heading">
                <div>
                  <div className="eyebrow">KNOWLEDGE ENRICHMENT</div>
                  <h1>Threat Intelligence</h1>
                  <p>
                    MCP-retrieved threat context and ATT&amp;CK mappings from
                    your existing integration.
                  </p>
                </div>
              </div>

              <ResultCard title="MCP threat knowledge">
                {result ? (
                  <>
                    <div className="intel-status">
                      <span
                        className={`status-dot ${
                          knowledge.status === "found" ? "online-dot" : ""
                        }`}
                      />

                      <strong>
                        {pretty(knowledge.status || "Status not provided")}
                      </strong>
                    </div>

                    <pre>{pretty(knowledge)}</pre>
                  </>
                ) : (
                  <p className="empty-state">
                    Run a threat investigation to display the actual MCP lookup
                    response.
                  </p>
                )}
              </ResultCard>

              <ResultCard title="Threat investigation findings">
                {result ? (
                  <pre>{pretty(investigation)}</pre>
                ) : (
                  <p className="empty-state">
                    Investigation findings will appear here after analysis.
                  </p>
                )}
              </ResultCard>

              <div className="notice">
                <span>ⓘ</span>
                <p>
                  ATT&amp;CK identifiers and threat descriptions are displayed
                  only when returned by your knowledge retriever. An
                  unavailable mapping is not treated as confirmed intelligence.
                </p>
              </div>
            </>
          )}

          {/* AUDIT RECORDS */}
          {page === "audit" && (
            <>
              <div className="page-heading">
                <div>
                  <div className="eyebrow">TRACEABILITY &amp; REVIEW</div>
                  <h1>Audit Records</h1>
                  <p>
                    Persisted workflow and simulated-response records returned
                    by the audit API.
                  </p>
                </div>

                <button
                  className="secondary-button"
                  onClick={loadAudit}
                >
                  ↻ Refresh records
                </button>
              </div>

              <div className="metric-grid audit-metrics">
                <div className="metric-card">
                  <div className="metric-top">
                    <span>RECORDS LOADED</span>
                    <span className="metric-icon cyan-icon">▤</span>
                  </div>

                  <div className="metric-value">{auditCount}</div>

                  <div className="metric-foot">
                    {workflowAuditCount} workflow · {simulatedAuditCount} simulated
                    responses
                  </div>
                </div>

                <div className="metric-card">
                  <div className="metric-top">
                    <span>SIMULATED RESPONSES</span>
                    <span className="metric-icon purple-icon">✓</span>
                  </div>

                  <div className="metric-value">{simulatedAuditCount}</div>

                  <div className="metric-foot">
                    No real mitigation performed
                  </div>
                </div>

                <div className="metric-card">
                  <div className="metric-top">
                    <span>RESPONSE ACTIONS</span>
                    <span className="metric-icon amber-icon">⛨</span>
                  </div>

                  <div className="metric-value success-text">Disabled</div>

                  <div className="metric-foot">
                    Dashboard does not execute actions
                  </div>
                </div>
              </div>

              {auditError && (
                <div className="error-banner">{auditError}</div>
              )}

              <ResultCard title="Audit log">
                {audit.length ? (
                  <div className="audit-table-wrap">
                    <table className="audit-table">
                      <thead>
                        <tr>
                          <th>RECORD / TIME</th>
                          <th>PREDICTED ATTACK</th>
                          <th>CONFIDENCE</th>
                          <th>DECISION / PRIORITY</th>
                          <th>VERIFICATION</th>
                          <th>ACTION</th>
                          <th>DETAILS</th>
                        </tr>
                      </thead>

                      <tbody>
                        {audit.map((record, i) => {
                          const isSimulation = isSimulationRecord(record);
                          const recordVerification =
                            record.verification || {};

                          const timestamp = getRecordTimestamp(record);
                          const predictedAttack = getRecordAttack(record);
                          const confidence = getRecordConfidence(record);
                          const decision = getRecordDecision(record);
                          const priority = getRecordPriority(record);
                          const verified = getRecordVerification(record);
                          const actionTaken = getRecordAction(record);
                          const permissionText =
                            getRecordPermissionText(record);

                          return (
                            <tr key={`${timestamp || "record"}-${i}`}>
                              <td className="audit-record-cell">
                                <span className="record-id">
                                  #{audit.length - i}
                                </span>

                                <small className="audit-timestamp">
                                  {formatTimestamp(timestamp)}
                                </small>
                              </td>

                              <td className="audit-attack-cell">
                                <div>{predictedAttack}</div>

                                {isSimulation && (
                                  <small className="audit-secondary-text">
                                    Simulated response
                                  </small>
                                )}
                              </td>

                              <td>{confidence}</td>

                              <td className="audit-decision-cell">
                                <span
                                  className={`table-tag audit-decision-tag ${getDecisionClass(
                                    decision
                                  )}`}
                                >
                                  {decision}
                                </span>

                                {!isSimulation && (
                                  <small className="audit-secondary-text">
                                    <span className="audit-secondary-label">
                                      Priority
                                    </span>

                                    <span className="audit-priority-value">
                                      {priority}
                                    </span>
                                  </small>
                                )}
                              </td>

                              <td className="audit-verification-cell">
                                <span
                                  className={`table-tag ${
                                    verified === true
                                      ? "audit-verified-tag"
                                      : verified === false
                                        ? "audit-failed-tag"
                                        : ""
                                  }`}
                                >
                                  {verified === true
                                    ? "Verified"
                                    : verified === false
                                      ? "Failed"
                                      : "Not recorded"}
                                </span>

                                {recordVerification.errors?.length > 0 && (
                                  <small className="audit-secondary-text">
                                    {recordVerification.errors.length} error(s)
                                  </small>
                                )}
                              </td>

                              <td className="audit-action-cell">
                                <span className="table-tag safe-tag">
                                  {isSimulation
                                    ? "Simulated"
                                    : actionTaken
                                      ? "Action taken"
                                      : "No action"}
                                </span>

                                <small className="audit-secondary-text">
                                  {permissionText}
                                </small>
                              </td>

                              <td className="audit-details-cell">
                                <details>
                                  <summary>Inspect JSON</summary>
                                  <pre>{pretty(record)}</pre>
                                </details>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="empty-state">
                    {auditError ||
                      "No audit records found in the API response."}
                  </p>
                )}
              </ResultCard>

              <div className="notice">
                <span>ⓘ</span>
                <p>
                  These are application audit records, not cryptographically
                  tamper-proof logs. Protect the file and its permissions
                  separately for production use.
                </p>
              </div>
            </>
          )}

          <footer className="footer">
            <span>CYBER THREAT DEFENSE</span>

            <span>
              Local research dashboard <i /> Safe mode active
            </span>
          </footer>
        </div>
      </main>
    </div>
  );
}
