
import { useEffect, useMemo, useState } from "react";
import "./AttackReplay.css";

const STAGES = [
  {
    title: "Traffic activity",
    description:
      "Illustrative traffic moves through the monitored network.",
  },
  {
    title: "Threat detection",
    description:
      "The detector result is displayed for the selected sample.",
  },
  {
    title: "Investigation",
    description:
      "The workflow result and detection confidence are reviewed.",
  },
  {
    title: "Response simulation",
    description:
      "A response rule is illustrated without applying it to a real system.",
  },
  {
    title: "Verification",
    description:
      "The simulation result and safety checks are reviewed; real response execution remains disabled.",
  },
];

const SCENARIOS = {
  BENIGN: {
    title: "Normal traffic",
    subtitle: "Steady client-to-server requests",
    kind: "benign",
    sources: ["Client A", "Client B", "Client C"],
    target: "Application server",
    trafficLabel: "Normal requests",
    responseLabel: "Monitor only",
    color: "#35d49a",
    eventRate: 4,
    packetLimit: 4,
  },
  DDOS: {
    title: "Distributed denial of service",
    subtitle: "Many sources converge rapidly on one target",
    kind: "ddos",
    sources: ["Source A", "Source B", "Source C", "Source D", "Source E"],
    target: "Target server",
    trafficLabel: "High-volume requests",
    responseLabel: "Simulated rate limiting",
    color: "#ff647c",
    eventRate: 30,
    packetLimit: 24,
  },
  DOS: {
    title: "Denial of service",
    subtitle: "Repeated requests from a limited source set",
    kind: "dos",
    sources: ["Source A", "Source B", "Source C"],
    target: "Target server",
    trafficLabel: "Repeated requests",
    responseLabel: "Simulated traffic filtering",
    color: "#ff8b61",
    eventRate: 20,
    packetLimit: 16,
  },
  BRUTE_FORCE: {
    title: "Brute-force activity",
    subtitle: "Repeated login attempts toward authentication",
    kind: "brute-force",
    sources: ["Client A", "Client B", "Client C"],
    target: "Login service",
    trafficLabel: "Login attempts",
    responseLabel: "Simulated login guard",
    color: "#ffb454",
    eventRate: 8,
    packetLimit: 8,
  },
  PORT_SCAN: {
    title: "Port scanning",
    subtitle: "Probes attempt to discover exposed services",
    kind: "port-scan",
    sources: ["Scanner A", "Scanner B", "Scanner C"],
    target: "Service ports",
    trafficLabel: "Service probes",
    responseLabel: "Simulated source restriction",
    color: "#c19aff",
    eventRate: 12,
    packetLimit: 12,
  },
  BOT: {
    title: "Automated bot traffic",
    subtitle: "Repeated automated requests target the application",
    kind: "bot",
    sources: ["Bot A", "Bot B", "Bot C"],
    target: "Application server",
    trafficLabel: "Automated requests",
    responseLabel: "Simulated traffic filtering",
    color: "#62b8ff",
    eventRate: 15,
    packetLimit: 12,
  },
  WEB_ATTACK: {
    title: "Web application attack",
    subtitle: "Suspicious requests target a web endpoint",
    kind: "web-attack",
    sources: ["Client A", "Client B", "Client C"],
    target: "Web application",
    trafficLabel: "Suspicious web requests",
    responseLabel: "Manual review",
    color: "#ff779e",
    eventRate: 10,
    packetLimit: 10,
  },
};

function normalizeLabel(value) {
  return String(value ?? "")
    .trim()
    .toUpperCase()
    .replace(/[\s-]+/g, "_");
}

function getScenario(result, selectedLabel) {
  const actualLabel =
    result?.simulation?.actual_dataset_label ??
    result?.actual_label ??
    result?.actual ??
    selectedLabel;

  const normalized = normalizeLabel(actualLabel);

  return {
    label: normalized || "UNKNOWN",
    actualLabel: actualLabel || "Unknown",
    scenario: SCENARIOS[normalized] || {
      title: "Unclassified traffic",
      subtitle: "Generic traffic pattern for an unrecognized label",
      kind: "unknown",
      sources: ["Source A", "Source B", "Source C"],
      target: "Application server",
      trafficLabel: "Unclassified requests",
      responseLabel: "Review recommended",
      color: "#a8b4c8",
      eventRate: 8,
      packetLimit: 8,
    },
  };
}

function formatConfidence(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "—";
  }

  return `${(value * 100).toFixed(2)}%`;
}

function getConfidence(result) {
  const candidates = [
    result?.workflow?.analysis?.detection?.confidence,
    result?.analysis?.detection?.confidence,
    result?.detection?.confidence,
    result?.confidence,
  ];

  return candidates.find(
    (value) => typeof value === "number" && Number.isFinite(value)
  );
}

function getPredictedLabel(result) {
  return (
    result?.simulation?.predicted_label ??
    result?.workflow?.analysis?.detection?.predicted_attack ??
    result?.analysis?.detection?.predicted_attack ??
    result?.predicted_label ??
    result?.predicted ??
    "Not available"
  );
}

function getDecision(result) {
  return (
    result?.workflow?.plan?.decision ??
    result?.workflow?.decision ??
    result?.plan?.decision ??
    result?.simulation?.workflow_decision ??
    "Not recorded"
  );
}

function getResponseAction(result, scenario) {
  return result?.simulation?.action ?? scenario.responseLabel;
}

/*
 * Describe the simulator's outcome separately from the workflow decision.
 * An INVESTIGATE decision does not prevent the UI from displaying a
 * completed simulated response. Neither result authorizes real actions.
 */
function getResponseDescription(action, decision, result) {
  const simulation = result?.simulation;
  const outcome =
    simulation?.response?.outcome ?? simulation?.outcome;

  const simulationOnly = simulation?.simulation_only === true;
  const realActionExecuted =
    simulation?.real_action_executed === true;

  if (realActionExecuted) {
    return {
      title: "Unexpected real-action status",
      description:
        "The result reports a real action. Review the backend safety controls and audit record.",
      mode: "review",
    };
  }

  if (simulationOnly && outcome === "SIMULATED_ONLY") {
    switch (action) {
      case "SIMULATE_RATE_LIMIT":
        return {
          title: "Simulated rate limiting",
          description:
            "The visualization represents rate limiting. No real traffic was limited.",
          mode: "rate-limit",
        };

      case "SIMULATE_TRAFFIC_FILTER":
        return {
          title: "Simulated traffic filtering",
          description:
            "The visualization represents traffic filtering. No real traffic was filtered.",
          mode: "filter",
        };

      case "SIMULATE_LOGIN_BLOCK":
        return {
          title: "Simulated login guard",
          description:
            "The visualization represents a login guard. No login attempts were actually blocked.",
          mode: "login-block",
        };

      case "SIMULATE_SOURCE_BLOCK":
        return {
          title: "Simulated source restriction",
          description:
            "The visualization represents source restriction. No real source was blocked.",
          mode: "source-block",
        };

      case "SIMULATE_MANUAL_REVIEW":
        return {
          title: "Manual review simulation",
          description:
            "The event is represented as flagged for analyst review, not automatically blocked.",
          mode: "review",
        };

      case "MONITOR":
        return {
          title: "Monitoring retained",
          description:
            "The visualization represents continued monitoring without mitigation.",
          mode: "monitor",
        };

      default:
        return {
          title: "Response simulation completed",
          description:
            "The simulator returned a result, but no specific response behavior was matched.",
          mode: "review",
        };
    }
  }

  if (decision === "INVESTIGATE") {
    return {
      title: "Investigation required",
      description:
        "The workflow recommends investigation. No real containment action is authorized.",
      mode: "review",
    };
  }

  return {
    title: "Response not confirmed",
    description:
      "The simulator did not confirm a completed simulated response.",
    mode: "review",
  };
}

function getVerification(result) {
  return result?.simulation?.verification ?? result?.verification ?? {};
}

function getVerificationStatus(result) {
  const verification = getVerification(result);

  if (verification.verified === true) return "Passed";
  if (verification.verified === false) return "Failed";

  return "Not available";
}

/* -------------------------------------------------------
   TRAFFIC ANIMATION
   All packet movement is illustrative.
------------------------------------------------------- */

function TrafficPaths({
  scenario,
  stage,
  playing,
  packetCount,
  eventCount,
  responseAction,
}) {
  const sourcePositions = [
    { x: 58, y: 78 },
    { x: 58, y: 143 },
    { x: 58, y: 208 },
    { x: 58, y: 273 },
    { x: 58, y: 338 },
  ];

  const isPortScan = scenario.kind === "port-scan";
  const isDdos = scenario.kind === "ddos";
  const isDos = scenario.kind === "dos";
  const isBruteForce = scenario.kind === "brute-force";
  const isWebAttack = scenario.kind === "web-attack";
  const isBenign = scenario.kind === "benign";
  const isBot = scenario.kind === "bot";

  const targetX = 565;
  const targetY = 208;

  const paths = scenario.sources.map((_, index) => {
    const source = sourcePositions[index % sourcePositions.length];

    if (isPortScan) {
      const serviceY = 100 + (index % 5) * 52;

      return {
        path: `M ${source.x + 42} ${source.y}
          L 260 ${source.y}
          L 390 ${serviceY}
          L ${targetX} ${serviceY}`,
      };
    }

    if (isBruteForce) {
      const authY = 208 + ((index % 3) - 1) * 25;

      return {
        path: `M ${source.x + 42} ${source.y}
          C 210 ${source.y}, 330 ${authY}, ${targetX} ${authY}`,
      };
    }

    if (isWebAttack) {
      const webY = 208 + ((index % 3) - 1) * 18;

      return {
        path: `M ${source.x + 42} ${source.y}
          C 220 ${source.y}, 340 ${webY}, ${targetX} ${webY}`,
      };
    }

    if (isDdos) {
      const offset = ((index % 5) - 2) * 7;

      return {
        path: `M ${source.x + 42} ${source.y}
          C 220 ${source.y}, 330 ${targetY + offset},
          ${targetX} ${targetY + offset}`,
      };
    }

    if (isDos) {
      const offset = ((index % 3) - 1) * 12;

      return {
        path: `M ${source.x + 42} ${source.y}
          L 210 ${source.y}
          L 330 ${targetY + offset}
          L ${targetX} ${targetY + offset}`,
      };
    }

    if (isBot) {
      const offset = ((index % 3) - 1) * 16;

      return {
        path: `M ${source.x + 42} ${source.y}
          Q 190 ${source.y - 22} 260 ${targetY + offset}
          T ${targetX} ${targetY + offset}`,
      };
    }

    return {
      path: `M ${source.x + 42} ${source.y}
        C 230 ${source.y}, 350 ${targetY}, ${targetX} ${targetY}`,
    };
  });

  const visiblePacketCount = Math.max(
    2,
    Math.min(Number(packetCount) || 2, 24)
  );

  const baseDuration =
    3.4 / Math.sqrt(Math.max(1, Number(eventCount || 100) / 100));

  const speedMultiplier = isBenign
    ? 1.7
    : isDdos
      ? 0.45
      : isDos
        ? 0.62
        : isBruteForce
          ? 1.25
          : isPortScan
            ? 1.05
            : isBot
              ? 0.75
              : isWebAttack
                ? 0.95
                : 1;

  const responseStage = stage >= 3;

  const responseSpeedMultiplier =
    responseStage && responseAction === "SIMULATE_RATE_LIMIT"
      ? 3.5
      : 1;

  const packetDuration = Math.max(
    0.45,
    Math.min(
      10,
      baseDuration * speedMultiplier * responseSpeedMultiplier
    )
  );

  return (
    <g className={`traffic-paths ${playing ? "traffic-active" : ""}`}>
      {paths.map((item, index) => (
        <path
          key={`${scenario.kind}-path-${index}`}
          d={item.path}
          className={`traffic-path ${
            isBenign ? "path-normal" : "path-suspicious"
          }`}
          style={{
            opacity:
              stage === 3 && !isBenign
                ? 0.45
                : isDdos
                  ? 0.9
                  : 0.65,
          }}
        />
      ))}

      {playing &&
        Array.from({ length: visiblePacketCount }, (_, packetIndex) => {
          const pathIndex = packetIndex % paths.length;
          const item = paths[pathIndex];
          const lane = Math.floor(packetIndex / paths.length);

          const duration = packetDuration * (1 + (lane % 3) * 0.08);
          const radius = isDdos ? 4.2 : isPortScan ? 3 : 3.5;

          return (
            <circle
              key={`${scenario.kind}-packet-${packetIndex}`}
              r={radius}
              fill={scenario.color}
              className="packet-dot"
            >
              <animateMotion
                dur={`${duration}s`}
                repeatCount="indefinite"
                begin={`${-(packetIndex * 0.19)}s`}
                path={item.path}
              />
            </circle>
          );
        })}
    </g>
  );
}

/* -------------------------------------------------------
   NETWORK SCENE
------------------------------------------------------- */

function NetworkScene({
  scenario,
  stage,
  playing,
  packetCount,
  eventCount,
  responseAction,
  responseInfo,
}) {
  const isBenign = scenario.kind === "benign";
  const isDdos = scenario.kind === "ddos";
  const isPortScan = scenario.kind === "port-scan";
  const isBruteForce = scenario.kind === "brute-force";
  const isWebAttack = scenario.kind === "web-attack";

  return (
    <svg
      className={`network-scene stage-${stage} scenario-${scenario.kind}`}
      viewBox="0 0 700 410"
      role="img"
      aria-label={`${scenario.title} network traffic animation`}
    >
      <defs>
        <linearGradient id="networkBackground" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#102137" />
          <stop offset="100%" stopColor="#091321" />
        </linearGradient>

        <pattern
          id="networkGrid"
          width="28"
          height="28"
          patternUnits="userSpaceOnUse"
        >
          <path
            d="M 28 0 L 0 0 0 28"
            fill="none"
            stroke="#26394d"
            strokeWidth="0.6"
          />
        </pattern>

        <filter id="nodeGlow">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <rect width="700" height="410" rx="16" fill="url(#networkBackground)" />
      <rect width="700" height="410" rx="16" fill="url(#networkGrid)" />

      <text x="25" y="29" className="svg-heading">
        NETWORK TRAFFIC VIEW
      </text>

      <text x="675" y="29" textAnchor="end" className="svg-subheading">
        ILLUSTRATIVE • NOT LIVE PACKETS
      </text>

      <TrafficPaths
        scenario={scenario}
        stage={stage}
        playing={playing}
        packetCount={packetCount}
        eventCount={eventCount}
        responseAction={responseAction}
      />

      {scenario.sources.map((source, index) => {
        const y = 78 + index * 65;

        return (
          <g key={source} className="source-node">
            <rect
              x="18"
              y={y - 20}
              width="110"
              height="40"
              rx="9"
              fill={isBenign ? "#123b35" : "#30212c"}
              stroke={isBenign ? "#35d49a" : scenario.color}
              strokeWidth="1.2"
            />

            <circle cx="32" cy={y} r="4" fill={scenario.color} />

            <text x="43" y={y + 4} className="svg-node-label">
              {source}
            </text>
          </g>
        );
      })}

      {isPortScan ? (
        <g className={`service-stack ${stage >= 1 ? "services-alert" : ""}`}>
          <text x="435" y="65" className="svg-subheading">
            SERVICE PORTS
          </text>

          {[80, 443, 22, 3389, 8080].map((port, index) => {
            const y = 91 + index * 47;

            return (
              <g key={port}>
                <rect
                  x="422"
                  y={y}
                  width="120"
                  height="31"
                  rx="6"
                  className={`service-port ${
                    stage >= 1 ? "service-port-alert" : ""
                  }`}
                />

                <text
                  x="482"
                  y={y + 20}
                  textAnchor="middle"
                  className="svg-node-label"
                >
                  Port {port}
                </text>
              </g>
            );
          })}
        </g>
      ) : (
        <g
          className={`target-node ${
            stage >= 1 && !isBenign ? "target-alert" : ""
          }`}
        >
          <rect
            x="505"
            y="162"
            width="150"
            height="94"
            rx="12"
            fill={isBenign ? "#123b35" : "#33202d"}
            stroke={isBenign ? "#35d49a" : scenario.color}
            strokeWidth="1.5"
          />

          <rect
            x="522"
            y="179"
            width="30"
            height="27"
            rx="4"
            fill="#142235"
            stroke={scenario.color}
          />

          <path
            d="M 527 188 L 547 188 M 527 195 L 547 195"
            stroke={scenario.color}
            strokeWidth="1.4"
          />

          <text x="561" y="193" className="svg-node-label">
            {isBruteForce ? "AUTH" : isWebAttack ? "WEB APP" : "SERVER"}
          </text>

          <text x="580" y="229" textAnchor="middle" className="svg-subheading">
            {scenario.target}
          </text>
        </g>
      )}

      <g className={`inspection-node ${stage >= 1 ? "inspection-active" : ""}`}>
        <rect
          x="260"
          y="164"
          width="104"
          height="86"
          rx="12"
          fill="#122c40"
          stroke={stage >= 1 && !isBenign ? scenario.color : "#46708b"}
          strokeWidth="1.5"
        />

        <circle
          cx="312"
          cy="191"
          r="13"
          fill="none"
          stroke={stage >= 1 && !isBenign ? scenario.color : "#76b6d7"}
          strokeWidth="2"
        />

        <path
          d="M 305 191 L 310 196 L 320 185"
          fill="none"
          stroke={stage >= 1 && !isBenign ? scenario.color : "#76b6d7"}
          strokeWidth="2"
        />

        <text x="312" y="221" textAnchor="middle" className="svg-node-label">
          Detector
        </text>

        <text x="312" y="237" textAnchor="middle" className="svg-small-label">
          {stage >= 1 ? "Evaluated" : "Monitoring"}
        </text>
      </g>

      {stage >= 1 && !isBenign && (
        <g className="alert-marker" filter="url(#nodeGlow)">
          <circle
            cx="390"
            cy="120"
            r="12"
            fill="#5b2635"
            stroke={scenario.color}
          />

          <text
            x="390"
            y="125"
            textAnchor="middle"
            fill="#ffffff"
            fontSize="15"
            fontWeight="800"
          >
            !
          </text>
        </g>
      )}

      {stage >= 2 && (
        <g className="investigation-indicator">
          <rect
            x="244"
            y="281"
            width="190"
            height="44"
            rx="8"
            fill="#162e48"
            stroke="#62b8ff"
          />

          <text x="339" y="299" textAnchor="middle" className="svg-node-label">
            Investigation agent
          </text>

          <text x="339" y="314" textAnchor="middle" className="svg-small-label">
            Evidence review
          </text>
        </g>
      )}

      {stage >= 3 && !isBenign && (
        <g className="response-indicator">
          <rect
            x="445"
            y="292"
            width="210"
            height="44"
            rx="8"
            fill="#3a2d1c"
            stroke="#ffb454"
          />

          <text
            x="550"
            y="310"
            textAnchor="middle"
            className="svg-node-label"
          >
            {responseInfo.title}
          </text>

          <text
            x="550"
            y="325"
            textAnchor="middle"
            className="svg-small-label"
          >
            {responseAction}
          </text>
        </g>
      )}

      {stage >= 3 && isBenign && (
        <g className="response-indicator">
          <rect
            x="445"
            y="292"
            width="210"
            height="44"
            rx="8"
            fill="#123b35"
            stroke="#35d49a"
          />

          <text x="550" y="310" textAnchor="middle" className="svg-node-label">
            Monitoring retained
          </text>

          <text x="550" y="325" textAnchor="middle" className="svg-small-label">
            No mitigation needed
          </text>
        </g>
      )}

      {stage >= 4 && (
        <g className="verification-indicator">
          <circle
            cx="669"
            cy="370"
            r="12"
            fill="#123b35"
            stroke="#35d49a"
          />

          <path
            d="M 663 370 L 667 374 L 675 365"
            fill="none"
            stroke="#35d49a"
            strokeWidth="2"
          />

          <text x="647" y="393" textAnchor="end" className="svg-small-label">
            Verification stage
          </text>
        </g>
      )}

      {isDdos && stage === 0 && (
        <text x="350" y="386" textAnchor="middle" className="svg-alert-caption">
          Multiple sources • Illustrative high request volume
        </text>
      )}

      {isPortScan && stage === 0 && (
        <text x="350" y="386" textAnchor="middle" className="svg-alert-caption">
          Sequential service probes • Illustrative only
        </text>
      )}

      {isBruteForce && stage === 0 && (
        <text x="350" y="386" textAnchor="middle" className="svg-alert-caption">
          Repeated authentication attempts • Illustrative only
        </text>
      )}

      {isWebAttack && stage === 0 && (
        <text x="350" y="386" textAnchor="middle" className="svg-alert-caption">
          Suspicious web requests • No exploit payload is sent
        </text>
      )}
    </svg>
  );
}

/* -------------------------------------------------------
   MAIN ATTACK REPLAY COMPONENT
------------------------------------------------------- */

export default function AttackReplay({
  result,
  selectedLabel,
  eventCount = 100,
}) {
  const [stage, setStage] = useState(0);
  const [playing, setPlaying] = useState(false);

  const scenarioData = useMemo(
    () => getScenario(result, selectedLabel),
    [result, selectedLabel]
  );

  const scenario = scenarioData.scenario;
  const hasResult = Boolean(result?.simulation || result);

  const safeEventCount = Math.max(
    1,
    Math.min(1000, Number(eventCount) || 100)
  );

  const packetCount = Math.max(
    2,
    Math.min(
      scenario.packetLimit ?? 8,
      Math.ceil((scenario.packetLimit ?? 8) * safeEventCount / 100)
    )
  );

  // Illustrative rate only; it is not measured network throughput.
  const simulatedRate =
    (scenario.eventRate ?? 8) * (safeEventCount / 100);

  useEffect(() => {
    setStage(0);
    setPlaying(Boolean(hasResult));
  }, [result, hasResult]);

  useEffect(() => {
    if (!playing || !hasResult) return undefined;

    const timer = window.setInterval(() => {
      setStage((current) => {
        if (current >= STAGES.length - 1) {
          setPlaying(false);
          return STAGES.length - 1;
        }

        return current + 1;
      });
    }, 2200);

    return () => window.clearInterval(timer);
  }, [playing, hasResult]);

  const confidence = getConfidence(result);
  const predictedLabel = getPredictedLabel(result);
  const decision = getDecision(result);
  const responseAction = getResponseAction(result, scenario);

  const responseInfo = getResponseDescription(
    responseAction,
    decision,
    result
  );

  const verificationStatus = getVerificationStatus(result);

  function replay() {
    setStage(0);
    setPlaying(true);
  }

  function nextStage() {
    setStage((current) => {
      const next = Math.min(current + 1, STAGES.length - 1);

      if (next === STAGES.length - 1) {
        setPlaying(false);
      }

      return next;
    });
  }

  function togglePlayback() {
    if (playing) {
      setPlaying(false);
      return;
    }

    if (stage >= STAGES.length - 1) {
      setStage(0);
    }

    setPlaying(true);
  }

  return (
    <section className="attack-replay panel">
      <div className="replay-header">
        <div>
          <span className="eyebrow">ANIMATED NETWORK REPLAY</span>
          <h2>{scenario.title}</h2>
          <p>{scenario.subtitle}</p>
        </div>

        <span className={`replay-status ${hasResult ? "status-ready" : ""}`}>
          <span className="status-dot" />
          {hasResult ? "Simulation loaded" : "Waiting for simulation"}
        </span>
      </div>

      {!hasResult ? (
        <div className="replay-empty">
          <div className="replay-empty-icon">↗</div>
          <h3>Ready to visualize traffic</h3>
          <p>
            Select a dataset sample and scenario, then click
            <strong> Run attack simulation </strong>
            to start the animation.
          </p>
        </div>
      ) : (
        <>
          <div className="replay-stage-heading">
            <div className="stage-index">
              STAGE {stage + 1} / {STAGES.length}
            </div>

            <h3>{STAGES[stage].title}</h3>
            <p>{STAGES[stage].description}</p>
          </div>

          <div className="replay-traffic-stats">
            <div>
              <span>Simulated events</span>
              <strong>{safeEventCount.toLocaleString()}</strong>
            </div>

            <div>
              <span>Illustrative event rate</span>
              <strong>{simulatedRate.toFixed(1)} events/s</strong>
            </div>

            <div>
              <span>Animated packets</span>
              <strong>{packetCount}</strong>
            </div>
          </div>

          <div className="replay-progress" aria-label="Replay progress">
            {STAGES.map((item, index) => (
              <div
                key={item.title}
                className={`replay-progress-step ${
                  index < stage
                    ? "completed"
                    : index === stage
                      ? "current"
                      : ""
                }`}
              >
                <span>{index < stage ? "✓" : index + 1}</span>
                <small>{item.title}</small>
              </div>
            ))}
          </div>

          <div className="network-scene-wrap">
            <NetworkScene
              scenario={scenario}
              stage={stage}
              playing={playing}
              packetCount={packetCount}
              eventCount={safeEventCount}
              responseAction={responseAction}
              responseInfo={responseInfo}
            />
          </div>

          <div className={`replay-response-panel response-${responseInfo.mode}`}>
            <span className="replay-response-label">
              RESPONSE SIMULATION
            </span>

            <h4>{responseInfo.title}</h4>
            <p>{responseInfo.description}</p>

            <p>
              <strong>Simulated action:</strong> {responseAction}
            </p>

            <p>
              <strong>Workflow decision:</strong> {String(decision)}
            </p>

            <p>
              <strong>Simulation verification:</strong> {verificationStatus}
            </p>

            <p className="replay-response-note">
              Simulation only. A simulated response does not authorize or
              execute a real security action.
            </p>
          </div>

          <div className="replay-controls">
            <button className="secondary-button" onClick={replay}>
              Replay from start
            </button>

            <button className="secondary-button" onClick={togglePlayback}>
              {playing ? "Pause" : "Play"}
            </button>

            <button
              className="primary-button"
              onClick={nextStage}
              disabled={stage >= STAGES.length - 1}
            >
              Next stage →
            </button>
          </div>

          <div className="replay-facts">
            <div>
              <span>Actual dataset label</span>
              <strong>{String(scenarioData.actualLabel)}</strong>
            </div>

            <div>
              <span>Predicted label</span>
              <strong>{String(predictedLabel)}</strong>
            </div>

            <div>
              <span>Confidence</span>
              <strong>{formatConfidence(confidence)}</strong>
            </div>

            <div>
              <span>Workflow decision</span>
              <strong>{String(decision)}</strong>
            </div>

            <div>
              <span>Simulated response</span>
              <strong>{String(responseAction)}</strong>
            </div>

            <div>
              <span>Verification</span>
              <strong>{verificationStatus}</strong>
            </div>
          </div>
        </>
      )}

      <div className="replay-disclaimer">
        <strong>Simulation only:</strong> packet movement is a visual
        representation. No network traffic is generated, no system is blocked,
        and no real response action is executed.
      </div>
    </section>
  );
}
