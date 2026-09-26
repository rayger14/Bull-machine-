const DOMAIN_DEFAULTS = [
  { name: "wyckoff", weight: 0.20 },
  { name: "liquidity", weight: 0.20 },
  { name: "smc", weight: 0.25 },
  { name: "momentum", weight: 0.20 },
  { name: "temporal", weight: 0.15 },
  { name: "macro_context", weight: 0.15 }
];

const PRESETS = {
  balanced: {
    macroSuppression: false,
    macroDelta: 0,
    maxVeto: 0.30,
    minDomains: 3,
    minConfidence: 0.65,
    minStrength: 0.60,
    domainDirection: "long",
    domainStrength: 0.70,
    domainConfidence: 0.70
  },
  macro_veto: {
    macroSuppression: true,
    macroDelta: -0.05,
    maxVeto: 0.95,
    minDomains: 3,
    minConfidence: 0.65,
    minStrength: 0.60,
    domainDirection: "long",
    domainStrength: 0.75,
    domainConfidence: 0.78
  },
  low_confluence: {
    macroSuppression: false,
    macroDelta: -0.03,
    maxVeto: 0.50,
    minDomains: 4,
    minConfidence: 0.70,
    minStrength: 0.68,
    domainDirection: "neutral",
    domainStrength: 0.45,
    domainConfidence: 0.50
  },
  risk_on: {
    macroSuppression: false,
    macroDelta: 0.08,
    maxVeto: 0.25,
    minDomains: 3,
    minConfidence: 0.62,
    minStrength: 0.58,
    domainDirection: "long",
    domainStrength: 0.80,
    domainConfidence: 0.76
  }
};

const domainRows = document.getElementById("domainRows");
const traceList = document.getElementById("trace");

function renderDomainRows() {
  domainRows.innerHTML = "";
  DOMAIN_DEFAULTS.forEach((domain) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${domain.name}</td>
      <td><input type="checkbox" data-k="active" data-domain="${domain.name}" checked></td>
      <td>
        <select data-k="direction" data-domain="${domain.name}">
          <option value="long">long</option>
          <option value="short">short</option>
          <option value="neutral">neutral</option>
        </select>
      </td>
      <td><input type="number" min="0" max="1" step="0.01" value="0.70" data-k="strength" data-domain="${domain.name}"></td>
      <td><input type="number" min="0" max="1" step="0.01" value="0.70" data-k="confidence" data-domain="${domain.name}"></td>
      <td><input type="number" min="0" max="1" step="0.01" value="${domain.weight.toFixed(2)}" data-k="weight" data-domain="${domain.name}"></td>
    `;
    domainRows.appendChild(tr);
  });
}

function pickDomainInput(name, key) {
  return document.querySelector(`[data-domain="${name}"][data-k="${key}"]`);
}

function collectDomains() {
  return DOMAIN_DEFAULTS.map(({ name }) => ({
    domain: name,
    active: pickDomainInput(name, "active").checked,
    direction: pickDomainInput(name, "direction").value,
    strength: Number(pickDomainInput(name, "strength").value),
    confidence: Number(pickDomainInput(name, "confidence").value),
    weight: Number(pickDomainInput(name, "weight").value)
  }));
}

function applyPreset(presetKey) {
  const preset = PRESETS[presetKey] || PRESETS.balanced;

  document.getElementById("macroSuppression").checked = preset.macroSuppression;
  document.getElementById("macroDelta").value = preset.macroDelta.toFixed(2);
  document.getElementById("maxVeto").value = preset.maxVeto.toFixed(2);

  document.getElementById("minDomains").value = preset.minDomains;
  document.getElementById("minConfidence").value = preset.minConfidence.toFixed(2);
  document.getElementById("minStrength").value = preset.minStrength.toFixed(2);

  DOMAIN_DEFAULTS.forEach(({ name }) => {
    pickDomainInput(name, "active").checked = true;
    pickDomainInput(name, "direction").value = preset.domainDirection;
    pickDomainInput(name, "strength").value = preset.domainStrength.toFixed(2);
    pickDomainInput(name, "confidence").value = preset.domainConfidence.toFixed(2);
  });

  if (presetKey === "low_confluence") {
    pickDomainInput("macro_context", "active").checked = false;
    pickDomainInput("temporal", "active").checked = false;
    pickDomainInput("smc", "direction").value = "short";
    pickDomainInput("liquidity", "direction").value = "long";
  }
}

function addTrace(step, pass) {
  const li = document.createElement("li");
  li.textContent = `${pass ? "✅" : "❌"} ${step}`;
  traceList.appendChild(li);
}

function runSimulation() {
  traceList.innerHTML = "";

  const minDomains = Number(document.getElementById("minDomains").value);
  const minConfidence = Number(document.getElementById("minConfidence").value);
  const minStrength = Number(document.getElementById("minStrength").value);

  const macroSuppression = document.getElementById("macroSuppression").checked;
  const macroDelta = Number(document.getElementById("macroDelta").value);
  const maxVeto = Number(document.getElementById("maxVeto").value);

  const domains = collectDomains().filter((d) => d.active);

  if (macroSuppression) {
    addTrace("Hard macro veto check", false);
    return { decision: "NO TRADE", reason: "Hard macro veto (suppression flag true).", passing: false };
  }
  addTrace("Hard macro veto check", true);

  if (domains.length < minDomains) {
    addTrace(`Min active domains (${domains.length}/${minDomains})`, false);
    return {
      decision: "NO TRADE",
      reason: `Insufficient active domains (${domains.length} < ${minDomains}).`,
      passing: false
    };
  }
  addTrace(`Min active domains (${domains.length}/${minDomains})`, true);

  if (maxVeto >= 0.8) {
    addTrace(`Blocking veto severity (${maxVeto.toFixed(2)})`, false);
    return { decision: "NO TRADE", reason: `Blocking veto detected (severity ${maxVeto.toFixed(2)}).`, passing: false };
  }
  addTrace(`Blocking veto severity (${maxVeto.toFixed(2)})`, true);

  const dirScore = { long: 0, short: 0 };
  let weightedStrength = 0;
  let weightedConfidence = 0;
  let totalWeight = 0;

  for (const d of domains) {
    const impact = d.weight * d.strength * d.confidence;
    if (d.direction === "long") dirScore.long += impact;
    if (d.direction === "short") dirScore.short += impact;
    weightedStrength += d.weight * d.strength;
    weightedConfidence += d.weight * d.confidence;
    totalWeight += d.weight;
  }

  const direction = dirScore.long >= dirScore.short ? "long" : "short";
  addTrace(`Domain consensus direction (${direction})`, true);

  const baseStrength = totalWeight ? weightedStrength / totalWeight : 0;
  const baseConfidence = totalWeight ? weightedConfidence / totalWeight : 0;

  const finalStrength = Math.max(0, Math.min(1, baseStrength + macroDelta));
  const finalConfidence = Math.max(0, Math.min(1, baseConfidence + macroDelta));

  const passStrength = finalStrength >= minStrength;
  const passConfidence = finalConfidence >= minConfidence;

  addTrace(`Final strength ${finalStrength.toFixed(2)} >= ${minStrength.toFixed(2)}`, passStrength);
  addTrace(`Final confidence ${finalConfidence.toFixed(2)} >= ${minConfidence.toFixed(2)}`, passConfidence);

  const passing = passStrength && passConfidence;

  return {
    decision: passing ? `ENTER ${direction.toUpperCase()}` : "NO TRADE",
    reason: passing
      ? "Consensus and threshold checks passed."
      : `Threshold miss: strength ${finalStrength.toFixed(2)} / confidence ${finalConfidence.toFixed(2)}.`,
    passing,
    direction,
    activeDomains: domains.length,
    finalStrength,
    finalConfidence,
    longPressure: dirScore.long,
    shortPressure: dirScore.short,
    macroDelta,
    thresholds: { minDomains, minStrength, minConfidence }
  };
}

async function loadConfigSnapshot() {
  const path = document.getElementById("configPath").value;
  const configOutput = document.getElementById("configOutput");

  try {
    const response = await fetch(`../${path}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();

    const fusion = data.fusion || data;
    const domainWeights = fusion.domain_weights || data.domain_weights || null;
    const minDomains = fusion.min_domains ?? data.min_active_domains ?? 3;
    const minConfidence = fusion.min_confidence ?? data.entry_threshold ?? 0.65;
    const minStrength = fusion.min_strength ?? 0.6;

    if (domainWeights && typeof domainWeights === "object") {
      DOMAIN_DEFAULTS.forEach(({ name }) => {
        if (domainWeights[name] != null) pickDomainInput(name, "weight").value = Number(domainWeights[name]).toFixed(2);
      });
    }

    document.getElementById("minDomains").value = minDomains;
    document.getElementById("minConfidence").value = Number(minConfidence).toFixed(2);
    document.getElementById("minStrength").value = Number(minStrength).toFixed(2);

    configOutput.textContent = JSON.stringify({
      source: path,
      resolved: { minDomains, minConfidence, minStrength, domainWeights }
    }, null, 2);
  } catch (error) {
    configOutput.textContent = `Failed to load ${path}: ${error.message}`;
  }
}

document.getElementById("runBtn").addEventListener("click", () => {
  const out = runSimulation();
  document.getElementById("output").textContent = JSON.stringify(out, null, 2);
});

document.getElementById("applyPresetBtn").addEventListener("click", () => {
  applyPreset(document.getElementById("presetSelect").value);
});

document.getElementById("loadConfigBtn").addEventListener("click", loadConfigSnapshot);

renderDomainRows();
applyPreset("balanced");
