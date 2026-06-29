const DATA_URL = "ops-data.json";
let _data = null;

function el(tag, attrs, ...children) {
  const e = document.createElement(tag);
  if (attrs) Object.entries(attrs).forEach(([k, v]) => {
    if (k === "className") e.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(e.style, v);
    else if (k === "innerHTML") e.innerHTML = v;
    else if (k === "onclick") e.addEventListener("click", v);
    else e.setAttribute(k, v);
  });
  children.flat().forEach(c => {
    if (c == null) return;
    e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  });
  return e;
}

function badge(label, style) { return el("span", { className: `badge ${style}` }, label); }
function fmt(val, d = 3) { return val != null ? val.toFixed(d) : "\u2014"; }

function card(title, content, rawTitle) {
  const c = el("div", { className: "card" });
  const t = el("div", { className: "card-title" });
  if (rawTitle) t.innerHTML = title; else t.textContent = title;
  c.appendChild(t);
  if (typeof content === "string") c.appendChild(el("div", { innerHTML: content }));
  else c.appendChild(content);
  return c;
}

function metricsGrid(items) {
  const grid = el("div", { className: "metrics-grid" });
  items.forEach(([label, val, extra, cls]) => {
    const box = el("div", { className: "metric-box" },
      el("div", { className: "metric-label" }, label),
      el("div", { className: `metric-val ${cls || ""}` }, typeof val === "string" ? val : fmt(val)));
    if (extra) box.appendChild(el("div", { className: "metric-threshold" }, extra));
    grid.appendChild(box);
  });
  return grid;
}

// Modal
function openModal(title, contentEl) {
  closeModal();
  const overlay = el("div", { className: "modal-overlay", onclick: (e) => { if (e.target === overlay) closeModal(); } });
  const modal = el("div", { className: "modal" });
  modal.appendChild(el("div", { className: "modal-header" },
    el("div", { className: "modal-title", innerHTML: title }),
    el("button", { className: "modal-close", onclick: closeModal, innerHTML: "&times;" })));
  modal.appendChild(el("div", { className: "modal-body" }, contentEl));
  overlay.appendChild(modal);
  document.body.appendChild(overlay);
  document.body.style.overflow = "hidden";
}

function closeModal() {
  const existing = document.querySelector(".modal-overlay");
  if (existing) { existing.remove(); document.body.style.overflow = ""; }
}

document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });

// Zone 1: Status - is the model healthy right now?
function renderStatus(root, data) {
  const champion = (data.models || []).find(m => m.status === "Approved");
  if (!champion) return;

  const monYears = Object.keys(data.monitoring || {}).sort();
  const latestYear = monYears[monYears.length - 1];
  const latestMon = latestYear ? data.monitoring[latestYear] : null;

  const hasDrift = latestMon && latestMon.mq_violations && latestMon.mq_violations.length > 0;
  const statusLabel = hasDrift ? "Drift detected" : "Healthy";
  const statusCls = hasDrift ? "pink" : "accent";
  const statusDot = hasDrift ? "status-dot-red" : "status-dot-green";

  const section = el("div", { className: "status-bar" });
  section.appendChild(el("div", { className: `status-indicator ${statusDot}` }));

  const info = el("div", { className: "status-info" });
  info.appendChild(el("div", { className: "status-headline" },
    el("span", { className: `status-label ${statusCls}-text` }, statusLabel)));
  const detail = `Champion v${champion.version} (trained on HMDA ${champion.trained_on})`;
  const monDetail = latestMon
    ? "" : "";
  info.appendChild(el("div", { className: "status-detail" }, detail + monDetail));
  section.appendChild(info);

  root.appendChild(section);
}

// Zone 2: Champion Timeline - how did we get here?
function renderTimeline(root, data) {
  const models = [...(data.models || [])].sort((a, b) => a.version - b.version);
  if (models.length === 0) return;

  const content = el("div", { className: "timeline-track" });

  models.forEach((m, i) => {
    const isChamp = m.status === "Approved";
    const node = el("div", {
      className: `timeline-node ${isChamp ? "active" : ""} clickable`,
      onclick: () => showModelDetail(m),
    });

    const dot = el("div", { className: `tl-dot ${isChamp ? "accent" : "dim"}` });
    node.appendChild(dot);

    const label = el("div", { className: "tl-label" });
    label.appendChild(el("div", { className: `tl-version ${isChamp ? "" : "dim"}` }, `v${m.version}`));
    label.appendChild(el("div", { className: "tl-trained" }, m.trained_on || "?"));
    if (m.challenger_mae != null) {
      label.appendChild(el("div", { className: "tl-mae" }, `MAE ${fmt(m.challenger_mae)}`));
    }
    if (isChamp) {
      label.appendChild(badge("champion", "accent"));
    }
    node.appendChild(label);

    content.appendChild(node);

    if (i < models.length - 1) {
      content.appendChild(el("div", { className: "tl-connector" }));
    }
  });

  root.appendChild(card("Champion Timeline", content));
}

// Zone 3: Accuracy over time - the money chart
function renderAccuracy(root, data) {
  const monYears = Object.keys(data.monitoring || {}).sort();
  const recoveries = data.recoveries || {};
  if (monYears.length === 0) return;

  const content = el("div");

  // Build data points. Skip the training year because its monitoring MAE is train-contaminated.
  const baselineYear = data.baseline ? "2021" : null;
  const points = [];
  monYears.forEach(year => {
    if (year === baselineYear) return;
    const mon = data.monitoring[year];
    if (mon.mae == null) return;
    const r = recoveries[year];
    points.push({
      year,
      monitoring_mae: mon.mae,
      before_mae: r ? r.frozen_eval_mae : mon.mae,
      after_mae: r ? r.new_eval_mae : null,
      retrained: !!r,
      before_version: r ? r.frozen_version : mon.champion_version,
      new_version: r ? r.new_version : null,
      recovery_magnitude: r ? r.recovery_magnitude : null,
    });
  });

  if (points.length === 0) return;

  const maxMae = Math.max(...points.map(p => Math.max(p.before_mae || 0, p.after_mae || 0)), 0.5);
  const threshold = data.baseline ? data.baseline.mae_threshold : null;
  const threshPct = threshold ? Math.min(threshold / maxMae * 100, 100) : null;

  const legend = el("div", { className: "chart-legend" });
  legend.appendChild(el("div", { className: "legend-item" },
    el("div", { className: "legend-swatch", style: { background: "rgba(240,111,170,0.35)" } }),
    "Frozen champion (eval slice)"));
  legend.appendChild(el("div", { className: "legend-item" },
    el("div", { className: "legend-swatch", style: { background: "rgba(245,230,66,0.35)" } }),
    "Retrained (eval slice)"));
  content.appendChild(legend);

  points.forEach(p => {
    const beforePct = Math.min(p.before_mae / maxMae * 100, 100);
    const beforeCls = threshold && p.before_mae > threshold ? "pink" : "accent";

    const group = el("div", { className: "chart-group" });
    group.appendChild(el("span", { className: "chart-year" }, p.year));

    const bars = el("div", { className: "chart-bars" });

    const beforeRow = el("div", { className: "chart-bar-row" });
    beforeRow.appendChild(el("span", { className: "chart-version dim" }, p.before_version ? `v${p.before_version}` : ""));
    const beforeTrack = el("div", { className: "drift-bar-track" });
    if (threshPct) beforeTrack.appendChild(el("div", { className: "drift-threshold-line", style: { left: threshPct + "%" } }));
    beforeTrack.appendChild(el("div", { className: `drift-bar ${beforeCls}`, style: { width: beforePct + "%" } }));
    beforeRow.appendChild(beforeTrack);
    beforeRow.appendChild(el("span", { className: `drift-val ${beforeCls}-text` }, fmt(p.before_mae)));
    bars.appendChild(beforeRow);

    if (p.retrained) {
      const afterPct = Math.min(p.after_mae / maxMae * 100, 100);
      const afterCls = "accent";

      const afterRow = el("div", { className: "chart-bar-row" });
      afterRow.appendChild(el("span", { className: "chart-version dim" }, p.new_version ? `v${p.new_version}` : ""));
      const afterTrack = el("div", { className: "drift-bar-track" });
      if (threshPct) afterTrack.appendChild(el("div", { className: "drift-threshold-line", style: { left: threshPct + "%" } }));
      afterTrack.appendChild(el("div", { className: `drift-bar ${afterCls}`, style: { width: afterPct + "%" } }));
      afterRow.appendChild(afterTrack);
      afterRow.appendChild(el("span", { className: `drift-val ${afterCls}-text` }, fmt(p.after_mae)));
      bars.appendChild(afterRow);

      const mag = p.recovery_magnitude || 0;
      const annot = mag > 0
        ? `recovery +${fmt(mag, 4)}`
        : `no recovery ${fmt(mag, 4)}`;
      bars.appendChild(el("div", { className: "chart-annotation" }, annot));
    } else {
      bars.appendChild(el("div", { className: "chart-annotation" }, "monitoring MAE only"));
    }

    group.appendChild(bars);
    content.appendChild(group);
  });

  const notes = [];
  if (threshold) notes.push(`Dashed line = ${fmt(threshold)} MAE alarm threshold (baseline x 1.25).`);
  notes.push("Recovery measured on held-out eval slice (rows the retrained model never trained on). Monitoring table shows full-year MAE on all rows.");
  content.appendChild(el("div", { className: "drift-legend dim" }, notes.join(" ")));

  root.appendChild(card("Accuracy Over Time (eval slice)", content));
}

// Zone 4: Per-vintage detail - drill down
function renderVintages(root, data) {
  const monYears = Object.keys(data.monitoring || {}).sort();
  if (monYears.length === 0) return;

  const table = el("table");
  table.appendChild(el("tr", null,
    el("th", null, "Year"),
    el("th", null, "Model"),
    el("th", null, "Data Drift (A)"),
    el("th", null, "Model Quality (B)"),
    el("th", null, "MAE (full year)"),
    el("th", null, "Status")));

  monYears.forEach(year => {
    const m = data.monitoring[year];
    const dqCount = m.dq_violations ? m.dq_violations.length : 0;
    const mqCount = m.mq_violations ? m.mq_violations.length : 0;
    const hasDrift = mqCount > 0;
    const status = hasDrift ? badge("degraded", "pink") : badge("healthy", "accent");
    const maeCls = data.baseline && m.mae > data.baseline.mae_threshold ? "pink-text" : "accent-text";
    const modelLabel = m.champion_version ? `v${m.champion_version}` : "\u2014";

    const tr = el("tr", { className: "clickable", onclick: () => showVintageDetail(year) },
      el("td", null, year),
      el("td", null, modelLabel),
      el("td", null, `${dqCount} violation${dqCount !== 1 ? "s" : ""}`),
      el("td", null, `${mqCount} violation${mqCount !== 1 ? "s" : ""}`),
      el("td", { className: maeCls }, fmt(m.mae)),
      el("td", null, status));
    table.appendChild(tr);
  });

  root.appendChild(card("Monitoring", table));
}

// Vintage detail modal
function showVintageDetail(year) {
  const data = _data;
  const m = data.monitoring[year];
  if (!m) return;

  const content = el("div");
  const bl = data.baseline;

  // Metrics
  const sec1 = el("div", { className: "modal-section" });
  sec1.appendChild(el("div", { className: "modal-section-title" },
    `Metrics (${(m.item_count || 0).toLocaleString()} rows, champion v${m.champion_version || "?"})`));
  const maeCls = bl && m.mae > bl.mae_threshold ? "pink-text" : "accent-text";
  const rmseCls = bl && m.rmse > bl.rmse_threshold ? "pink-text" : "accent-text";
  const r2Cls = bl && m.r2 < bl.r2_threshold ? "pink-text" : "accent-text";
  sec1.appendChild(metricsGrid([
    ["MAE", m.mae, bl ? `threshold ${fmt(bl.mae_threshold)}` : null, maeCls],
    ["RMSE", m.rmse, bl ? `threshold ${fmt(bl.rmse_threshold)}` : null, rmseCls],
    ["R\u00b2", m.r2, bl ? `threshold ${fmt(bl.r2_threshold)}` : null, r2Cls],
  ]));
  content.appendChild(sec1);

  // Violations
  const sec2 = el("div", { className: "modal-section" });
  const split = el("div", { className: "monitor-split" });

  const dqCol = el("div");
  dqCol.appendChild(el("div", { className: "monitor-col-title" },
    `Data Quality (A) \u2014 ${m.dq_violations.length} violation${m.dq_violations.length !== 1 ? "s" : ""}`));
  if (m.dq_violations.length === 0) {
    dqCol.appendChild(el("div", { className: "dim" }, "No violations"));
  } else {
    m.dq_violations.forEach(v => {
      const short = v.check && v.check.includes("drift") ? "drift" : "completeness";
      dqCol.appendChild(el("div", { className: "viol-item" },
        el("span", { className: "viol-feat" }, v.feature), ` (${short})`));
    });
  }
  split.appendChild(dqCol);

  const mqCol = el("div");
  mqCol.appendChild(el("div", { className: "monitor-col-title" },
    `Model Quality (B) \u2014 ${m.mq_violations.length} violation${m.mq_violations.length !== 1 ? "s" : ""}`));
  if (m.mq_violations.length === 0) {
    mqCol.appendChild(el("div", { className: "dim" }, "No violations"));
  } else {
    m.mq_violations.forEach(v => {
      mqCol.appendChild(el("div", { className: "viol-item" },
        el("span", { className: "viol-feat" }, v.metric)));
    });
  }
  split.appendChild(mqCol);
  sec2.appendChild(split);
  content.appendChild(sec2);

  // Feature drift (if any)
  const driftData = data.drift && data.drift[year];
  if (driftData && driftData.length > 0) {
    const sec3 = el("div", { className: "modal-section" });
    sec3.appendChild(el("div", { className: "modal-section-title" }, "Feature Drift"));
    const means21 = (data.feature_means && data.feature_means["2021"]) || {};
    const meansY = (data.feature_means && data.feature_means[year]) || {};
    const threshold = data.drift_threshold || 0.1;
    const thresholdPct = threshold / 0.35 * 100;

    driftData.forEach(d => {
      const dist = d.distance || 0;
      const pct = Math.min(dist / 0.35 * 100, 100);
      const barCls = dist > threshold ? "pink" : "accent";
      const m21 = means21[d.feature], mY = meansY[d.feature];
      const shift = (m21 && mY && m21 !== 0) ? ((mY - m21) / Math.abs(m21) * 100) : null;
      const shiftStr = shift != null ? ` (${shift > 0 ? "+" : ""}${shift.toFixed(0)}%)` : "";

      const row = el("div", { className: "drift-row" });
      row.appendChild(el("span", { className: "drift-feat", innerHTML: `${d.feature}<span class="dim">${shiftStr}</span>` }));
      const track = el("div", { className: "drift-bar-track" });
      track.appendChild(el("div", { className: "drift-threshold-line", style: { left: thresholdPct + "%" } }));
      track.appendChild(el("div", { className: `drift-bar ${barCls}`, style: { width: pct + "%" } }));
      row.appendChild(track);
      row.appendChild(el("span", { className: `drift-val ${barCls}-text` }, dist.toFixed(3)));
      sec3.appendChild(row);
    });
    content.appendChild(sec3);
  }

  // Recovery (if any)
  const r = data.recoveries && data.recoveries[year];
  if (r) {
    const sec4 = el("div", { className: "modal-section" });
    sec4.appendChild(el("div", { className: "modal-section-title" },
      `Recovery (${r.eval_rows.toLocaleString()} eval rows)`));
    sec4.appendChild(metricsGrid([
      ["Before", r.frozen_eval_mae, "frozen champion", "pink-text"],
      ["After", r.new_eval_mae, "retrained", "accent-text"],
      ["Recovery", `${r.recovery_magnitude > 0 ? "+" : ""}${fmt(r.recovery_magnitude, 4)}`, null],
    ]));
    const note = r.recovery_magnitude < 0.05
      ? "Marginal recovery \u2014 degradation is structural (feature gap), not model staleness. Retraining alone cannot recover; the current features don't capture what changed."
      : `Retrained model clawed back ${(r.recovery_magnitude / r.frozen_eval_mae * 100).toFixed(0)}% of degradation.`;
    sec4.appendChild(el("div", { className: "recovery-summary" }, note));
    content.appendChild(sec4);
  }

  // Evidently report links
  const reports = data.report_links && data.report_links[year];
  if (reports) {
    const sec = el("div", { className: "modal-section" });
    sec.appendChild(el("div", { className: "modal-section-title" }, "Evidently Reports"));
    const links = el("div", { className: "report-links" });
    if (reports.data_quality) {
      links.appendChild(el("a", { href: reports.data_quality, target: "_blank", className: "report-link" }, "Data Drift Report"));
    }
    if (reports.model_quality) {
      links.appendChild(el("a", { href: reports.model_quality, target: "_blank", className: "report-link" }, "Model Quality Report"));
    }
    sec.appendChild(links);
    content.appendChild(sec);
  }

  const mqCount = m.mq_violations ? m.mq_violations.length : 0;
  const statusBadge = mqCount > 0
    ? '<span class="badge pink">degraded</span>'
    : '<span class="badge accent">healthy</span>';
  openModal(`${year} ${statusBadge}`, content);
}

// Model detail modal
function showModelDetail(model) {
  const data = _data;
  const content = el("div");

  // Model info
  const info = el("div", { className: "modal-section" });
  info.appendChild(el("div", { className: "modal-section-title" }, "Model Info"));
  info.appendChild(metricsGrid([
    ["Data", `HMDA ${model.trained_on || "?"}`],
    ["Objective", model.objective || "\u2014"],
    ["Features", model.num_features != null ? `${model.num_features}` : "\u2014"],
    ["Train Rows", model.train_rows != null ? model.train_rows.toLocaleString() : "\u2014"],
    ["Val Rows", model.val_rows != null ? model.val_rows.toLocaleString() : "\u2014"],
    ["Split Key", model.group_split_key || "\u2014"],
  ]));
  content.appendChild(info);

  // Evaluation metrics
  if (model.challenger_mae != null || model.challenger_rmse != null) {
    const met = el("div", { className: "modal-section" });
    met.appendChild(el("div", { className: "modal-section-title" }, "Evaluation Metrics (val set)"));
    met.appendChild(metricsGrid([
      ["MAE", model.challenger_mae, null, "accent-text"],
      ["RMSE", model.challenger_rmse, null, "accent-text"],
    ]));
    content.appendChild(met);
  }

  // Model quality drift - only years where this model was the champion
  const monYears = Object.keys(data.monitoring || {}).sort().filter(year => {
    const m = data.monitoring[year];
    return m.champion_version === model.version && m.mae != null;
  });
  if (monYears.length > 0 && data.baseline) {
    const sec = el("div", { className: "modal-section" });
    sec.appendChild(el("div", { className: "modal-section-title" }, "Model Quality Drift"));

    const bl = data.baseline;
    const maxMae = Math.max(...monYears.map(y => data.monitoring[y].mae), bl.mae_threshold, 0.5);

    monYears.forEach(year => {
      const mon = data.monitoring[year];
      const breached = mon.mae > bl.mae_threshold;
      const barCls = breached ? "pink" : "accent";
      const pct = Math.min(mon.mae / maxMae * 100, 100);
      const threshPct = Math.min(bl.mae_threshold / maxMae * 100, 100);

      const row = el("div", { className: "drift-row" });
      row.appendChild(el("span", { className: "drift-feat", style: { width: "4rem" } }, year));
      const track = el("div", { className: "drift-bar-track" });
      track.appendChild(el("div", { className: "drift-threshold-line", style: { left: threshPct + "%" } }));
      track.appendChild(el("div", { className: `drift-bar ${barCls}`, style: { width: pct + "%" } }));
      row.appendChild(track);
      row.appendChild(el("span", { className: `drift-val ${barCls}-text` }, fmt(mon.mae)));
      sec.appendChild(row);
    });

    sec.appendChild(el("div", { className: "drift-legend dim" },
      `Dashed line = ${fmt(bl.mae_threshold)} MAE threshold.`));
    content.appendChild(sec);
  }

  const isChamp = model.status === "Approved";
  const statusBadge = isChamp ? '<span class="badge accent">champion</span>' :
    model.status === "Rejected" ? '<span class="badge pink">rejected</span>' : '<span class="badge purple">pending</span>';
  openModal(`v${model.version} ${statusBadge}`, content);
}

async function init() {
  const root = document.getElementById("dashboard");
  try {
    const resp = await fetch(DATA_URL);
    if (!resp.ok) throw new Error(`Failed to load data: ${resp.status}`);
    _data = await resp.json();

    const genAt = document.getElementById("generated-at");
    if (genAt) genAt.textContent = _data.generated_at || "";

    renderStatus(root, _data);
    renderTimeline(root, _data);
    renderAccuracy(root, _data);
    renderVintages(root, _data);
  } catch (err) {
    root.appendChild(el("div", { className: "card" },
      el("p", { style: { color: "#f06faa", fontFamily: "'DM Mono', monospace", fontSize: "0.85rem" } },
        `Error: ${err.message}`)));
  }
}

document.addEventListener("DOMContentLoaded", init);
