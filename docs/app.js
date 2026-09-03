/* Rendering for the SETU operator console.
 *
 * The page has no build step and no dependencies. Charts are drawn as inline SVG
 * so that the whole console works from the file system during a demonstration
 * where the network cannot be trusted.
 */
(function () {
  "use strict";

  var DATA = window.SETU_DATA || {};
  var NETWORK = DATA.network || { substations: [], lines: [] };
  var PLACEMENT = DATA.placement || null;

  var replays = Object.keys(DATA)
    .filter(function (k) { return k.indexOf("replay_") === 0; })
    .map(function (k) { return { key: k, data: DATA[k] }; });

  var state = { replay: replays[0] ? replays[0].data : null, index: 0 };

  function el(id) { return document.getElementById(id); }
  function fmt(v, d) { return Number(v).toFixed(d === undefined ? 2 : d); }

  function severityColour(amps) {
    if (amps >= 5) return "var(--bad)";
    if (amps >= 2) return "var(--warn)";
    return "var(--good)";
  }

  function probabilityClass(p) {
    if (p >= 0.5) return "p-bad";
    if (p >= 0.2) return "p-warn";
    return "p-good";
  }

  /* ---------- key figures ---------- */

  function renderKpis() {
    var step = currentStep();
    var box = el("kpis");
    if (!step) { box.innerHTML = ""; return; }
    var p = step.probability;
    var levels = Object.keys(p).sort(function (a, b) { return a - b; });
    var mid = levels[1];

    var cards = [
      {
        title: "Probability the ground passes " + mid + " nT per second",
        value: Math.round(p[mid] * 100) + "%",
        note: "at " + state.replay.horizon_minutes + " minutes ahead, from the forecast distribution",
        cls: probabilityClass(p[mid])
      },
      {
        title: "Largest current per phase per transformer",
        value: fmt(step.peak_per_phase_amp, 2) + " A",
        note: "at " + step.worst_site + ", from the median forecast",
        cls: step.peak_per_phase_amp >= 5 ? "p-bad" : step.peak_per_phase_amp >= 2 ? "p-warn" : "p-good"
      },
      {
        title: "Extra reactive power the region must absorb",
        value: Math.round(step.reactive_mvar) + " MVAr",
        note: step.reserve_exhausted.length
          ? "reserve used up at " + step.reserve_exhausted.join(", ")
          : "every site still inside its own reactive reserve",
        cls: step.reserve_exhausted.length ? "p-warn" : "p-good"
      }
    ];

    box.innerHTML = cards.map(function (c) {
      return '<div class="card"><h2>' + c.title + '</h2>' +
        '<div class="kpi">' + c.value +
        ' <span class="pill ' + c.cls + '">' + (c.cls === "p-bad" ? "act" : c.cls === "p-warn" ? "watch" : "quiet") + '</span>' +
        '<small>' + c.note + '</small></div></div>';
    }).join("");
  }

  /* ---------- forecast chart ---------- */

  function renderChart() {
    var r = state.replay;
    var host = el("chart");
    if (!r) { host.innerHTML = '<p class="empty">No replay has been exported yet. Run the replay command.</p>'; return; }

    var steps = r.steps;
    var W = 1160, H = 300, padL = 52, padR = 14, padT = 14, padB = 30;
    var iw = W - padL - padR, ih = H - padT - padB;

    var maxV = 0.05;
    steps.forEach(function (s) {
      maxV = Math.max(maxV, s.observed_dbdt, s.forecast_quantiles[s.forecast_quantiles.length - 1]);
    });
    maxV *= 1.08;

    function X(i) { return padL + (i / Math.max(1, steps.length - 1)) * iw; }
    function Y(v) { return padT + ih - (Math.min(v, maxV) / maxV) * ih; }

    var loIdx = 0, hiIdx = 4, medIdx = 2;
    var upper = steps.map(function (s, i) { return X(i) + "," + Y(s.forecast_quantiles[hiIdx]); });
    var lower = steps.map(function (s, i) { return X(i) + "," + Y(s.forecast_quantiles[loIdx]); }).reverse();
    var band = '<polygon points="' + upper.concat(lower).join(" ") + '" fill="var(--accent)" opacity="0.22"/>';

    function line(values, colour, width, dash) {
      var pts = values.map(function (v, i) { return X(i) + "," + Y(v); }).join(" ");
      return '<polyline points="' + pts + '" fill="none" stroke="' + colour +
        '" stroke-width="' + width + '" stroke-linejoin="round"' +
        (dash ? ' stroke-dasharray="' + dash + '"' : "") + "/>";
    }

    var median = line(steps.map(function (s) { return s.forecast_quantiles[medIdx]; }), "var(--accent)", 1.8);
    var observed = line(steps.map(function (s) { return s.observed_dbdt; }), "var(--ink)", 1.4);

    var ticks = "";
    for (var k = 0; k <= 4; k++) {
      var v = maxV * k / 4;
      ticks += '<line x1="' + padL + '" x2="' + (W - padR) + '" y1="' + Y(v) + '" y2="' + Y(v) +
        '" stroke="var(--line)" stroke-width="1"/>' +
        '<text x="' + (padL - 8) + '" y="' + (Y(v) + 4) + '" text-anchor="end" font-size="10" fill="var(--dim)">' + fmt(v, 2) + "</text>";
    }

    var alert = 0.3;
    var alertLine = alert < maxV
      ? '<line x1="' + padL + '" x2="' + (W - padR) + '" y1="' + Y(alert) + '" y2="' + Y(alert) +
        '" stroke="var(--warn)" stroke-width="1.2" stroke-dasharray="5 4"/>'
      : "";

    var cx = X(state.index);
    var cursor = '<line x1="' + cx + '" x2="' + cx + '" y1="' + padT + '" y2="' + (padT + ih) +
      '" stroke="var(--extreme)" stroke-width="1.5"/>';

    var labels = "";
    [0, Math.floor(steps.length / 2), steps.length - 1].forEach(function (i) {
      if (!steps[i]) return;
      labels += '<text x="' + X(i) + '" y="' + (H - 8) + '" text-anchor="middle" font-size="10" fill="var(--dim)">' +
        steps[i].time.slice(5, 16).replace("T", " ") + "</text>";
    });

    host.innerHTML = '<svg viewBox="0 0 ' + W + " " + H + '" width="100%" role="img" ' +
      'aria-label="Forecast distribution of the ground magnetic rate of change against what was observed">' +
      ticks + alertLine + band + median + observed + cursor + labels +
      '<text x="12" y="' + (padT + 10) + '" font-size="10" fill="var(--dim)" transform="rotate(-90 12 ' + (padT + 10) + ')">nT per second</text>' +
      "</svg>";
  }

  /* ---------- map ---------- */

  function renderMap() {
    var step = currentStep();
    var host = el("map");
    var subs = NETWORK.substations || [];
    if (!subs.length) { host.innerHTML = '<p class="empty">Run the benchmark command to export the network.</p>'; return; }

    var lats = subs.map(function (s) { return s.lat; });
    var lons = subs.map(function (s) { return s.lon; });
    var minLat = Math.min.apply(null, lats) - 0.4, maxLat = Math.max.apply(null, lats) + 0.4;
    var minLon = Math.min.apply(null, lons) - 0.4, maxLon = Math.max.apply(null, lons) + 0.4;
    var W = 640, H = 420, pad = 26;

    function px(lon) { return pad + (lon - minLon) / (maxLon - minLon) * (W - 2 * pad); }
    function py(lat) { return H - pad - (lat - minLat) / (maxLat - minLat) * (H - 2 * pad); }

    var byCode = {};
    subs.forEach(function (s) { byCode[s.code] = s; });

    var lines = (NETWORK.lines || []).map(function (l) {
      var a = byCode[l.from], b = byCode[l.to];
      if (!a || !b) return "";
      var w = l.kv >= 400 ? 2 : 1;
      return '<line x1="' + px(a.lon) + '" y1="' + py(a.lat) + '" x2="' + px(b.lon) + '" y2="' + py(b.lat) +
        '" stroke="var(--line)" stroke-width="' + w + '"/>';
    }).join("");

    var amps = step ? step.per_site_amp : {};
    var nodes = subs.map(function (s) {
      var a = amps[s.code] || 0;
      var r = 4 + Math.min(14, Math.sqrt(a) * 4.5);
      return '<circle cx="' + px(s.lon) + '" cy="' + py(s.lat) + '" r="' + fmt(r, 1) +
        '" fill="' + severityColour(a) + '" opacity="0.8" stroke="var(--bg)" stroke-width="1"><title>' +
        s.name + " (" + s.state + ")\n" + fmt(a, 2) + " A per phase per transformer\n" +
        s.transformer_type.replace(/_/g, " ") + "\nground: " + s.earth_model.replace(/_/g, " ") +
        "</title></circle>";
    }).join("");

    host.innerHTML = '<svg viewBox="0 0 ' + W + " " + H + '" width="100%" role="img" ' +
      'aria-label="Map of the North East Region substations coloured by induced current">' +
      lines + nodes + "</svg>";
  }

  /* ---------- tables and plan ---------- */

  function renderSites() {
    var step = currentStep();
    var host = el("siteTable");
    if (!step) { host.innerHTML = ""; return; }
    var byCode = {};
    (NETWORK.substations || []).forEach(function (s) { byCode[s.code] = s; });

    var rows = Object.keys(step.per_site_amp).map(function (code) {
      return { code: code, amps: step.per_site_amp[code], meta: byCode[code] || {} };
    }).sort(function (a, b) { return b.amps - a.amps; }).slice(0, 10);

    var maxA = rows.length ? rows[0].amps : 1;
    host.innerHTML = "<thead><tr><th>Substation</th><th>Transformer</th><th>Ground</th>" +
      '<th style="text-align:right">A per phase</th></tr></thead><tbody>' +
      rows.map(function (r) {
        var w = Math.max(2, (r.amps / (maxA || 1)) * 100);
        return "<tr><td>" + (r.meta.name || r.code) + "</td>" +
          "<td>" + String(r.meta.transformer_type || "").replace(/_/g, " ") + "</td>" +
          "<td>" + String(r.meta.earth_model || "").replace(/_/g, " ") + "</td>" +
          '<td class="num">' + fmt(r.amps, 2) +
          '<div class="bar"><i style="width:' + w + "%;background:" + severityColour(r.amps) + '"></i></div></td></tr>';
      }).join("") + "</tbody>";
  }

  function renderPlan() {
    var step = currentStep();
    var host = el("plan");
    if (!step) { host.innerHTML = ""; return; }
    if (!step.plan || !step.plan.actions.length) {
      host.innerHTML = '<p class="empty">No action recommended. The forecast risk at this moment does not justify the cost of acting, which is the right answer for most of any storm.</p>';
      return;
    }
    var p = step.plan;
    host.innerHTML = p.actions.map(function (a) {
      return '<div class="action"><b>' + a.label + "</b><span>" +
        a.cost_lakh + " lakh rupees, needs " + a.lead_time_min + " minutes</span></div>";
    }).join("") +
      '<table style="margin-top:12px"><tbody>' +
      "<tr><td>Tail risk without action</td><td class='num'>" + p.baseline_tail_lakh + " lakh</td></tr>" +
      "<tr><td>Tail risk with the plan</td><td class='num'>" + p.planned_tail_lakh + " lakh</td></tr>" +
      "<tr><td>Cost of the plan</td><td class='num'>" + p.action_cost_lakh + " lakh</td></tr>" +
      "<tr><td>Load exposed, before and after</td><td class='num'>" +
      p.baseline_load_at_risk_mw + " to " + p.planned_load_at_risk_mw + " MW</td></tr>" +
      (p.benefit_to_cost ? "<tr><td>Benefit for each rupee spent</td><td class='num'>" + p.benefit_to_cost + "</td></tr>" : "") +
      "</tbody></table>";
  }

  function renderPlacement() {
    var host = el("placement");
    if (!PLACEMENT) { host.innerHTML = '<p class="empty">Run the placement command to export this.</p>'; return; }
    var steps = PLACEMENT.greedy.steps;
    var single = PLACEMENT.single_site_ranking.slice(0, steps.length);
    host.innerHTML = "<table><thead><tr><th>Devices</th><th>Site the search picks</th>" +
      "<th style='text-align:right'>Cut</th><th>Site a one at a time ranking picks</th></tr></thead><tbody>" +
      steps.map(function (s, i) {
        var alt = single[i] ? single[i].name : "";
        var differs = alt && alt !== s.name;
        return "<tr><td class='num'>" + s.devices + "</td><td>" + s.name + "</td>" +
          "<td class='num'>" + s.reduction_percent + "%</td>" +
          "<td" + (differs ? ' style="color:var(--extreme)"' : "") + ">" + alt + "</td></tr>";
      }).join("") + "</tbody></table>";
  }

  /* ---------- wiring ---------- */

  function currentStep() {
    if (!state.replay) return null;
    return state.replay.steps[Math.min(state.index, state.replay.steps.length - 1)];
  }

  function renderAll() {
    var step = currentStep();
    el("stamp").textContent = step ? step.time.replace("T", " ").slice(0, 16) + " UTC" : "";
    renderKpis(); renderChart(); renderMap(); renderSites(); renderPlan();
  }

  function init() {
    var select = el("eventSelect");
    if (!replays.length) {
      select.innerHTML = "<option>no replay exported</option>";
    } else {
      select.innerHTML = replays.map(function (r, i) {
        return '<option value="' + i + '">' + r.data.event.name + " (" + r.data.event.key + ")</option>";
      }).join("");
      select.addEventListener("change", function () {
        state.replay = replays[Number(select.value)].data;
        state.index = 0;
        setupScrub();
        renderAll();
      });
    }
    setupScrub();
    el("scrub").addEventListener("input", function (e) {
      state.index = Number(e.target.value);
      renderAll();
    });
    renderPlacement();
    renderAll();
  }

  function setupScrub() {
    var scrub = el("scrub");
    var n = state.replay ? state.replay.steps.length - 1 : 0;
    scrub.max = String(Math.max(0, n));
    // Open on the most exposed moment of the storm, because that is what a
    // reader wants to see first.
    if (state.replay) {
      var worst = 0, best = -1;
      state.replay.steps.forEach(function (s, i) {
        if (s.peak_per_phase_amp > best) { best = s.peak_per_phase_amp; worst = i; }
      });
      state.index = worst;
    }
    scrub.value = String(state.index);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
