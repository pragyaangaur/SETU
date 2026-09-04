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
  var LIVE = DATA.live || null;

  // Most recent storm first, because that is the one a reader wants to see.
  var replays = Object.keys(DATA)
    .filter(function (k) { return k.indexOf("replay_") === 0; })
    .sort().reverse()
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

  function renderLive() {
    if (!LIVE) { return; }
    var c = LIVE.conditions || {};
    el("liveCard").hidden = false;
    var rows = [
      ["Measured at", c.observed_at + " UTC"],
      ["Reporting spacecraft", (c.sources || []).join(", ") || "unknown"],
      ["Field magnitude", c.b_total_nt + " nT"],
      ["Southward field", c.southward_field_nt + " nT"],
      ["Speed", c.speed_km_s + " km/s"],
      ["Reaches the Earth in", c.propagation_delay_min + " minutes"]
    ];
    var check = LIVE.delay_check || {};
    el("live").innerHTML =
      '<table><tbody>' + rows.map(function (r) {
        return "<tr><td>" + r[0] + "</td><td class='num'>" + r[1] + "</td></tr>";
      }).join("") + "</tbody></table>" +
      '<p class="note">Snapshot taken ' + (LIVE.generated_at || "").slice(0, 16) +
      ". The travel time above is computed from spacecraft distance and measured speed. " +
      "Checked against the operational feed's own propagated product, the two agree to " +
      (check.median_difference_min !== undefined ? check.median_difference_min : "?") +
      " minutes in the median.</p>";
  }


  /* ---------- the standing record ----------
   *
   * This is the only part of the console that is not a replay of a storm that has
   * already been studied. The service writes a forecast into the ledger every
   * quarter of an hour, before its outcome exists, and attaches what the ground
   * actually did once the valid minute has passed.
   *
   * The file is fetched rather than read from the bundle, because the service
   * commits it on its own schedule and the bundle is only rebuilt by hand. When
   * the page is opened from the file system a browser will not allow the fetch, so
   * the bundled copy is used and the panel says which one it is looking at.
   */

  function renderLedger(ledger, stale) {
    var host = el("ledger");
    if (!ledger || !ledger.scoreboard) {
      host.innerHTML = '<p class="empty">No forecast record yet. Run ' +
        "<code>python -m setu.cli nowcast</code> to open one.</p>";
      return;
    }
    var board = ledger.scoreboard;
    var latest = ledger.latest || {};
    var horizons = ["30", "45", "60", "90"];

    var alarm = horizons.some(function (h) {
      return latest.horizons && latest.horizons[h] && latest.horizons[h].alarm;
    });

    var head =
      '<div class="grid g3" style="margin-bottom:14px">' +
      card(board.forecasts_issued.toLocaleString(),
           "forecasts issued over " + board.days_running + " days, " +
           board.forecasts_issued_live + " of them written before the outcome existed") +
      card(board.horizon_forecasts_verified.toLocaleString(),
           "of them scored against what the ground magnetometer actually recorded") +
      card(alarm ? "ALARM" : "Quiet",
           "state at " + (latest.issued_at || "?") + " UTC, from " +
           ((latest.sources || []).join(", ") || "the operational monitor"),
           alarm ? "var(--bad)" : "var(--good)") +
      "</div>";

    var rows = horizons.map(function (h) {
      var b = board.horizons[h] || {};
      var m = b.model, p = b.persistence;
      if (!m || !m.n) {
        return "<tr><td>" + h + " min</td><td class='num'>0</td>" +
               "<td colspan='4' class='num'>nothing scored yet</td></tr>";
      }
      return "<tr><td>" + h + " min</td>" +
        "<td class='num'>" + m.n.toLocaleString() + "</td>" +
        "<td class='num'>" + b.events + "</td>" +
        "<td class='num'>" + m.false_alarms + "</td>" +
        "<td class='num'>" + p.false_alarms + "</td>" +
        "<td class='num'>" + fmt(b.mean_absolute_error, 3) + "</td>" +
        "<td class='num'>" + (m.hss === null ? "&mdash;" : fmt(m.hss, 2)) +
          " / " + (p.hss === null ? "&mdash;" : fmt(p.hss, 2)) + "</td></tr>";
    }).join("");

    var table =
      "<div class='scroll'><table><thead><tr>" +
      "<th>Horizon</th><th>Minutes scored</th><th>Times the ground passed 0.1 nT/s</th>" +
      "<th>False alarms, model</th><th>False alarms, persistence</th>" +
      "<th>Mean error, nT/s</th><th>Heidke, model / persistence</th>" +
      "</tr></thead><tbody>" + rows + "</tbody></table></div>";

    var note = board.quiet
      ? "The ground has not passed 0.1 nT per second once since this record opened, " +
        "so there is no detection skill to report yet and the table shows what the " +
        "system does in quiet conditions instead. That is the number an operator " +
        "asks about first, because an alarm system that cries wolf through a quiet " +
        "week will be switched off before the storm arrives."
      : "Skill is shown against the persistence baseline on exactly the same minutes. " +
        "Every row was written down before its outcome existed.";

    var source = stale
      ? "This page was opened from the file system, so it is showing the copy " +
        "bundled at build time rather than the live one."
      : "Updated by the service every quarter of an hour. Last written " +
        (ledger.written_at || "").slice(0, 16) + " UTC.";

    host.innerHTML = head + table +
      '<p class="note">' + note + " " + source + "</p>" +
      '<p class="note">' + (ledger.what_this_is || "") + "</p>";
  }

  function card(value, note, colour) {
    return '<div class="card" style="background:var(--panel2)">' +
      '<div class="kpi"' + (colour ? ' style="color:' + colour + '"' : "") + ">" +
      value + "<small>" + note + "</small></div></div>";
  }

  function loadLedger() {
    if (!window.fetch || location.protocol === "file:") {
      renderLedger(DATA.ledger, true);
      return;
    }
    fetch("data/ledger.json", { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(r.status); })
      .then(function (live) { renderLedger(live, false); })
      .catch(function () { renderLedger(DATA.ledger, true); });
  }

  function renderLeadTime() {
    var host = el("leadtime");
    var r = state.replay;
    if (!r || !r.lead_time) { host.innerHTML = '<p class="empty">Run the replay command to export this.</p>'; return; }

    var blocks = Object.keys(r.lead_time).map(function (level) {
      var s = r.lead_time[level];
      if (s.episodes === undefined || s.episodes === 0) {
        return '<div class="card" style="background:var(--panel2)"><h2>' + level +
          " nT per second</h2><p class=\"note\">" +
          (s.note || "nothing to report") + "</p></div>";
      }
      var cls = s.warned === s.episodes ? "p-good" : s.warned ? "p-warn" : "p-bad";
      var lead = s.median_lead_minutes === null ? "none" : s.median_lead_minutes + " min";
      return '<div class="card" style="background:var(--panel2)">' +
        "<h2>" + level + " nT per second</h2>" +
        '<div class="kpi">' + lead +
        ' <span class="pill ' + cls + '">' + s.warned + " of " + s.episodes +
        " warned</span><small>median warning before a disturbed period. " +
        (s.missed ? s.missed + " period" + (s.missed > 1 ? "s" : "") + " missed." : "None missed.") +
        "</small></div></div>";
    });
    host.innerHTML = '<div class="grid g2">' + blocks.join("") + "</div>";
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
    renderLeadTime();
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
    renderLive();
    loadLedger();
    renderAll();
  }

  function setupScrub() {
    var scrub = el("scrub");
    var n = state.replay ? state.replay.steps.length - 1 : 0;
    scrub.max = String(Math.max(0, n));
    // Open on the moment the ground was roughest, because that is what a reader
    // wants to see first and it is also where the forecast is worth judging.
    if (state.replay) {
      var worst = 0, best = -1;
      state.replay.steps.forEach(function (s, i) {
        if (s.observed_dbdt > best) { best = s.observed_dbdt; worst = i; }
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
