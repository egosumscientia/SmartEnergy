(() => {
  const form = document.getElementById("filters-form");
  const stateOptions = document.getElementById("state-options");
  const requestMessage = document.getElementById("request-message");
  const connectionStatus = document.getElementById("connection-status");
  const tabs = [...document.querySelectorAll(".view-tabs [role='tab']")];
  let stateFiltersInitialized = false;
  let latestPayload = null;

  const variableLabels = {
    corriente: "Corriente",
    voltaje: "Voltaje",
    potencia_activa: "Potencia activa",
    temperatura_motor: "Temperatura del motor",
  };

  function formatNumber(value, maximumFractionDigits = 2) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "N/A";
    return new Intl.NumberFormat("es", { maximumFractionDigits }).format(Number(value));
  }

  function setNotice(message, error = false) {
    requestMessage.textContent = message;
    requestMessage.classList.toggle("notice-error", error);
    requestMessage.hidden = !message;
  }

  function setConnection(message, status) {
    connectionStatus.textContent = message;
    connectionStatus.className = `status-pill status-${status}`;
  }

  function renderStateFilters(availableStates) {
    const selected = new Set(
      [...stateOptions.querySelectorAll('input[name="states"]:checked')].map((item) => item.value),
    );
    stateOptions.replaceChildren();
    for (const state of availableStates) {
      const label = document.createElement("label");
      label.className = "state-option";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.name = "states";
      checkbox.value = state;
      checkbox.checked = stateFiltersInitialized ? selected.has(state) : true;
      const text = document.createElement("span");
      text.textContent = state;
      label.append(checkbox, text);
      stateOptions.append(label);
    }
    if (availableStates.length === 0) {
      const message = document.createElement("span");
      message.className = "muted";
      message.textContent = "No hay estados disponibles.";
      stateOptions.append(message);
    }
    stateFiltersInitialized = true;
  }

  function setEmpty(chartId, emptyId, visible, message) {
    const chart = document.getElementById(chartId);
    const empty = document.getElementById(emptyId);
    chart.hidden = !visible;
    empty.hidden = visible;
    empty.textContent = message;
  }

  function populateTable(records) {
    const body = document.getElementById("records-body");
    body.replaceChildren();
    const empty = document.getElementById("records-empty");
    empty.hidden = records.length > 0;
    empty.textContent = "No hay registros disponibles para los filtros seleccionados.";
    const fields = [
      "id",
      "timestamp",
      "corriente",
      "voltaje",
      "potencia_activa",
      "temperatura_motor",
      "estado",
      "label_anomalia",
    ];
    const numericFields = new Set([
      "corriente",
      "voltaje",
      "potencia_activa",
      "temperatura_motor",
    ]);
    for (const record of records) {
      const row = document.createElement("tr");
      for (const field of fields) {
        const cell = document.createElement("td");
        const value = record[field];
        cell.textContent = value === null || value === undefined
          ? "—"
          : numericFields.has(field)
            ? formatNumber(value, 4)
            : String(value);
        if (field === "estado") {
          cell.className = value === "Anomalo"
            ? "state-anomalo"
            : value === "Normal"
              ? "state-normal"
              : "state-unknown";
        }
        row.append(cell);
      }
      body.append(row);
    }
  }

  function updateSummary(payload) {
    document.getElementById("record-count").textContent = formatNumber(payload.data.record_count, 0);
    document.getElementById("filtered-count").textContent = formatNumber(payload.data.filtered_count, 0);

    const modelLabel = {
      available: "Disponible",
      absent: "No disponible",
      unavailable: "No disponible",
    }[payload.model.status];
    document.getElementById("model-status").textContent = modelLabel;
    const modelDetail = document.getElementById("model-detail");
    if (payload.model.status === "available") {
      modelDetail.textContent = payload.model.modified_at
        ? `Archivo actualizado: ${payload.model.modified_at}`
        : "Artefacto de modelo disponible";
    } else if (payload.model.status === "unavailable") {
      modelDetail.textContent = "El artefacto existe, pero no está disponible para evaluación.";
    } else {
      modelDetail.textContent = "No se encontró un modelo.";
    }

    const latest = payload.persisted_metrics.latest;
    const latestMetric = document.getElementById("latest-metric");
    const latestDetail = document.getElementById("latest-metric-detail");
    if (latest) {
      latestMetric.textContent = `${formatNumber(latest.percentage)}%`;
      const previous = payload.persisted_metrics.previous_percentage;
      const delta = previous === null
        ? ""
        : ` · variación ${formatNumber(latest.percentage - previous)} puntos porcentuales`;
      latestDetail.textContent = `${latest.state} · ${latest.recorded_at}${delta}`;
    } else if (payload.persisted_metrics.status === "unavailable") {
      latestMetric.textContent = "N/A";
      latestDetail.textContent = "No se pudo consultar el historial opcional.";
    } else {
      latestMetric.textContent = "N/A";
      latestDetail.textContent = "No hay métricas persistidas.";
    }

    const evaluation = payload.model.evaluation;
    const count = document.getElementById("anomaly-count");
    const coverage = document.getElementById("coverage-value");
    const state = document.getElementById("evaluation-state");
    const originalAnomalies = document.getElementById("anomaly-original-count");
    count.textContent = evaluation.status === "available"
      ? formatNumber(evaluation.detected, 0)
      : "N/A";
    originalAnomalies.textContent = count.textContent;
    coverage.textContent = evaluation.status === "available"
      ? `${formatNumber(evaluation.coverage)}%`
      : "N/A";
    state.textContent = evaluation.status === "available"
      ? evaluation.state
      : payload.model.status === "absent"
        ? "ML no disponible"
        : evaluation.reason === "no_data"
          ? "Sin población"
          : evaluation.reason === "incomplete_features"
            ? "Variables incompletas"
            : "Análisis no disponible";
    document.getElementById("last-analysis-note").textContent = latest
      ? `Último análisis persistido: ${latest.recorded_at} · estado: ${latest.state}`
      : "Las métricas persistidas son históricas y opcionales; no equivalen a la evaluación en memoria.";
  }

  function renderCharts(payload) {
    const points = payload.chart.points;
    const variable = payload.chart.variable;
    const hasVariableValues = points.some((point) => point.values[variable] !== null);
    const caption = `${variableLabels[variable]} · unidad no especificada · ${formatNumber(payload.chart.point_count, 0)} puntos representados`;
    document.getElementById("overview-chart-caption").textContent = caption;
    document.getElementById("energy-chart-caption").textContent = caption;

    setEmpty(
      "overview-chart",
      "overview-empty",
      hasVariableValues,
      payload.data.has_filtered_data
        ? "No hay valores y fechas válidos para representar esta variable."
        : "No hay mediciones para los filtros seleccionados.",
    );
    if (hasVariableValues) window.SmartEnergyCharts.lineChart("overview-chart", points, variable);

    const evaluation = payload.model.evaluation;
    const canShowCoverage = evaluation.status === "available";
    setEmpty(
      "coverage-chart",
      "coverage-empty",
      canShowCoverage,
      payload.model.status === "absent"
        ? "La cobertura no está disponible sin un modelo."
        : "No hay evaluación válida para la población filtrada.",
    );
    if (canShowCoverage) {
      window.SmartEnergyCharts.coverageChart("coverage-chart", evaluation.coverage);
    }

    setEmpty(
      "energy-chart",
      "energy-empty",
      hasVariableValues,
      payload.data.has_filtered_data
        ? "No hay valores y fechas válidos para esta variable."
        : "No hay mediciones para los filtros seleccionados.",
    );
    if (hasVariableValues) window.SmartEnergyCharts.lineChart("energy-chart", points, variable);

    const histogram = payload.chart.histograms[variable] || [];
    const hasHistogram = histogram.some((series) => series.bins.some((bin) => bin.count > 0));
    setEmpty(
      "histogram-chart",
      "histogram-empty",
      hasHistogram,
      "No hay valores válidos para construir el histograma.",
    );
    if (hasHistogram) window.SmartEnergyCharts.histogramChart("histogram-chart", histogram);

    const completePoints = points.filter((point) =>
      ["corriente", "voltaje", "potencia_activa", "temperatura_motor"]
        .every((key) => point.values[key] !== null),
    );
    setEmpty(
      "comparison-chart",
      "comparison-empty",
      completePoints.length > 0,
      "No hay puntos completos para comparar las variables.",
    );
    if (completePoints.length) {
      window.SmartEnergyCharts.comparativeChart(
        "comparison-chart",
        completePoints,
        payload.chart.comparison_histograms,
      );
    }

    const anomalyPanelAvailable =
      evaluation.status === "available" &&
      payload.chart.prediction_status === "available" &&
      points.some((point) => point.predicted_anomaly !== null);
    document.getElementById("anomaly-point-count").textContent = anomalyPanelAvailable
      ? formatNumber(points.filter((point) => point.predicted_anomaly === 1).length, 0)
      : "N/A";
    setEmpty(
      "anomaly-chart",
      "anomaly-empty",
      anomalyPanelAvailable,
      payload.model.status === "absent"
        ? "No hay un modelo ML disponible. Las demás vistas continúan operativas."
        : payload.data.has_filtered_data
          ? "No fue posible evaluar puntos completos de la gráfica."
          : "No hay mediciones para los filtros seleccionados.",
    );
    if (anomalyPanelAvailable) {
      window.SmartEnergyCharts.anomalyChart("anomaly-chart", points, variable);
    }
  }

  function render(payload) {
    latestPayload = payload;
    renderStateFilters(payload.data.available_states);
    if (!document.getElementById("date-from").value && payload.data.date_range.from_date) {
      document.getElementById("date-from").value = payload.data.date_range.from_date;
    }
    if (!document.getElementById("date-to").value && payload.data.date_range.to_date) {
      document.getElementById("date-to").value = payload.data.date_range.to_date;
    }
    updateSummary(payload);
    renderCharts(payload);
    populateTable(payload.table.records);
    document.getElementById("table-caption").textContent =
      `Máximo ${formatNumber(payload.table.limit, 0)} filas; más nuevos primero.`;
    document.getElementById("csv-count").textContent =
      `CSV: ${formatNumber(payload.data.filtered_count, 0)} registros filtrados`;
    document.getElementById("csv-download").href =
      window.SmartEnergyApi.csvUrl(form, stateFiltersInitialized);
    const serviceAvailable = payload.database === "available";
    setConnection(
      serviceAvailable ? "Servicio disponible" : "Servicio no disponible",
      serviceAvailable ? "online" : "error",
    );
    const warning = payload.model.status === "unavailable"
      ? "El modelo no está disponible; las vistas de mediciones siguen operativas."
      : payload.persisted_metrics.status === "unavailable"
        ? "No se pudo consultar el historial opcional de métricas."
        : "";
    setNotice(warning);
    window.SmartEnergyCharts.resizeVisible();
  }

  async function refresh() {
    const refreshButton = document.getElementById("refresh-button");
    const submitButton = form.querySelector('button[type="submit"]');
    refreshButton.disabled = true;
    submitButton.disabled = true;
    setConnection("Consultando", "waiting");
    try {
      const payload = await window.SmartEnergyApi.getMonitoring(form, stateFiltersInitialized);
      render(payload);
    } catch (error) {
      setConnection("Servicio no disponible", "error");
      setNotice(
        latestPayload
          ? `${error.message} Se conservan los últimos resultados cargados.`
          : error.message,
        true,
      );
      if (!latestPayload) {
        window.SmartEnergyCharts.clear();
        for (const [chartId, emptyId] of [
          ["overview-chart", "overview-empty"],
          ["coverage-chart", "coverage-empty"],
          ["energy-chart", "energy-empty"],
          ["histogram-chart", "histogram-empty"],
          ["comparison-chart", "comparison-empty"],
          ["anomaly-chart", "anomaly-empty"],
        ]) {
          setEmpty(chartId, emptyId, false, error.message);
        }
      }
    } finally {
      refreshButton.disabled = false;
      submitButton.disabled = false;
    }
  }

  function showTab(button) {
    const panelId = button.dataset.panel;
    for (const tab of tabs) {
      const selected = tab === button;
      tab.setAttribute("aria-selected", String(selected));
      document.getElementById(tab.getAttribute("aria-controls")).hidden = !selected;
    }
    requestAnimationFrame(() => window.SmartEnergyCharts.resizeVisible());
    document.getElementById(panelId).focus({ preventScroll: true });
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    refresh();
  });
  document.getElementById("refresh-button").addEventListener("click", refresh);
  for (const tab of tabs) {
    tab.addEventListener("click", () => showTab(tab));
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
      event.preventDefault();
      const step = event.key === "ArrowRight" ? 1 : -1;
      const index = (tabs.indexOf(tab) + step + tabs.length) % tabs.length;
      tabs[index].focus();
      showTab(tabs[index]);
    });
  }
  window.addEventListener("beforeunload", () => window.SmartEnergyCharts.resizeVisible());
  refresh();
})();
