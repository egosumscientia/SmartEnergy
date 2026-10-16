(() => {
  const instances = new Map();
  const colors = { Normal: "#41c7a6", Anomalo: "#f07178", "Sin estado": "#96a8b7" };
  const labels = {
    corriente: "Corriente",
    voltaje: "Voltaje",
    potencia_activa: "Potencia activa",
    temperatura_motor: "Temperatura del motor",
  };

  function baseOption() {
    return {
      backgroundColor: "transparent",
      textStyle: { color: "#c2ced8", fontFamily: "Inter, system-ui, sans-serif" },
      animation: false,
      tooltip: { trigger: "axis", confine: true },
      grid: { left: 58, right: 22, top: 42, bottom: 58 },
    };
  }

  function getChart(id) {
    const element = document.getElementById(id);
    if (!element || !window.echarts) return null;
    let chart = instances.get(id);
    if (!chart) {
      chart = window.echarts.init(element, null, { renderer: "canvas" });
      instances.set(id, chart);
    }
    return chart;
  }

  function draw(id, option) {
    const chart = getChart(id);
    if (chart) chart.setOption(option, { notMerge: true, lazyUpdate: true });
  }

  function temporalOption(points, variable, predicted = false) {
    const option = baseOption();
    const states = predicted ? [0, 1] : ["Normal", "Anomalo", "Sin estado"];
    option.legend = { top: 4, textStyle: { color: "#bdc9d3" } };
    option.xAxis = {
      type: "time",
      axisLine: { lineStyle: { color: "#405365" } },
      axisLabel: { color: "#99aaba" },
      splitLine: { show: false },
    };
    option.yAxis = {
      type: "value",
      name: "Unidad no especificada",
      nameTextStyle: { color: "#99aaba" },
      axisLine: { show: false },
      axisLabel: { color: "#99aaba" },
      splitLine: { lineStyle: { color: "#253545" } },
    };
    option.dataZoom = [
      { type: "inside", filterMode: "none" },
      { type: "slider", height: 18, bottom: 13, borderColor: "#344759" },
    ];
    option.series = states.map((state) => ({
      name: predicted ? (state === 1 ? "Anómalo" : "Normal") : state,
      type: "line",
      showSymbol: predicted ? state === 1 : false,
      symbolSize: predicted ? 7 : 4,
      connectNulls: false,
      itemStyle: { color: predicted ? (state === 1 ? colors.Anomalo : colors.Normal) : colors[state] },
      lineStyle: { width: predicted && state === 1 ? 1 : 2 },
      data: points.map((point) => [
        point.timestamp,
        (predicted ? point.predicted_anomaly === state : point.state === state)
          ? point.values[variable]
          : null,
      ]),
    }));
    return option;
  }

  function lineChart(id, points, variable) {
    draw(id, temporalOption(points, variable));
  }

  function coverageChart(id, coverage) {
    draw(id, {
      backgroundColor: "transparent",
      series: [{
        type: "gauge",
        min: 0,
        max: 100,
        progress: { show: true, width: 14, itemStyle: { color: "#39b8a4" } },
        axisLine: { lineStyle: { width: 14, color: [[1, "#263747"]] } },
        axisTick: { show: false },
        splitLine: { distance: -19, length: 7, lineStyle: { color: "#91a6b5" } },
        axisLabel: { distance: 24, color: "#99aaba" },
        pointer: { show: false },
        anchor: { show: false },
        title: { show: true, offsetCenter: [0, "43%"], color: "#99aaba", fontSize: 12 },
        detail: { valueAnimation: false, formatter: "{value}%", color: "#e5edf4", fontSize: 24, offsetCenter: [0, "8%"] },
        data: [{ value: Number(coverage.toFixed(2)), name: "Cobertura original" }],
      }],
    });
  }

  function histogramChart(id, histogram) {
    if (!histogram.length) return;
    const bins = histogram[0].bins;
    const categories = bins.map((bin) => `${Number(bin.lower.toPrecision(4))}–${Number(bin.upper.toPrecision(4))}`);
    const option = baseOption();
    option.legend = { top: 4, textStyle: { color: "#bdc9d3" } };
    option.xAxis = {
      type: "category",
      data: categories,
      axisLabel: { color: "#99aaba", rotate: 35, hideOverlap: true },
      axisLine: { lineStyle: { color: "#405365" } },
    };
    option.yAxis = {
      type: "value",
      name: "Registros",
      axisLabel: { color: "#99aaba" },
      splitLine: { lineStyle: { color: "#253545" } },
    };
    option.series = histogram.map((group) => ({
      name: group.state,
      type: "bar",
      barMaxWidth: 18,
      itemStyle: { color: colors[group.state], opacity: 0.75 },
      data: group.bins.map((bin) => bin.count),
    }));
    draw(id, option);
  }

  function comparativeOption(points, histograms) {
    const variables = ["corriente", "voltaje", "potencia_activa", "temperatura_motor"];
    const marginLeft = 8;
    const marginTop = 5;
    const width = 21;
    const height = 21;
    const gap = 2;
    const option = baseOption();
    option.tooltip = { trigger: "item", confine: true };
    option.legend = { top: 2, textStyle: { color: "#bdc9d3" } };
    option.grid = [];
    option.xAxis = [];
    option.yAxis = [];
    option.series = [];

    variables.forEach((yVariable, row) => {
      variables.forEach((xVariable, column) => {
        const index = row * variables.length + column;
        const left = `${marginLeft + column * (width + gap)}%`;
        const top = `${marginTop + row * (height + gap)}%`;
        const diagonal = row === column;
        option.grid.push({
          left,
          top,
          width: `${width}%`,
          height: `${height}%`,
          containLabel: true,
        });
        option.xAxis.push({
          type: diagonal ? "category" : "value",
          gridIndex: index,
          data: diagonal
            ? (histograms[xVariable]?.[0]?.bins || []).map((bin) => `${Number(bin.lower.toPrecision(3))}`)
            : undefined,
          name: labels[xVariable],
          nameLocation: "middle",
          nameGap: 25,
          nameTextStyle: { color: "#8fa3b2", fontSize: 8 },
          axisLabel: { show: row === 3, color: "#91a6b5", fontSize: 8, hideOverlap: true },
          axisLine: { lineStyle: { color: "#3b5062" } },
          splitLine: { show: false },
        });
        option.yAxis.push({
          type: "value",
          gridIndex: index,
          name: column === 0 ? labels[yVariable] : "",
          nameTextStyle: { color: "#8fa3b2", fontSize: 8 },
          axisLabel: { show: column === 0, color: "#91a6b5", fontSize: 8 },
          axisLine: { show: false },
          splitLine: { lineStyle: { color: "#233342" } },
        });

        if (diagonal) {
          for (const group of histograms[xVariable] || []) {
            option.series.push({
              name: group.state,
              type: "bar",
              xAxisIndex: index,
              yAxisIndex: index,
              barMaxWidth: 5,
              itemStyle: { color: colors[group.state], opacity: 0.72 },
              data: group.bins.map((bin) => bin.count),
            });
          }
        } else {
          for (const state of ["Normal", "Anomalo", "Sin estado"]) {
            const data = points
              .filter((point) => point.state === state)
              .map((point) => [point.values[xVariable], point.values[yVariable]])
              .filter((pair) => pair[0] !== null && pair[1] !== null);
            if (data.length) {
              option.series.push({
                name: state,
                type: "scatter",
                xAxisIndex: index,
                yAxisIndex: index,
                symbolSize: 4,
                itemStyle: { color: colors[state], opacity: 0.68 },
                data,
              });
            }
          }
        }
      });
    });
    return option;
  }

  function comparativeChart(id, points, histograms) {
    draw(id, comparativeOption(points, histograms));
  }

  function anomalyChart(id, points, variable) {
    draw(id, temporalOption(points, variable, true));
  }

  function resizeVisible() {
    for (const chart of instances.values()) chart.resize();
  }

  function clear() {
    for (const chart of instances.values()) chart.clear();
  }

  window.addEventListener("resize", resizeVisible);
  window.SmartEnergyCharts = {
    lineChart,
    coverageChart,
    histogramChart,
    comparativeChart,
    anomalyChart,
    resizeVisible,
    clear,
  };
})();
