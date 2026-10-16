(() => {
  class ApiError extends Error {
    constructor(message, status) {
      super(message);
      this.name = "ApiError";
      this.status = status;
    }
  }

  function buildParams(form, includeStates, csvOnly = false) {
    const values = new FormData(form);
    const params = new URLSearchParams();
    const keys = csvOnly
      ? ["date_from", "date_to"]
      : [
      "date_from",
      "date_to",
      "variable",
      "sample_plots",
      "smoothing_method",
      "rolling_window",
      "resample_minutes",
      "table_limit",
      ];
    for (const key of keys) {
      const value = values.get(key);
      if (value !== null && value !== "") {
        params.set(key, value);
      }
    }
    if (!csvOnly) {
      params.set(
        "sample_plots",
        form.querySelector("#sample-plots").checked ? "true" : "false",
      );
    }
    if (includeStates) {
      const selected = [...form.querySelectorAll('input[name="states"]:checked')];
      if (selected.length === 0) {
        params.set("states", "");
      } else {
        for (const checkbox of selected) {
          params.append("states", checkbox.value);
        }
      }
    }
    return params;
  }

  async function request(path, params) {
    let response;
    try {
      response = await fetch(`${path}?${params.toString()}`, {
        headers: { Accept: "application/json" },
        credentials: "same-origin",
      });
    } catch {
      throw new ApiError("No se pudo conectar con el servicio de monitoreo.", 0);
    }

    let body;
    try {
      body = await response.json();
    } catch {
      throw new ApiError("El servicio devolvió una respuesta no válida.", response.status);
    }
    if (!response.ok) {
      throw new ApiError(
        body?.error?.message || "No se pudo completar la consulta.",
        response.status,
      );
    }
    return body;
  }

  async function getMonitoring(form, includeStates) {
    return request("/api/v1/monitoring", buildParams(form, includeStates));
  }

  function csvUrl(form, includeStates) {
    return `/api/v1/monitoring/records.csv?${buildParams(form, includeStates, true).toString()}`;
  }

  window.SmartEnergyApi = { ApiError, getMonitoring, csvUrl };
})();
