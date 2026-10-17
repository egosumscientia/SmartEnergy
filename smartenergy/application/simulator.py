"""Composición S06 paso 7: sólo lo recibido, sin acceso a verdad del ensayo.

La admisión S03 queda bloqueada por canonicalización pendiente. reception es
metadato previo a admisión, NO una entrega S03 con status inventado. Cada elemento
conserva origen y evaluación por separado; no es almacén ni proyección legada.
"""

from smartenergy.analysis.simulator_acquisition import Delivery
from smartenergy.analysis.simulator_contract import ObservationConfig, observation_from_reading


def compose_received(deliveries, config: ObservationConfig, *, lineage_refs):
    """Recibe sólo Delivery públicas y refs (capture_id, channel) → string opaco.

    Conserva orden y multiplicidad sin decidir duplicado/conflicto ni último.
    Silencios/fallos no llegan a esta interfaz; no originan missing ni ceros.
    Devuelve diccionarios independientes, sin retener estado entre llamadas.
    """
    if not isinstance(config, ObservationConfig):
        raise ValueError("Configuración S03 requerida")
    result = []
    for delivery in deliveries:
        if not isinstance(delivery, Delivery):
            raise ValueError("Se requieren recepciones públicas")
        capture = delivery.capture
        if delivery.received_s < capture.produced_s:
            raise ValueError("Recepción anterior a producción")
        for reading in capture.readings:
            key = (capture.capture_id, reading.channel)
            if key not in lineage_refs:
                raise ValueError("Referencia de linaje ausente")
            observation = observation_from_reading(capture, reading, config,
                                                   lineage_ref=lineage_refs[key])
            observation_key = [config.installation_id, capture.device, observation["observation_id"]]
            freshness = reading.freshness_at(delivery.received_s)
            reason = ("age_exceeded" if freshness == "stale" else
                      "time_unknown" if freshness == "unknown" else None)
            # Las razones de origen (p.ej. range_violation) permanecen intactas.
            result.append(dict(
                observation=observation,
                reception=dict(contract_version="1.0.0",
                               delivery_id=f'{delivery.delivery_id}/observation/{observation["observation_id"]}',
                               observation_key=observation_key,
                               received_at=config.timestamp(delivery.received_s), recorded_at=None,
                               transport_ref=delivery.delivery_id),
                evaluation=dict(observation_key=list(observation_key), freshness=freshness,
                                reason=reason, evidence_ref=delivery.delivery_id,
                                assessment=dict(at=config.timestamp(delivery.received_s),
                                                max_age_s=5, policy_ref="s05-age@1"))))
    return tuple(result)
