"""Rotated surface-code memory circuits with a single-qubit Pauli channel of chosen bias on the data qubits
(phenomenological model), in the CSS frame or in the XZZX frame.

Noise placement: stim's generated circuit with all built-in noise off, rebuilt with
  - PAULI_CHANNEL_1(p_X, p_Y, p_Z) on every data qubit before every round (where stim puts
    before_round_data_depolarization),
  - X_ERROR(q) before every stabiliser-ancilla measurement (MR); the final data measurement is ideal,
  - ideal gates.
XZZX frame: the Clifford-deformed surface code (Bonilla Ataides et al., Nat. Commun. 12, 2172 (2021), arXiv:2009.07851;
Dua et al., arXiv:2201.07802) is the CSS code with a Hadamard on the data qubits of one diagonal family. Rather than
rewriting the stabilisers, the deformation is applied to the NOISE: on the deformed qubits (x − y)/2 even in stim's
coordinates, (p_X, p_Z) are swapped. Detectors and the observable are those of the CSS circuit. With ideal gates this is
exact: H maps X ↔ Z and leaves Y alone. Every face of the rotated lattice then reads X on its main-diagonal corners and
Z on the other two — the XZZX code — which `deformed_face_pattern` checks.
"""
from __future__ import annotations

import stim


def data_qubits(circuit: stim.Circuit) -> list:
    """Qubits of the final data measurement (M or MX)."""
    out = []
    for inst in circuit.flattened():
        if inst.name in ("M", "MX"):
            out = [t.value for t in inst.targets_copy()]
    return out


def deformed_qubits(circuit: stim.Circuit) -> set:
    co = circuit.get_final_qubit_coordinates()
    return {q for q in data_qubits(circuit) if int(round((co[q][0] - co[q][1]) / 2)) % 2 == 0}


def biased_circuit(distance: int, rounds: int, basis: str, p_x: float, p_y: float, p_z: float, q: float,
                   xzzx: bool = False) -> stim.Circuit:
    base = stim.Circuit.generated(f"surface_code:rotated_memory_{basis}", distance=distance, rounds=rounds,
                                  before_round_data_depolarization=0.5,          # markers, replaced below
                                  before_measure_flip_probability=0.5)
    deformed = deformed_qubits(base) if xzzx else set()
    data = set(data_qubits(base))

    def rebuild(block: stim.Circuit) -> stim.Circuit:
        out = stim.Circuit()
        for inst in block:
            if isinstance(inst, stim.CircuitRepeatBlock):
                out.append(stim.CircuitRepeatBlock(inst.repeat_count, rebuild(inst.body_copy())))
                continue
            if inst.name == "DEPOLARIZE1":                                   # the data-qubit marker
                targets = [t.value for t in inst.targets_copy()]
                if p_x == p_z or not deformed:                              # one instruction, stim's target order
                    out.append("PAULI_CHANNEL_1", targets, [p_x, p_y, p_z])
                    continue
                out.append("PAULI_CHANNEL_1", [t for t in targets if t not in deformed], [p_x, p_y, p_z])
                out.append("PAULI_CHANNEL_1", [t for t in targets if t in deformed], [p_z, p_y, p_x])
                continue
            if inst.name in ("X_ERROR", "Z_ERROR"):                          # the measurement-flip marker
                targets = [t.value for t in inst.targets_copy()]
                if set(targets) <= data:                                     # final data measurement: ideal
                    continue
                if q > 0:
                    out.append("X_ERROR", targets, q)
                continue
            out.append(inst)
        return out

    return rebuild(base)


def deformed_face_pattern(distance: int) -> dict:
    """For every stabiliser face of the rotated code (any basis), the Pauli each data corner carries after the
    deformation, keyed by the ancilla coordinate: {(x, y): {(dx, dy): 'X'|'Z'}} with (dx, dy) ∈ {±1}^2 the corner
    offset. The face's native type comes from the circuit (X-type ancillas are the ones stim wraps in H)."""
    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=distance, rounds=1)
    co = c.get_final_qubit_coordinates()
    data = set(data_qubits(c))
    deformed = deformed_qubits(c)
    x_type = set()
    for inst in c.flattened():
        if inst.name == "H":
            x_type |= {t.value for t in inst.targets_copy()}
    pos = {q: (co[q][0], co[q][1]) for q in co}
    by_pos = {pos[q]: q for q in co}
    faces = {}
    for a in co:
        if a in data:
            continue
        native = "X" if a in x_type else "Z"
        ax, ay = pos[a]
        corners = {}
        for dx in (-1, 1):
            for dy in (-1, 1):
                qq = by_pos.get((ax + dx, ay + dy))
                if qq is None or qq not in data:
                    continue
                pauli = native
                if qq in deformed:
                    pauli = "Z" if native == "X" else "X"
                corners[(dx, dy)] = pauli
        faces[(ax, ay)] = corners
    return faces


def parallel_edges_with_different_effect(dem: stim.DetectorErrorModel) -> dict:
    """Groups of error mechanisms (after decomposition into graph-like components) that light the SAME detectors but
    flip DIFFERENT sets of observables. PyMatching merges such parallel edges keeping the logical effect of the first
    one in the model (its docstring; PyMatching#103). Returns counts."""
    effects = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        comps, cur = [], []
        for t in inst.targets_copy():
            if t.is_separator():
                comps.append(cur)
                cur = []
            else:
                cur.append(t)
        comps.append(cur)
        for comp in comps:
            dets = frozenset(t.val for t in comp if t.is_relative_detector_id())
            obs = frozenset(t.val for t in comp if t.is_logical_observable_id())
            if not dets:
                continue
            effects.setdefault(dets, set()).add(obs)
    bad = {k: v for k, v in effects.items() if len(v) > 1}
    return {"detector_sets": len(effects), "parallel_edge_sets_with_different_effect": len(bad)}


def code_distance(circuit: stim.Circuit, max_size: int = 6) -> int:
    """Smallest number of error mechanisms that flips the observable without lighting a detector (stim's search)."""
    errs = circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=max_size,
        dont_explore_edges_with_degree_above=9999,
        dont_explore_edges_increasing_symptom_degree=False,
        canonicalize_circuit_errors=True)
    return len(errs)
