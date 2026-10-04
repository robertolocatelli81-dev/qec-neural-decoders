"""A rotated surface-code memory circuit with long-range correlated errors added.

On top of stim's uniform circuit-level noise, after EVERY single-qubit depolarising instruction of the generated circuit
(three per round: the data-qubit depolarisation and the two after the Hadamard layers on the measure qubits), each chosen
PAIR of far-apart data qubits suffers a joint X error with probability `pc` (stim CORRELATED_ERROR X_a X_b); per pair and
round the joint error happens with probability 1 - (1 - pc)^3 (two insertions in the first round, one before the final
data measurement). For a Z-basis memory such an error flips the Z stabiliser next to each of the two qubits: TWO detectors
plus the logical observable, i.e. an ordinary (long) edge of the matching graph, not a hyper-edge. At distance 3 that
syndrome is also the syndrome of a single X error on the central data qubit, which does NOT flip the observable: the two
hypotheses become parallel edges with different logical effect, and a decoder must choose per syndrome (see README).
`pc = 0` is the plain circuit.
"""
from __future__ import annotations

import itertools

import stim


def _data_qubits(circuit: stim.Circuit) -> list:
    """The qubits measured by the final M (data qubits of a memory experiment)."""
    data = []
    for inst in circuit.flattened():
        if inst.name == "M":
            data = [t.value for t in inst.targets_copy()]
    return data


def far_pairs(circuit: stim.Circuit, n_pairs: int) -> list:
    """The n_pairs data-qubit pairs with the largest distance, from stim's QUBIT_COORDS (deterministic order)."""
    coords = circuit.get_final_qubit_coordinates()
    data = _data_qubits(circuit)
    pairs = sorted(itertools.combinations(sorted(data), 2),
                   key=lambda ab: (-sum((x - y) ** 2 for x, y in zip(coords[ab[0]], coords[ab[1]])), ab))
    return pairs[:n_pairs]


def correlated_circuit(distance: int, rounds: int, p: float, pc: float, n_pairs: int = 2) -> stim.Circuit:
    base = stim.Circuit.generated("surface_code:rotated_memory_z", distance=distance, rounds=rounds,
                                  after_clifford_depolarization=p, before_round_data_depolarization=p,
                                  before_measure_flip_probability=p, after_reset_flip_probability=p)
    if pc == 0:
        return base
    pairs = far_pairs(base, n_pairs)

    def rebuild(block: stim.Circuit) -> stim.Circuit:
        out = stim.Circuit()
        for inst in block:
            if isinstance(inst, stim.CircuitRepeatBlock):
                out.append(stim.CircuitRepeatBlock(inst.repeat_count, rebuild(inst.body_copy())))
                continue
            out.append(inst)
            if inst.name == "DEPOLARIZE1":                     # after EVERY single-qubit depolarisation (three per round)
                for a, b in pairs:
                    out.append("CORRELATED_ERROR", [stim.target_x(a), stim.target_x(b)], pc)
        return out

    return rebuild(base)
