"""This module implements the GeneralSQDecomposition."""
from __future__ import annotations
import numpy as np

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit, CircuitGate, CircuitPoint
from bqskit.ir.operation import Operation
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.gates.constant import HGate, TGate, SGate, XGate, YGate, ZGate, TdgGate, SdgGate
import cyclosynth

from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


def cyclo_decompose(un: np.ndarray, epsilon: float) -> tuple[Circuit, bool]:
    """
    Decompose a unitary into a circuit of Clifford + T gates using cyclosynth.

    Args:
        un (np.ndarray): The unitary matrix to decompose.
        epsilon (float): The approximation error tolerance.

    Returns:
        Circuit: A circuit that approximates the given unitary.
    """
    # Use cyclosynth to synthesize the unitary
    synth = cyclosynth.Synthesizer(epsilon=epsilon)
    result = synth.synthesize(un)
    if result:
        circ = Circuit(1)
        for gate_str in reversed(result.gates):
            if gate_str == "H":
                circ.append_gate(HGate(), [0])
            elif gate_str == "T":
                circ.append_gate(TGate(), [0])
            elif gate_str == "S":
                circ.append_gate(SGate(), [0])
            elif gate_str == "X":
                circ.append_gate(XGate(), [0])
            elif gate_str == "Y":
                circ.append_gate(YGate(), [0])
            elif gate_str == "Z":
                circ.append_gate(ZGate(), [0])
            elif gate_str == "t":
                circ.append_gate(TdgGate(), [0])
            elif gate_str == "s":
                circ.append_gate(SdgGate(), [0])
        return circ, True
    circ = Circuit(1)
    circ.append_gate(U3Gate(), [0], params=U3Gate().calc_params(UnitaryMatrix(un)))
    return circ, False

class U3ToTPass(BasePass):
    """Convert all U3 Gates to Clifford + T using cyclosynth"""

    def __init__(self, epsilon: float = 1e-9) -> None:
        """Construct a U3ToTPass."""
        self.epsilon = epsilon

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        error_per_gate = self.epsilon / circuit.count(U3Gate())

        # Round error per gate to lower power of 10
        error_per_gate = 10 ** np.floor(np.log10(error_per_gate))

        # print(f"Decomposing U3 gates with error tolerance: {error_per_gate}")

        if error_per_gate < 1e-6:
            print(f"Warning: Decomposing U3 gates with error tolerance: {error_per_gate}. This may take a long time so we will skip for now.")
            return

        pts_to_replace = []
        new_ops = []
        for cycle, op in circuit.operations_with_cycles():
            if isinstance(op.gate, U3Gate):
                # Decompose the GeneralGate into U3 gates
                un = op.get_unitary()
                u3_circuit, success = cyclo_decompose(un.numpy, error_per_gate)
                if success:
                    pts_to_replace.append(CircuitPoint(cycle, op.location[0]))
                    new_ops.append(Operation(CircuitGate(u3_circuit), op.location))
                circuit.replace_with_circuit((cycle, op.location[0]), u3_circuit,
                                             as_circuit_gate=True) 

        circuit.batch_replace(pts_to_replace, new_ops)
        circuit.unfold_all()  # Unfold the circuit to remove any nested circuits
