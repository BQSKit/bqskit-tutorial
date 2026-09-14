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
    # cyclosynth takes rotation angles, not matrices, so go through the
    # U3 parameterization (cyclosynth uses the qiskit/bqskit U3 convention).
    theta, phi, lam = U3Gate().calc_params(UnitaryMatrix(un))

    # Use cyclosynth to synthesize the unitary
    synth = cyclosynth.Synthesizer(epsilon=epsilon)
    result = synth.synthesize_u3(theta, phi, lam)
    if result and result.gates is not None:
        # In cyclosynth's gate string the leftmost gate is the leftmost matrix
        # factor, so append in reverse to build the same product in BQSKit.
        gate_map = {
            "H": HGate(), "S": SGate(), "T": TGate(),
            "X": XGate(), "Y": YGate(), "Z": ZGate(),
            "s": SdgGate(), "t": TdgGate(),
        }
        circ = Circuit(1)
        for gate_str in reversed(result.gates):
            if gate_str not in gate_map:
                raise ValueError(
                    f"cyclosynth returned an unhandled gate {gate_str!r} "
                    f"in {result.gates!r}."
                )
            circ.append_gate(gate_map[gate_str], [0])
        return circ, True
    circ = Circuit(1)
    circ.append_gate(U3Gate(), [0], params=[theta, phi, lam])
    return circ, False

class U3ToTPass(BasePass):
    """Convert all U3 Gates to Clifford + T using cyclosynth"""

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        num_u3s = circuit.count(U3Gate())

        # Nothing to do on a block with no U3 gates (e.g. an all-CNOT block)
        if num_u3s == 0:
            return

        # Try to get error from data
        algorithmic_error = None
        for key in data.keys():
            if 'algorithmic_error' in key:
                algorithmic_error = data[key]
                break

        if algorithmic_error is None:
            raise ValueError('Error must be set to run U3ToTPass.')

        error_per_gate = algorithmic_error / num_u3s

        # Round error per gate to lower power of 10
        error_per_gate = 10 ** np.floor(np.log10(error_per_gate))

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
