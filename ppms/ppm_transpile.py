"""This module implements the GeneralSQDecomposition."""
from __future__ import annotations

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.generalgate import GeneralGate

from ppm_transpiler import transpile_circuit, TransItem
from .ppm import PPMPlaceholder  # adjust import to your structure

def trans_items_to_placeholders(
    layers: list[list[TransItem]],
) -> list[tuple[PPMPlaceholder, list[int], float, int]]:
    """
    Convert layers of PyTransItems into a flattened list of PPMPlaceholders.

    Each item in the output is a tuple with PPMPlaceholder gate, the
    corresponding qubits, and the sign parameter. Note that only bases that 
    are not the identity will be included, and the qubits are determined by 
    the index of the item in the layer.
    """
    result = []
    for layer in layers:
        # Skip pure Clifford layers
        if all(item.is_clifford for item in layer):
            continue
        for item in layer:
            if item.is_pauli:
                # Only include non-identity bases, tracking qubit indices
                non_identity = [
                    (idx, basis) for idx, basis in enumerate(item.ops)
                    if basis != 'I'
                ]
                qubits = [idx for idx, _ in non_identity]
                bases = [basis for _, basis in non_identity]
                if item.label == "T":
                    angle = 0.25  # T gate corresponds to pi/4 rotation
                elif item.label == "M":
                    angle = 0.0  # Measurement placeholder, no rotation
                else:
                    angle = 0.0  # Default to 0 for other cases
                result.append((PPMPlaceholder(bases=bases), qubits, [angle, float(item.sign == '+')]))
    return result


class PPMTranspilePass(BasePass):
    """Convert a Clifford + T circuit into a sequence of Pauli Product 
    Measurements"""

    def __init__(self, max_width: int = -1) -> None:
        """Construct a PPMTranspilePass."""
        self.max_width = max_width

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        # Convert circuit to format that can be read by the PPM transpiler
        qasm = circuit.to("qasm")

        # Pass in qasm to transpiler (parse_and_transpile connects to Rust)
        items = transpile_circuit(qasm, self.max_width)
        placeholders = trans_items_to_placeholders(items)

        # Now flatten and add to Circuit appropriately
        new_circuit = Circuit(circuit.num_qudits)
        for item in placeholders:
            placeholder, qubits, params = item
            new_circuit.append_gate(placeholder, location=qubits, params=params)

        circuit.become(new_circuit)