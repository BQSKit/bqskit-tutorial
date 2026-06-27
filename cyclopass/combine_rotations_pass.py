"""This module implements the GroupSingleQuditGatePass."""
from __future__ import annotations

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import MeasurementPlaceholder
from bqskit.ir.gates import Reset
from bqskit.ir.gates.barrier import BarrierPlaceholder
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.gates.constant.sx import SqrtXGate
from bqskit.ir.gates.constant.sxdg import SqrtXdgGate
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.region import CircuitRegion


class CombineRotationsPass(BasePass):
    """
    The CombineRotations Pass.

    This pass combines consecutive rotation gates.
    """

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        # Go through each qudit individually
        for q in range(circuit.num_qudits):

            single_qubit_regions = []
            region_start = None

            for c in range(circuit.num_cycles):
                if circuit.is_point_idle((c, q)):
                    continue

                op = circuit[c, q]

                # Try to capture Rz Gates, or Rz + SqrtX/SqrtXdg Gates, but not any other gates
                condition_1 = len(op.params) > 0 

                # Otherwise, if it's already started, we can add SX and SxDg
                condition_2 = region_start is not None and (isinstance(op.gate, SqrtXGate) or isinstance(op.gate, SqrtXdgGate))

                if (
                    op.num_qudits == 1
                    and (condition_1 or condition_2)
                ):
                    if region_start is None:
                        region_start = c
                else:
                    if region_start is not None:
                        region = CircuitRegion({q: (region_start, c - 1)})
                        single_qubit_regions.append(region)
                        region_start = None

            if region_start is not None:
                region = CircuitRegion(
                    {q: (region_start, circuit.num_cycles - 1)},
                )
                single_qubit_regions.append(region)
                region_start = None

            for region in reversed(single_qubit_regions):
                circuit.fold(region)

        # Now replace all CircuitGates with the corresponding U3 gate
        for cycle, op in circuit.operations_with_cycles():
            if isinstance(op.gate, CircuitGate):
                new_u3_params = U3Gate().calc_params(op.get_unitary())
                circuit.replace_gate(
                    (cycle, op.location[0]),
                    U3Gate(),
                    op.location,
                    new_u3_params
                )

        
