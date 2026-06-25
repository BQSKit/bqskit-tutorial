"""This module implements the MeasurementPlaceholder class."""
from __future__ import annotations

from bqskit.ir.gate import Gate
from bqskit.ir.location import CircuitLocation
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class PPMPlaceholder(Gate):
    """Pseudogate to hold measurement information."""

    def __init__(
        self,
        bases: list[str],
    ) -> None:
        """
        Construct a Pauli Product Measurement placeholder gate.

        Args:
            bases (list[str]): The Pauli bases for each qubit in the measurement.
        """
        self._name = 'PPM'
        self._qasm_name = 'ppm'
        self._num_qudits = len(bases)
        self._radixes = tuple([2] * self._num_qudits)
        self._num_params = 2 # (rotation angle, sign)
        self.bases = bases

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        raise RuntimeError(
            'Cannot compute unitary for a measurement placeholder.'
        )
    
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PPMPlaceholder)
            and other.bases == self.bases
        )

    def __hash__(self) -> int:
        return hash(tuple(self.bases))

    def __repr__(self) -> str:
        return f"{self.name} {self.bases}"


