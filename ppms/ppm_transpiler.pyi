from typing import List
from .ppm import PPMPlaceholder

class TransItem:
    is_pauli: bool
    is_clifford: bool
    label: str  # "T", "M", "S", "CX", etc.
    sign: str   # "+" or "-"
    ops: list[str]  # per-qubit Pauli ops

    def __repr__(self) -> str: ...

def transpile_circuit(qasm_str: str, max_width: int) -> list[list[TransItem]]:
    """
    Parse a QASM string and run the PPM transpiler.

    Args:
        qasm_str: OpenQASM source as a string.
        max_width: Maximum resource width constraint.

    Returns:
        Layers of TransItems representing the transpiled circuit.
    """
    ...


