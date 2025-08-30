"""This module implements the Frontier class."""
from __future__ import annotations

import heapq
import itertools
from typing import Any
from typing import NamedTuple

from bqskit.ir.circuit import Circuit
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
import numpy as np


class FrontierElement(NamedTuple):
    """The Frontier contains FrontierElements."""
    cost: float
    element_id: int
    circuit: Circuit
    extra_data: Any


class Frontier:
    """The Frontier class."""

    def __init__(
            self,
            targets: list[np.ndarray],
            branch_circuits: list[Circuit],
            ancillas: list,
    ) -> None:
        """
        Construct an empty frontier.

        Args:
            target (UnitaryMatrix | StateVector | StateSystem): The target to
                pass to the heuristic_function.

            heuristic_function (HeuristicFunction): The heuristic used
                to sort the Frontier.
        """
        # for target in targets:
        #     if not isinstance(target, (UnitaryMatrix, StateVector, StateSystem, np.ndarray)):
        #         raise TypeError(
        #             'Expected unitary or state, got %s.' % type(target),
        #         )
        self.targets = targets
        self.branch_circuits = branch_circuits
        self._frontier: list[FrontierElement] = []
        self._counter = itertools.count()
        self.ancillas = ancillas

    def add(self, circuit: Circuit, best_params: list, extra_data: Any = None) -> None:
        """Add `circuit` into the frontier."""
        heuristic_value = self.heuristic_value(circuit, best_params)
        count = next(self._counter)
        elem = FrontierElement(heuristic_value, count, circuit, extra_data)
        heapq.heappush(self._frontier, elem)

    def add_leap(self, circuit: Circuit, best_params: list, extra_data: Any = None) -> None:
        """Add `circuit` into the frontier."""
        heuristic_value = self.heuristic_value_leap(circuit, best_params)
        count = next(self._counter)
        elem = FrontierElement(heuristic_value, count, circuit, extra_data)
        heapq.heappush(self._frontier, elem)

    def pop(self) -> tuple[Circuit, Any]:
        """Pop the top circuit."""
        elem = heapq.heappop(self._frontier)
        return elem.circuit, elem.extra_data

    def empty(self) -> bool:
        """Return true if the frontier is empty."""
        return len(self._frontier) == 0

    def clear(self) -> None:
        """Remove all elements from the frontier."""
        self._frontier.clear()

    def heuristic_value(self, circuit: Circuit, best_params: list, heuristic_factor: float = 10.0,
                        cost_factor: float = 1.0, ):
        # cost = 0.0
        cost = circuit.multi_qudit_depth
        # for gate in circuit.gate_set:
        #     if gate.num_qudits == 1:
        #         continue
        #     cost += float(circuit.count(gate))
        # for gate in self.branch_circ_0.gate_set:
        #     if gate.num_qudits == 1:
        #         continue
        #     cost += float(circuit.count(gate))
        # for gate in self.branch_circ_1.gate_set:
        #     if gate.num_qudits == 1:
        #         continue
        #     cost += float(circuit.count(gate))
        # heuristic = cost_function_modular(best_params, circuit, self.branch_circuits, self.targets, self.ancillas)
        # return heuristic_factor * heuristic + cost_factor * cost
        return cost

    def heuristic_value_leap(self, circuit: Circuit, best_params: list, heuristic_factor: float = 10.0,
                             cost_factor: float = 1.0, ):
        cost = 0.0
        for gate in circuit.gate_set:
            if gate.num_qudits == 1:
                continue
            cost += float(circuit.count(gate))
        from bqskit_0.leap_customize import cost_function
        heuristic = cost_function(best_params, circuit, self.targets, self.ancillas)
        return heuristic_factor * heuristic + cost_factor * cost