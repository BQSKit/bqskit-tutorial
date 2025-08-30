from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.stats import linregress
from bqskit.ir.circuit import Circuit
from utils_dc.frontier_multi import Frontier
from bqskit.passes.search.generator import LayerGenerator
from bqskit.passes.search.heuristic import HeuristicFunction
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_real_number
from utils_dc.simple_change import SimpleLayerGenerator
from utils_dc.dc_instantiation_unitary import cost_function
from utils_dc.dc_instantiation_unitary import unitary_grad_function
from scipy.optimize import minimize
from multiprocessing import Pool

_logger = logging.getLogger(__name__)


def optimize_params(start):
    # Unpack all necessary arguments if passed as a single tuple
    num_params, initial_layer, branch_circuits, target, ancillas = start
    np.random.seed(None)
    initial_params = np.random.uniform(low=-np.pi, high=np.pi, size=num_params)
    result = minimize(cost_function, initial_params,
                      args=(initial_layer, branch_circuits, target, ancillas),
                      method='BFGS', jac=unitary_grad_function)
    cost = cost_function(result.x, initial_layer, branch_circuits, target, ancillas)
    return cost, result.x


def optimize_successor(args):
    successor, branch_circuits, target, multi_starts, ancillas = args
    best_inst_cost = np.inf
    best_params = None
    successor_num_params = successor.num_params
    for branch_circ in branch_circuits:
        successor_num_params += branch_circ.num_params

    for _ in range(multi_starts):
        np.random.seed(None)
        initial_params = np.random.uniform(low=-np.pi, high=np.pi, size=successor_num_params)
        result = minimize(cost_function, initial_params,
                          args=(successor, branch_circuits, target, ancillas),
                          method='BFGS', jac=unitary_grad_function)
        cost = cost_function(result.x, successor, branch_circuits, target, ancillas)
        if cost < best_inst_cost:
            best_inst_cost = cost
            best_params = result.x

    return best_params


def set_params_circuits(qc, branch_circuits, best_params):
    params1 = best_params[:qc.num_params]
    qc.set_params(params1)

    # print(qc.to('qasm'))
    # print('------')
    idx = qc.num_params
    for branch_circuit in branch_circuits:
        params = best_params[idx : idx+branch_circuit.num_params]
        idx += branch_circuit.num_params
        branch_circuit.set_params(params)

        

class UnitarySynthesisPass():

    def __init__(
            self,
            branch_circuits: list,
            layer_generator: LayerGenerator | None = SimpleLayerGenerator(),
            success_threshold: float = 1e-8,
            max_layer: int | None = None,
            store_partial_solutions: bool = False,
            partials_per_depth: int = 25,
            min_prefix_size: int = 3,
            instantiate_options: dict[str, Any] = {},
    ) -> None:
        """
        Construct a search-based synthesis pass.

        Args:
            heuristic_function (HeuristicFunction): The heuristic to guide
                search.

            layer_generator (LayerGenerator | None): The successor function
                to guide node expansion. If left as none, then a default
                will be selected before synthesis based on the target
                model's gate set. (Default: None)

            success_threshold (float): The distance threshold that
                determines successful termintation. Measured in cost
                described by the cost function. (Default: 1e-8)

            cost (CostFunction | None): The cost function that determines
                distance during synthesis. The goal of this synthesis pass
                is to implement circuits for the given unitaries that have
                a cost less than the `success_threshold`.
                (Default: HSDistance())

            max_layer (int): The maximum number of layers to append without
                success before termination. If left as None it will default
                to unlimited. (Default: None)

            store_partial_solutions (bool): Whether to store partial solutions
                at different depths inside of the data dict. (Default: False)

            partials_per_depth (int): The maximum number of partials
                to store per search depth. No effect if
                `store_partial_solutions` is False. (Default: 25)

            min_prefix_size (int): The minimum number of layers needed
                to prefix the circuit.

            instantiate_options (dict[str: Any]): Options passed directly
                to circuit.instantiate when instantiating circuit
                templates. (Default: {})

        Raises:
            ValueError: If `max_depth` or `min_prefix_size` is nonpositive.
        """
        if layer_generator is not None:
            if not isinstance(layer_generator, LayerGenerator):
                raise TypeError(
                    f'Expected LayerGenerator, got {type(layer_generator)}.',
                )

        if not is_real_number(success_threshold):
            raise TypeError(
                'Expected real number for success_threshold'
                ', got %s' % type(success_threshold),
            )

        if max_layer is not None and not is_integer(max_layer):
            raise TypeError(
                'Expected max_layer to be an integer, got %s' % type(max_layer),
            )

        if max_layer is not None and max_layer <= 0:
            raise ValueError(
                'Expected max_layer to be positive, got %d.' % int(max_layer),
            )

        if min_prefix_size is not None and not is_integer(min_prefix_size):
            raise TypeError(
                'Expected min_prefix_size to be an integer, got %s'
                % type(min_prefix_size),
            )

        if min_prefix_size is not None and min_prefix_size <= 0:
            raise ValueError(
                'Expected min_prefix_size to be positive, got %d.'
                % int(min_prefix_size),
            )

        if not isinstance(instantiate_options, dict):
            raise TypeError(
                'Expected dictionary for instantiate_options, got %s.'
                % type(instantiate_options),
            )

        self.layer_gen = layer_generator
        self.success_threshold = success_threshold
        self.max_layer = max_layer
        self.min_prefix_size = min_prefix_size
        self.store_partial_solutions = store_partial_solutions
        self.partials_per_depth = partials_per_depth
        self.branch_circuits = branch_circuits

    def synthesize(
            self,
            target: Circuit,
            ancillas: list,
            coupling_graph: list = None,
            initial_layer: Circuit = None,
    ) -> Circuit:
        """Synthesize `utry`, see :class:`SynthesisPass` for more."""
        # Initialize run-dependent options
        # target_state = data['target_state']
        # target_state_0_length = 2 * len(target_state)
        # target_state_1_length = 2 * len(target_state)
        # target_state_0 = np.zeros(target_state_0_length)
        # target_state_1 = np.zeros(target_state_1_length)
        # # The original state vector corresponds to the first half of the new state vector
        # # since appending a qubit in state |0> doesn't change the coefficients of the original state
        # target_state_0[:len(target_state)] = target_state
        # target_state_1[len(target_state):] = target_state

        # Seed the PRNG
        # Get layer generator for search
        layer_gen = self.layer_gen
        num_qudits = int(np.log2(int(target.shape[0]))) + len(ancillas)
        if coupling_graph is None:
            coupling_graph = [(i, j) for i in range(num_qudits) for j in range(i + 1, target_states[0].num_qudits)]
        # Begin the search with an initial layer
        frontier = Frontier(target, self.branch_circuits, ancillas)

        if initial_layer is None:
            initial_layer = layer_gen.gen_initial_layer(num_qudits)
        num_params = initial_layer.num_params
        for circuit in self.branch_circuits:
            num_params += circuit.num_params

        multi_starts = 15

        starts = [(num_params, initial_layer, self.branch_circuits, target, ancillas) for _ in range(multi_starts)]

        with Pool() as pool:
            results = pool.map(optimize_params, starts)

        best_inst_cost = np.inf
        best_params = None
        for cost, params in results:
            if cost < best_inst_cost:
                best_inst_cost = cost
                best_params = params
        # print('initial layer cost:', best_inst_cost)
        initial_layer.set_params(best_params[:initial_layer.num_params])
        frontier.add(initial_layer, best_params, 0)

        # Track best circuit, initially the initial layer
        best_dist = best_inst_cost
        best_circ = initial_layer
        best_layer = 0
        best_dists = [best_dist]
        best_layers = [0]
        last_prefix_layer = 0

        # Track partial solutions
        psols: dict[int, list[tuple[Circuit, float]]] = {}

        _logger.debug(f'Search started, initial layer has cost: {best_dist}.')

        # Evalute initial layer
        if best_dist < self.success_threshold:
            _logger.debug('Successful synthesis.')
            return initial_layer, best_params

        # Main loop
        while not frontier.empty():
            top_circuit, layer = frontier.pop()
            # Generate successors
            successors = layer_gen.gen_successors(top_circuit, coupling_graph)
            # make all the successor in parallel instead of multistarts

            args_list = [(successor, self.branch_circuits, target, multi_starts, ancillas) for successor in successors]

            # Use Pool to execute optimizations in parallel
            with Pool() as pool:
                successor_best_params = pool.map(optimize_successor, args_list)

            # Evaluate successors
            for successor, best_params in zip(successors, successor_best_params):
                dist = cost_function(best_params, successor, self.branch_circuits, target, ancillas)
                # print('dist:', dist)
                # for op in successor:
                #     print(op)
                # print('-----')
                if dist < self.success_threshold:
                    _logger.debug('Successful synthesis.')
                    two_qubit_gate_locs = set()
                    for op in successor:
                        if len(op.location) > 1:
                            for l in op.location:
                                two_qubit_gate_locs.add(l)

                    if all(q in two_qubit_gate_locs for q in ancillas):
                        ancilla_active = True
                    else:
                        ancilla_active = False
                    if ancilla_active is True:
                        print(f'final dist: {dist}')
                        return successor, best_params
                    else:
                        frontier.add(successor, best_params,
                                     layer + 1)
                    # if self.store_partial_solutions:
                    #     data['psols'] = psols

                    # return successor, best_params

                if self.check_new_best(layer + 1, dist, best_layer, best_dist):
                    plural = '' if layer == 0 else 's'
                    _logger.debug(
                        f'New best circuit found with {layer + 1} layer{plural}'
                        f' and cost: {dist:.12e}.',
                    )
                    best_dist = dist
                    best_circ = successor
                    best_layer = layer + 1

                    if self.check_leap_condition(
                            layer + 1,
                            best_dist,
                            best_layers,
                            best_dists,
                            last_prefix_layer,
                    ):
                        _logger.debug(f'Prefix formed at {layer + 1} layers.')
                        last_prefix_layer = layer + 1
                        frontier.clear()
                        if self.max_layer is None or layer + 1 < self.max_layer:
                            frontier.add(successor, best_params, layer + 1)

                if self.store_partial_solutions:
                    if layer not in psols:
                        psols[layer] = []

                    psols[layer].append((successor.copy(), dist))

                    if len(psols[layer]) > self.partials_per_depth:
                        psols[layer].sort(key=lambda x: x[1])
                        del psols[layer][-1]

                if self.max_layer is None or layer + 1 < self.max_layer:
                    frontier.add(successor, best_params,
                                 layer + 1)

        _logger.warning('Frontier emptied.')
        _logger.warning(
            'Returning best known circuit with %d layer%s and cost: %e.'
            % (best_layer, '' if best_layer == 1 else 's', best_dist),
        )
        # if self.store_partial_solutions:
        #     data['psols'] = psols

        return best_circ, best_params


    def check_new_best(
            self,
            layer: int,
            dist: float,
            best_layer: int,
            best_dist: float,
    ) -> bool:
        """
        Check if the new layer depth and dist are a new best node.

        Args:
            layer (int): The current layer in search.

            dist (float): The current distance in search.

            best_layer (int): The current best layer in the search tree.

            best_dist (float): The current best distance in search.
        """
        better_layer = (
                dist < best_dist
                and (
                        best_dist >= self.success_threshold
                        or layer <= best_layer
                )
        )
        better_dist_and_layer = (
                dist < self.success_threshold and layer < best_layer
        )
        return better_layer or better_dist_and_layer

    def check_leap_condition(
            self,
            new_layer: int,
            best_dist: float,
            best_layers: list[int],
            best_dists: list[float],
            last_prefix_layer: int,
    ) -> bool:
        """
        Return true if the leap condition is satisfied.

        Args:
            new_layer (int): The current layer in search.

            best_dist (float): The current best distance in search.

            best_layers (list[int]): The list of layers associated
                with recorded best distances.

            best_dists (list[float]): The list of recorded best
                distances.

            last_prefix_layer (int): The last layer a prefix was formed.
        """

        with np.errstate(invalid='ignore', divide='ignore'):
            # Calculate predicted best value
            m, y_int, _, _, _ = linregress(best_layers, best_dists)

        predicted_best = m * (new_layer) + y_int

        # Track new values
        best_layers.append(new_layer)
        best_dists.append(best_dist)

        if np.isnan(predicted_best):
            return False

        # Compute difference between actual value
        delta = predicted_best - best_dist

        _logger.debug(
            'Predicted best value %f for new best best with delta %f.'
            % (predicted_best, delta),
        )

        layers_added = new_layer - last_prefix_layer
        return delta < 0 and layers_added >= self.min_prefix_size

