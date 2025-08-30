import itertools
import numpy as np
from bqskit.qis.state.state import StateVector
import itertools
from bqskit import Circuit
from bqskit.ir.gates import HGate
from bqskit.ir.gates.constant.cx import CNOTGate, CXGate
from bqskit.ir.gates.parameterized.u3 import U3Gate
from scipy.optimize import minimize
import copy

M0 = np.array([[1, 0], [0, 0]])
M1 = np.array([[0, 0], [0, 1]])
I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])

CX = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 1],
               [0, 0, 1, 0]]
               )


def generate_measurement_operators(ancillas, num_qubits):
    # Define the base operators
    M0 = np.array([[1, 0], [0, 0]])
    M1 = np.array([[0, 0], [0, 1]])
    I = np.array([[1, 0], [0, 1]])

    # Calculate the number of measurement outcomes for the ancillas
    num_outcomes = 2 ** len(ancillas)

    # Initialize the list to store all measurement operators
    measurement_operators = []

    # Iterate over all possible measurement outcomes
    for outcome in range(num_outcomes):
        # Generate the binary representation of the outcome
        binary_outcome = format(outcome, f'0{len(ancillas)}b')

        # Initialize the measurement operator for this outcome
        operator = None

        # Iterate over all qubits to construct the measurement operator
        for qubit in range(num_qubits):
            # If the qubit is an ancilla, decide between M0 and M1
            if qubit in ancillas:
                ancilla_index = ancillas.index(qubit)
                if binary_outcome[ancilla_index] == '0':
                    current_operator = M0
                else:
                    current_operator = M1
            else:
                # For non-ancilla qubits, use the identity operator
                current_operator = I

            # Tensor the current operator with the existing operator
            if operator is None:
                operator = current_operator
            else:
                operator = np.kron(operator, current_operator)

        # Add the constructed measurement operator to the list
        measurement_operators.append(operator)

    return measurement_operators


def generat_random_target_states(state_vector, new_qubit_position):
    # def extend_state_vector(state_vector, new_qubit_position):
    """
    Extend the state vector by adding a new qubit in the specified position for both |0> and |1> states.

    Parameters:
    state_vector (numpy array): The original state vector.
    new_qubit_position (int): The position of the new qubit (0-indexed).

    Returns:
    dict: Dictionary containing the extended state vectors for the new qubit in states 0 and 1.
    """
    num_qubits = int(np.log2(len(state_vector)))  # Number of qubits in the original state vector
    new_num_qubits = num_qubits + 1  # Number of qubits in the new state vector

    # Initialize new state vectors for the new qubit in state 0 and state 1
    new_state_vector_0 = np.zeros(2 ** new_num_qubits, dtype=state_vector.dtype)
    new_state_vector_1 = np.zeros(2 ** new_num_qubits, dtype=state_vector.dtype)

    for i in range(len(state_vector)):
        # Determine the new index for new qubit in state |0>
        binary_index_0 = list(np.binary_repr(i, width=num_qubits))
        binary_index_0.insert(new_qubit_position, '0')
        new_index_0 = int("".join(binary_index_0), 2)

        # Determine the new index for new qubit in state |1>
        binary_index_1 = list(np.binary_repr(i, width=num_qubits))
        binary_index_1.insert(new_qubit_position, '1')
        new_index_1 = int("".join(binary_index_1), 2)

        # Assign the original state vector element to the new state vectors
        new_state_vector_0[new_index_0] = state_vector[i]
        new_state_vector_1[new_index_1] = state_vector[i]

    return {'new_state_0': StateVector(new_state_vector_0),
            'new_state_1': StateVector(new_state_vector_1)}


def generate_ghz_target_states(n, ancillas):
    ancillas = list(ancillas)
    ancillas.sort()
    states = {}
    combinations = itertools.product('01', repeat=len(ancillas))
    # Join each tuple to form strings and create a list of these strings
    meas_results = [''.join(combo) for combo in combinations]
    for meas_res in meas_results:
        state0_list = ['0'] * n
        state1_list = ['1'] * n
        # Modify lists based on measurement results and ancilla positions
        for res, ancilla in zip(meas_res, ancillas):
            state0_list[ancilla] = res
            state1_list[ancilla] = res
        # Join lists to form final binary strings
        state0_string = ''.join(state0_list)
        state1_string = ''.join(state1_list)
        # print('state0 string:', state0_string)
        # print('state0 idx:', int(state0_string, 2))
        # print('state1 string:', state1_string)
        # print('state1 idx:', int(state1_string, 2))
        state = [0] * 2 ** n
        state[int(state0_string, 2)] = 1
        state[int(state1_string, 2)] = 1
        target_state = StateVector(1 / np.sqrt(2) * np.array(state))
        states[meas_res] = target_state
    return states


def generate_w_target_states(n, ancillas):
    states = {}
    ancillas.sort()
    combinations = itertools.product('01', repeat=len(ancillas))
    # Join each tuple to form strings and create a list of these strings
    meas_results = [''.join(combo) for combo in combinations]

    for meas_res in meas_results:
        state_list = []
        for i in range(n - 1):
            state = ['0'] * (n - 1)
            state[i] = '1'
            state_list.append(state)
        # Modify lists based on measurement results and ancilla positions
        update_state_list = []
        for state in state_list:
            for res, ancilla in zip(meas_res, ancillas):
                state.insert(ancilla, res)
            update_state_list.append(state)
        # Join lists to form final binary strings
        state_strings = []
        for state in update_state_list:
            state_string = ''.join(state)
            state_strings.append(state_string)
        # print('state0 string:', state0_string)
        # print('state0 idx:', int(state0_string, 2))
        # print('state1 string:', state1_string)
        # print('state1 idx:', int(state1_string, 2))
        state = [0] * 2 ** n
        for state_string in state_strings:
            state[int(state_string, 2)] = 1
        target_state = StateVector(1 / np.sqrt(n - 1) * np.array(state))
        states[meas_res] = target_state
    return states


def prepare_dicke_state(n, k):
    # Create a list to store the result bit strings
    result = []
    for combo in itertools.combinations(range(n), k):
        # Create a list of '0's of length 'n'
        bit_string = ['0'] * n

        # Place '1's in the positions specified by the current combination
        for pos in combo:
            bit_string[pos] = '1'

        # Append the list of characters directly to the result list
        result.append(bit_string)

    return result


def generate_dicke_target_states(n, k, ancillas):
    states = {}
    ancillas.sort()
    combinations = itertools.product('01', repeat=len(ancillas))
    # Join each tuple to form strings and create a list of these strings
    meas_results = [''.join(combo) for combo in combinations]
    state_list = prepare_dicke_state(n - 1, k)
    for meas_res in meas_results:
        # Modify lists based on measurement results and ancilla positions
        update_state_list = []
        state_list_copy = copy.deepcopy(state_list)
        for state in state_list_copy:
            for res, ancilla in zip(meas_res, ancillas):
                state.insert(ancilla, res)
            update_state_list.append(state)
        # Join lists to form final binary strings
        state_strings = []
        for state in update_state_list:
            state_string = ''.join(state)
            state_strings.append(state_string)
        # print('state0 string:', state0_string)
        # print('state0 idx:', int(state0_string, 2))
        # print('state1 string:', state1_string)
        # print('state1 idx:', int(state1_string, 2))
        state = [0] * 2 ** n
        for state_string in state_strings:
            state[int(state_string, 2)] = 1
        target_state = StateVector(1 / np.sqrt(len(state_list)) * np.array(state))
        states[meas_res] = target_state
    return states


def cost_function(params, qc0, qc1, qc2, T0, T1, ancilla):
    num_params_qc0 = qc0.num_params
    num_params_qc1 = qc1.num_params
    params_0 = params[:num_params_qc0]
    params_1 = params[num_params_qc0:num_params_qc0 + num_params_qc1]
    params_2 = params[num_params_qc0 + num_params_qc1:]
    qc0.set_params(params_0)
    qc1.set_params(params_1)
    qc2.set_params(params_2)
    num_qudits = qc0.num_qudits
    initial_state = [0] * 2 ** num_qudits
    initial_state[0] = 1
    initial_state = np.array(initial_state)
    M0s = I if ancilla != 0 else M0
    M1s = I if ancilla != 0 else M1
    for i in range(1, num_qudits):
        if i == ancilla:
            M0s = np.kron(M0s, M0)
        else:
            M0s = np.kron(M0s, I)
    for i in range(1, num_qudits):
        if i == ancilla:
            M1s = np.kron(M1s, M1)
        else:
            M1s = np.kron(M1s, I)
    U0 = qc0.get_unitary()
    U1 = qc1.get_unitary()
    U2 = qc2.get_unitary()
    state0 = U1 @ M0s @ U0 @ initial_state
    state1 = U2 @ M1s @ U0 @ initial_state
    norm0 = np.linalg.norm(state0)
    state0_normalized = state0 / norm0
    norm1 = np.linalg.norm(state1)
    state1_normalized = state1 / norm1
    d0 = np.vdot(T0, state0_normalized)
    d1 = np.vdot(T1, state1_normalized)
    infidelity = 2 - np.abs(d0) ** 2 - np.abs(d1) ** 2
    # print(infidelity)
    return infidelity


def cost_function_test(qc0, qc1, qc2, T0, T1, ancilla):
    num_qudits = qc0.num_qudits
    initial_state = [0] * 2 ** num_qudits
    initial_state[0] = 1
    initial_state = np.array(initial_state)
    M0s = I if ancilla != 0 else M0
    M1s = I if ancilla != 0 else M1
    for i in range(1, num_qudits):
        if i == ancilla:
            M0s = np.kron(M0s, M0)
        else:
            M0s = np.kron(M0s, I)
    for i in range(1, num_qudits):
        if i == ancilla:
            M1s = np.kron(M1s, M1)
        else:
            M1s = np.kron(M1s, I)
    U0 = qc0.get_unitary()
    U1 = qc1.get_unitary()
    U2 = qc2.get_unitary()
    state0 = U1 @ M0s @ U0 @ initial_state
    state1 = U2 @ M1s @ U0 @ initial_state
    norm0 = np.linalg.norm(state0)
    print(norm0)
    state0_normalized = state0 / norm0
    norm1 = np.linalg.norm(state1)
    print(norm1)
    state1_normalized = state1 / norm1
    d0 = np.vdot(T0, state0_normalized)
    d1 = np.vdot(T1, state1_normalized)
    infidelity1 = 2 - np.abs(d0) ** 2 - np.abs(d1) ** 2

    d_t0 = np.vdot(T0, state0)
    d_t1 = np.vdot(T1, state1)

    infidelity2 = 1 - np.abs(d_t0) ** 2 - np.abs(d_t1) ** 2
    # print(infidelity)
    return infidelity1, infidelity2


def cost_function_modular(params, qc, branch_circuits, Ts, ancillas):
    utrys = []
    qc.set_params(params[:qc.num_params])
    U0 = qc.get_unitary()
    params_branch_circuits = params[qc.num_params:]
    i = 0
    for circ_idx in range(0, len(branch_circuits), 2):
        branch_circ0 = branch_circuits[circ_idx]
        branch_circ1 = branch_circuits[circ_idx + 1]
        branch_circ0.set_params(params_branch_circuits[i:i + branch_circ0.num_params])
        branch_circ1.set_params(params_branch_circuits[i + branch_circ0.num_params:
                                                       i + branch_circ0.num_params + branch_circ1.num_params])
        i += branch_circ0.num_params + branch_circ1.num_params
        utrys.append([branch_circ0.get_unitary(), branch_circ1.get_unitary()])
    # IXII = np.kron(I, np.kron(X, np.kron(I, I)))
    # IIXI = np.kron(I, np.kron(I, np.kron(X, I)))
    # print(np.allclose(utrys[0][0], IXII, rtol=1e-05, atol=1e-08))
    # print(np.allclose(utrys[0][1], IIXI, rtol=1e-05, atol=1e-08))
    # print(np.allclose(utrys[1][0], IXII, rtol=1e-05, atol=1e-08))
    # print(np.allclose(utrys[1][1], IIXI, rtol=1e-05, atol=1e-08))
    num_qudits = qc.num_qudits
    products = []
    if len(utrys) > 1:
        combinations = itertools.product(*utrys)
        for combo in combinations:
            product = np.eye(2 ** num_qudits, dtype=np.complex128)
            for item in combo:
                product @= item
            products.append(product)
    else:
        for ele in utrys[0]:
            products.append(ele)
    initial_state = [0] * 2 ** num_qudits
    initial_state[0] = 1
    initial_state = np.array(initial_state)
    opts = generate_measurement_operators(ancillas, num_qudits)
    # opts_check = [
    #     np.kron(M0, np.kron(I, np.kron(I, M0))),
    #     np.kron(M0, np.kron(I, np.kron(I, M1))),
    #     np.kron(M1, np.kron(I, np.kron(I, M0))),
    #     np.kron(M1, np.kron(I, np.kron(I, M1))),
    # ]
    # for opt, opt_check in zip(opts, opts_check):
    #     print(np.allclose(opt, opt_check, rtol=1e-05, atol=1e-08))
    states = []
    for product, opt in zip(products, opts):
        state = product @ opt @ U0 @ initial_state
        if not np.all(state == 0):
            norm = np.linalg.norm(state)
            state_normalized = state / norm
            states.append(state_normalized)
        else:
            states.append(state)
    infidelity = 0
    for state, T in zip(states, Ts):
        d = np.vdot(T, state)
        infidelity += 1 - np.abs(d) ** 2
    return infidelity


def state_infidelity_grad_modular(params, qc, branch_circuits, Ts, ancillas):
    utrys = []
    grads = []
    qc.set_params(params[:qc.num_params])
    U0 = qc.get_unitary()
    grad0 = qc.get_grad()
    params_branch_circuits = params[qc.num_params:]
    i = 0
    idx_utry = {}
    idx_Us = []
    for circ_idx in range(0, len(branch_circuits), 2):
        branch_circ0 = branch_circuits[circ_idx]
        branch_circ1 = branch_circuits[circ_idx + 1]
        branch_circ0.set_params(params_branch_circuits[i:i + branch_circ0.num_params])
        branch_circ1.set_params(params_branch_circuits[i + branch_circ0.num_params:
                                                       i + branch_circ0.num_params + branch_circ1.num_params])
        i += branch_circ0.num_params + branch_circ1.num_params
        utrys.append([branch_circ0.get_unitary(), branch_circ1.get_unitary()])
        idx_utry[circ_idx] = branch_circ0.get_unitary()
        idx_utry[circ_idx + 1] = branch_circ1.get_unitary()
        idx_Us.append([circ_idx, circ_idx + 1])
        grads.append(branch_circ0.get_grad())
        grads.append(branch_circ1.get_grad())

    num_qudits = qc.num_qudits
    products = []
    idx_Us_combs = []
    if len(utrys) > 1:
        combinations = itertools.product(*utrys)
        idx_Us_combintations = itertools.product(*idx_Us)
        for combo in combinations:
            product = np.eye(2 ** num_qudits, dtype=np.complex128)
            for item in combo:
                product @= item
            products.append(product)
        for pair in idx_Us_combintations:
            idx_Us_combs.append(pair)
    else:
        for ele in utrys[0]:
            products.append(ele)
        for ele in idx_Us[0]:
            idx_Us_combs.append([ele])
    initial_state = [0] * 2 ** num_qudits
    initial_state[0] = 1
    initial_state = np.array(initial_state)
    opts = generate_measurement_operators(ancillas, num_qudits)
    states = []
    for product, opt in zip(products, opts):
        state = product @ opt @ U0 @ initial_state
        if not np.all(state == 0):
            norm = np.linalg.norm(state)
            state_normalized = state / norm
            states.append(state_normalized)
        else:
            states.append(state)
    ds = []
    for state, T in zip(states, Ts):
        d = np.vdot(T, state)
        ds.append(d)
    d_infidelity = []
    for dv_matrix0 in grad0:
        g_real = 0
        g_imag = 0
        for idx, product in enumerate(products):
            g_real += -2 * ds[idx].real * np.vdot(Ts[idx], product @ opts[idx] @ dv_matrix0 @ initial_state).real
            g_imag += -2 * ds[idx].imag * np.vdot(Ts[idx], product @ opts[idx] @ dv_matrix0 @ initial_state).imag
        d_infidelity.append(g_real + g_imag)

    for i in range(len(Ts)):
        for dv_matrix in grads[i]:
            g_real = 0
            g_imag = 0
            for pair_idx, pair in enumerate(idx_Us_combs):
                if i in pair:
                    utry_left = np.eye(2 ** num_qudits, dtype=np.complex128)
                    for p in pair:
                        if p == i:
                            utry_left @= dv_matrix
                        else:
                            utry_left @= idx_utry[p]
                    g_real += -2 * ds[pair_idx].real * np.vdot(Ts[pair_idx],
                                                               utry_left @ opts[pair_idx] @ U0 @ initial_state).real
                    g_imag += -2 * ds[pair_idx].imag * np.vdot(Ts[pair_idx],
                                                               utry_left @ opts[pair_idx] @ U0 @ initial_state).imag
            d_infidelity.append(g_real + g_imag)

    # for dv0_matrix in grad0:
    #     j_effect_real = 0
    #     j_effect_imag = 0
    #     idx = 0
    #     for utry, opt in zip(utrys, opts):
    #         j_effect_real += -2 * ds[idx].real * np.vdot(Ts[idx], utry @ opt @ dv0_matrix @ initial_state).real
    #         j_effect_imag += -2 * ds[idx].imag * np.vdot(Ts[idx], utry @ opt @ dv0_matrix @ initial_state).imag
    #         idx += 1
    #     d_infidelity.append(j_effect_real + j_effect_imag)

    # for idx, grad in enumerate(grads):
    #     for dv_matrix in grad:
    #         st = dv_matrix @ opts[idx] @ U0 @ initial_state
    #         j_effect_real = -2 * ds[idx].real * np.vdot(Ts[idx], st).real
    #         j_effect_imag = -2 * ds[idx].real * np.vdot(Ts[idx], st).imag
    #         d_infidelity.append(j_effect_real + j_effect_imag)
    return d_infidelity


def two_cost_function(params, qc0, qc1, qc2, T0, T1, ancilla):
    num_params_qc0 = qc0.num_params
    num_params_qc1 = qc1.num_params
    params_0 = params[:num_params_qc0]
    params_1 = params[num_params_qc0:num_params_qc0 + num_params_qc1]
    params_2 = params[num_params_qc0 + num_params_qc1:]
    qc0.set_params(params_0)
    qc1.set_params(params_1)
    qc2.set_params(params_2)
    num_qudits = qc0.num_qudits
    initial_state = [0] * 2 ** num_qudits
    initial_state[0] = 1
    initial_state = np.array(initial_state)
    M0s = I if ancilla != 0 else M0
    M1s = I if ancilla != 0 else M1
    for i in range(1, num_qudits):
        if i == ancilla:
            M0s = np.kron(M0s, M0)
        else:
            M0s = np.kron(M0s, I)
    for i in range(1, num_qudits):
        if i == ancilla:
            M1s = np.kron(M1s, M1)
        else:
            M1s = np.kron(M1s, I)
    U0 = qc0.get_unitary()
    U1 = qc1.get_unitary()
    U2 = qc2.get_unitary()
    state0 = U1 @ M0s @ U0 @ initial_state
    state1 = U2 @ M1s @ U0 @ initial_state
    norm0 = np.linalg.norm(state0)
    state0_normalized = state0 / norm0
    norm1 = np.linalg.norm(state1)
    state1_normalized = state1 / norm1
    d0 = np.vdot(T0, state0_normalized)
    d1 = np.vdot(T1, state1_normalized)
    infid1 = 1 - np.abs(d0) ** 2
    infid2 = 1 - np.abs(d1) ** 2
    infidelity = infid1 + infid2
    # print(infidelity)
    return infid1, infid2, infidelity


def state_infidelity_grad(params, qc0, qc1, qc2, T0, T1, ancilla):
    num_params_qc0 = qc0.num_params
    num_params_qc1 = qc1.num_params
    params_0 = params[:num_params_qc0]
    params_1 = params[num_params_qc0:num_params_qc0 + num_params_qc1]
    params_2 = params[num_params_qc0 + num_params_qc1:]
    qc0.set_params(params_0)
    qc1.set_params(params_1)
    qc2.set_params(params_2)
    U0 = qc0.get_unitary()
    U1 = qc1.get_unitary()
    U2 = qc2.get_unitary()
    j0 = qc0.get_grad()
    j1 = qc1.get_grad()
    j2 = qc2.get_grad()
    num_qudits = qc0.num_qudits
    M0s = I if ancilla != 0 else M0
    M1s = I if ancilla != 0 else M1
    for i in range(1, num_qudits):
        if i == ancilla:
            M0s = np.kron(M0s, M0)
        else:
            M0s = np.kron(M0s, I)
    for i in range(1, num_qudits):
        if i == ancilla:
            M1s = np.kron(M1s, M1)
        else:
            M1s = np.kron(M1s, I)
    initial_state = [0] * 2 ** num_qudits
    initial_state[0] = 1
    initial_state = np.array(initial_state)
    state0 = U1 @ M0s @ U0 @ initial_state
    state1 = U2 @ M1s @ U0 @ initial_state
    norm0 = np.linalg.norm(state0)
    state0_normalized = state0 / norm0
    norm1 = np.linalg.norm(state1)
    state1_normalized = state1 / norm1
    d0 = np.vdot(T0, state0_normalized)
    d1 = np.vdot(T1, state1_normalized)
    d_infidelity = []
    for dv0_matrix in j0:
        j0_effect_real_d0 = -2 * d0.real * np.vdot(T0, U1 @ M0s @ dv0_matrix @ initial_state).real
        j0_effect_imag_d0 = -2 * d0.imag * np.vdot(T0, U1 @ M0s @ dv0_matrix @ initial_state).imag
        j0_effect_real_d1 = -2 * d1.real * np.vdot(T1, U2 @ M1s @ dv0_matrix @ initial_state).real
        j0_effect_imag_d1 = -2 * d1.imag * np.vdot(T1, U2 @ M1s @ dv0_matrix @ initial_state).imag
        d_infidelity.append(j0_effect_real_d0 + j0_effect_imag_d0 + j0_effect_real_d1 + j0_effect_imag_d1)
    for dv1_matrix in j1:
        j1_effect_real = -2 * d0.real * np.vdot(T0, dv1_matrix @ M0s @ U0 @ initial_state).real
        j1_effect_imag = -2 * d0.imag * np.vdot(T0, dv1_matrix @ M0s @ U0 @ initial_state).imag
        d_infidelity.append(j1_effect_real + j1_effect_imag)
    for dv2_matrix in j2:
        j2_effect_real = -2 * d1.real * np.vdot(T1, dv2_matrix @ M1s @ U0 @ initial_state).real
        j2_effect_imag = -2 * d1.imag * np.vdot(T1, dv2_matrix @ M1s @ U0 @ initial_state).imag
        d_infidelity.append(j2_effect_real + j2_effect_imag)
    return d_infidelity


def test_unitary():
    qc = Circuit(4)
    qc.append_gate(HGate(), 0)
    qc.append_gate(HGate(), 3)
    qc.append_gate(CXGate(), (0, 1))
    qc.append_gate(CXGate(), (1, 2))
    qc.append_gate(CXGate(), (2, 3))

    U0 = qc.get_unitary()
    test_U0 = (np.kron(np.kron(I, I), CX)) @ (np.kron(np.kron(I, CX), I)) @ (np.kron(np.kron(CX, I), I)) @ np.kron(
        np.kron(np.kron(H, I), I), H)
    print(np.allclose(test_U0, U0, rtol=1e-05, atol=1e-08))

    qc1 = Circuit(4)
    qc1.append_gate(U3Gate(), 0)
    qc1.append_gate(U3Gate(), 3)
    qc1.append_gate(CXGate(), (0, 1))
    qc1.append_gate(CXGate(), (1, 2))
    qc1.append_gate(CXGate(), (2, 3))
    correct_initial_params = [np.pi / 2, 0, np.pi,
                              np.pi / 2, 0, np.pi, ]
    qc1.set_params(correct_initial_params)
    U1 = qc1.get_unitary()

    print(np.allclose(U1, U0, rtol=1e-05, atol=1e-08))


def state_instantiation(base_circuit, branch_circuits, ancillas, target_states):
    num_params = base_circuit.num_params
    for branch_circ in branch_circuits:
        num_params += branch_circ.num_params
    multi_starts = 10
    best_cost = np.inf
    for _ in range(multi_starts):
        initial_params = np.random.uniform(low=-np.pi, high=np.pi, size=num_params)
        result = minimize(cost_function_modular, initial_params,
                          args=(base_circuit, branch_circuits, target_states, ancillas),
                          method='BFGS', jac=state_infidelity_grad_modular)
        cost = cost_function_modular(result.x, base_circuit, branch_circuits, target_states, ancillas)
    if cost < best_cost:
        best_cost = cost
    return cost
    
def test_modular():
    qc = Circuit(4)
    qc.append_gate(U3Gate(), 0)
    qc.append_gate(U3Gate(), 3)
    qc.append_gate(CXGate(), (0, 1))
    qc.append_gate(CXGate(), (1, 2))
    qc.append_gate(CXGate(), (2, 3))
    ancillas = [0, 3]
    branch_circuits = []
    for i in range(4):
        branch_qc = Circuit(4)
        for i in range(1, 3):
            branch_qc.append_gate(U3Gate(), i)
        branch_circuits.append(branch_qc)

    target_states = []
    state0 = [0] * 2 ** 4
    state0[0] = 1
    state1 = [0] * 2 ** 4
    state1[7] = 1
    state2 = [0] * 2 ** 4
    state2[8] = 1
    state3 = [0] * 2 ** 4
    state3[15] = 1
    target_states.append(StateVector(state0))
    target_states.append(StateVector(state1))
    target_states.append(StateVector(state2))
    target_states.append(StateVector(state3))

    correct_initial_params = [np.pi / 2, 0, np.pi,
                              np.pi / 2, 0, np.pi,
                              np.pi, 0, np.pi,
                              0, 0, 0,
                              0, 0, 0,
                              np.pi, 0, np.pi,
                              np.pi, 0, np.pi,
                              0, 0, 0,
                              0, 0, 0,
                              np.pi, 0, np.pi]

    num_params = qc.num_params
    for branch_circ in branch_circuits:
        num_params += branch_circ.num_params
    multi_starts = 5
    best_cost = np.Inf
    for _ in range(multi_starts):
        initial_params = np.random.uniform(low=-np.pi, high=np.pi, size=num_params)
        result = minimize(cost_function_modular, initial_params,
                          args=(qc, branch_circuits, target_states, ancillas),
                          method='BFGS', jac=state_infidelity_grad_modular)
        cost = cost_function_modular(result.x, qc, branch_circuits, target_states, ancillas)
        if cost < best_cost:
            best_cost = cost
    print(best_cost)


def test_modular2():
    num_qudits = 3

    qc0 = Circuit(num_qudits)
    qc0.append_gate(U3Gate(), 0)
    qc0.append_gate(CXGate(), (0, 1))
    qc0.append_gate(CXGate(), (1, 2))

    qc1 = Circuit(num_qudits)
    qc1.append_gate(U3Gate(), 0)
    qc1.append_gate(U3Gate(), 1)

    qc2 = Circuit(num_qudits)
    qc2.append_gate(U3Gate(), 0)
    qc2.append_gate(U3Gate(), 1)

    T0 = np.array([0, 0, 1, 0, 0, 0, 0, 0])
    T1 = np.array([0, 0, 0, 1, 0, 0, 0, 0])
    branch_circuits = [qc1, qc2]
    ancillas = [2]
    target_states = [T0, T1]
    num_params = qc0.num_params
    for branch_circ in branch_circuits:
        num_params += branch_circ.num_params
    multi_starts = 5
    best_cost = np.Inf
    for _ in range(multi_starts):
        initial_params = np.random.uniform(low=-np.pi, high=np.pi, size=num_params)
        result = minimize(cost_function_modular, initial_params,
                          args=(qc0, branch_circuits, target_states, ancillas),
                          method='BFGS', jac=state_infidelity_grad_modular)

        cost = cost_function_modular(result.x, qc0, branch_circuits, target_states, ancillas)
        if cost < best_cost:
            best_cost = cost
    print(best_cost)
