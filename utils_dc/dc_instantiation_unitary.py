from bqskit import Circuit
from bqskit.ir.gates import CXGate
from bqskit.ir.gates import U3Gate
from bqskit.qis.state.state import StateVector
import numpy as np
import itertools
from scipy.optimize import minimize


M00 = np.array([[1, 0], [0, 0]])
M01 = np.array([[0, 0], [1, 0]])
M11 = np.array([[0, 0], [0, 1]])
I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])

SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])


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


def generate_branch_circuits_sim(num_qudits: int, ancillas: list):
    branch_circuits = []
    n = 2 ** len(ancillas)
    data_qubits = [i for i in range(num_qudits) if i not in ancillas]
    for _ in range(n):
        qc = Circuit(num_qudits)
        for q in data_qubits:
            qc.append_gate(U3Gate(), q)
        branch_circuits.append(qc)
    return branch_circuits


def generate_branch_circuits_indep(num_qudits: int,
                                   ancillas: list):
    branch_circuits = []
    tmp_ancillas = []
    ancillas_copy = ancillas.copy()
    while (ancillas_copy):
        tmp_ancillas.append(ancillas_copy.pop(0))
        data_qubits = [q for q in range(num_qudits) if q not in tmp_ancillas]
        for _ in range(2):
            qc = Circuit(num_qudits)
            for q in data_qubits:
                qc.append_gate(U3Gate(), q)
            branch_circuits.append(qc)
    return branch_circuits


def print_clean_matrix(matrix):
    real_arr = np.real(matrix)
    real_arr[np.abs(real_arr) < 1e-10] = 0  # You can adjust the threshold if needed

    # Convert the array to integers or floats
    # int_arr = real_arr.astype(int)
    return real_arr


def collect_by_bit_indices_general(n, bit_positions):
    numbers = [format(i, f'0{n}b') for i in range(2 ** n)]

    # Initialize a list of empty lists to collect indices for each bit pattern
    num_patterns = 2 ** len(bit_positions)
    bit_indices = [[] for _ in range(num_patterns)]

    # Loop through each number's binary representation
    for i, num in enumerate(numbers):
        # Extract the relevant bits based on the bit positions
        relevant_bits = ''.join([num[pos] for pos in bit_positions])
        # Convert the relevant bits to a corresponding index
        pattern_index = int(relevant_bits, 2)
        bit_indices[pattern_index].append(i)

    return bit_indices


def get_sub_matrices(matrix, bit_positions):
    # Determine the size of the matrix (2^n x 2^n)
    n = int(np.log2(matrix.shape[0]))

    # Get indices for all bit combinations for the specified bit positions
    bit_indices = collect_by_bit_indices_general(n, bit_positions)

    # Extract four submatrices based on the specified bit combinations for rows and columns
    sub_matrices = []
    sub_n = 2 ** len(bit_positions)
    # Use bit_indices[0] for columns and vary row indices (00, 01, 10, 11)
    for i in range(sub_n):
        sub_matrix = matrix[np.ix_(bit_indices[i], bit_indices[0])]
        sub_matrices.append(sub_matrix)

    return sub_matrices


def matrix_distance_squared(a: np.ndarray, b: np.ndarray) -> float:
    # Element-wise multiplication of a and the conjugate of b
    inner_product = np.trace(np.dot(a.conj().T, b))
    return inner_product
    # norm_A = np.sqrt(np.sum(np.abs(a)**2))
    # norm_B = np.sqrt(np.sum(np.abs(b)**2))
    # normalized_inner_product = np.abs(inner_product) / (norm_A * norm_B)
    # # print('normalized dist:', normalized_inner_product)
    # return 1 - normalized_inner_product


def computational_basis_states(n):
    basis_states = []

    # Loop over all possible states from 0 to 2^n - 1
    for i in range(2 ** n):
        # Create a zero vector of size 2^n
        state = np.zeros(2 ** n)
        # Set the ith element to 1 to create the computational basis state
        state[i] = 1
        basis_states.append(state)

    return basis_states


def tensor_with_middle_state(n, ancillas, middle_state):
    """
    Create a tensor product for a system with 'n' qubits, where the 'middle_indices' specify
    which qubits are in the state |i>, and the rest have identity operations applied.

    Parameters:
    - n (int): Total number of qubits.
    - middle_indices (list): The indices of the qubits that are in the state |i>.
    - middle_state (list or ndarray): The column vector (|i>) representing the specific state for those qubits.

    Returns:
    - result (ndarray): The tensor product of identity matrices and the state |i>.
    """

    # Initialize the result as an identity matrix for the whole system.
    identity = np.identity(2)

    # Start with an empty tensor product, iterating over each qubit.
    middle_flag = False
    if 0 in ancillas:
        result = middle_state
        middle_flag = True
    else:
        result = identity

    for i in range(1, n):
        if i in ancillas:
            if middle_flag is False:
                result = np.kron(result, middle_state)
                middle_flag = True
            else:
                continue
        else:
            result = np.kron(result, identity)
    return result


def cost_function(params, qc, branch_circuits, T, ancillas):
    qc.set_params(params[:qc.num_params])
    Ub = qc.get_unitary()  # already check Ub is correct
    params_branch_circuits = params[qc.num_params:]
    i = 0
    utrys = []
    for branch_ciruit in branch_circuits:  # already check all the branch circuits are correct
        branch_ciruit.set_params(params_branch_circuits[i:i + branch_ciruit.num_params])
        utrys.append(branch_ciruit.get_unitary())
        i += branch_ciruit.num_params
    d = 0

    output_states = computational_basis_states(len(ancillas))

    Ms = generate_measurement_operators(ancillas, qc.num_qudits)
    Us = np.zeros((Ub.shape[0], Ub.shape[0]), dtype='complex')

    for branch_utry, M in zip(utrys, Ms):
        Us += branch_utry @ M @ Ub

    num_system = qc.num_qudits - len(ancillas)
    ket_0 = np.array([[1], [0]])
    right = ket_0 if 0 in ancillas else I

    for i in range(1, qc.num_qudits):
        if i in ancillas:
            right = np.kron(right, ket_0)
        else:
            right = np.kron(right, I)

    ks = []
    for state in output_states:
        left = tensor_with_middle_state(qc.num_qudits, ancillas, state)
        k = left @ Us @ right
        ks.append(k)
    d = 0

    for k in ks:
        product = matrix_distance_squared(k, T)
        d += abs(product) ** 2
    distance = 1 - d / (2 ** num_system) ** 2

    return distance


def get_partial_unitary(qc, branch_circuits, ancillas):
    Ub = qc.get_unitary()  # already check Ub is correct
    utrys = []
    for branch_ciruit in branch_circuits:  # already check all the branch circuits are correct
        utrys.append(branch_ciruit.get_unitary())
    Ms = generate_measurement_operators(ancillas, qc.num_qudits)
    Us = np.zeros((Ub.shape[0], Ub.shape[0]), dtype='complex')

    for branch_utry, M in zip(utrys, Ms):
        Us += branch_utry @ M @ Ub

    return Us


def cost_function_indep_measure(params, circuits, branch_circuits, T, ancillas):
    base_i = 0
    Ubs = []
    for circuit in circuits:
        circuit.set_params(params[base_i: base_i + qc.num_params])
        Ubs.append(circuit.get_unitary())
        base_i += qc.num_params
    params_branch_circuits = params[base_i:]
    i = 0
    utrys = []
    for branch_ciruit in branch_circuits:  # already check all the branch circuits are correct
        branch_ciruit.set_params(params_branch_circuits[i:i + branch_ciruit.num_params])
        utrys.append(branch_ciruit.get_unitary())
        i += branch_ciruit.num_params

    Us = np.eye(2 ** circuits[0].num_qudits)
    for qc, ancilla, idx in zip(circuits, ancillas, range(len(circuits))):
        qc_branch_circuits = branch_circuits[idx * 2:idx * 2 + 2]
        Us = Us @ get_partial_unitary(qc, qc_branch_circuits, [ancilla])

    num_system = qc.num_qudits - len(ancillas)
    ket_0 = np.array([[1], [0]])
    right = ket_0 if 0 in ancillas else I

    for i in range(1, qc.num_qudits):
        if i in ancillas:
            right = np.kron(right, ket_0)
        else:
            right = np.kron(right, I)

    ks = []

    output_states = computational_basis_states(len(ancillas))

    for state in output_states:
        left = tensor_with_middle_state(qc.num_qudits, ancillas, state)
        k = left @ Us @ right
        ks.append(k)

    d = 0

    for k in ks:
        product = matrix_distance_squared(k, T)
        d += abs(product) ** 2
    distance = 1 - d / (2 ** num_system) ** 2

    return distance


def process_complex_matrix(matrix, threshold=1e-6):
    """
    Process a complex matrix:

    1. If real part < threshold, set it to 0.
    2. If real part >= threshold, round it to 2 decimals.
    3. Apply the same rules for the imaginary part.

    Parameters:
    - matrix: np.ndarray (complex matrix)
    - threshold: float (threshold for filtering)

    Returns:
    - processed_matrix: np.ndarray (processed complex matrix)
    """
    # Initialize the processed matrix with the same shape as the input
    processed_matrix = np.zeros_like(matrix, dtype=complex)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            real_part = matrix[i, j].real
            imag_part = matrix[i, j].imag

            # Process the real part
            if real_part < threshold and imag_part < threshold:
                processed_matrix[i, j] = 0
            elif real_part < threshold and imag_part >= threshold:
                processed_matrix[i, j] = 1j * round(imag_part, 2)
            elif real_part >= threshold and imag_part < threshold:
                processed_matrix[i, j] = round(real_part, 2)
            else:
                processed_matrix[i, j] = complex(round(real_part, 2), round(imag_part, 2))

    return processed_matrix


def process_complex_array(array, threshold=1e-6):
    """
    Process a complex matrix:

    1. If real part < threshold, set it to 0.
    2. If real part >= threshold, round it to 2 decimals.
    3. Apply the same rules for the imaginary part.

    Parameters:
    - matrix: np.ndarray (complex matrix)
    - threshold: float (threshold for filtering)

    Returns:
    - processed_matrix: np.ndarray (processed complex matrix)
    """
    # Initialize the processed matrix with the same shape as the input
    processed_array = np.zeros_like(array, dtype=complex)

    for i in range(array.shape[0]):
        real_part = array[i].real
        imag_part = array[i].imag

        # Process the real part
        if real_part < threshold and imag_part < threshold:
            processed_array[i] = 0
        elif real_part < threshold and imag_part >= threshold:
            processed_array[i] = 1j * round(imag_part, 2)
        elif real_part >= threshold and imag_part < threshold:
            processed_array[i] = round(real_part, 2)
        else:
            processed_array[i] = complex(round(real_part, 2), round(imag_part, 2))

    return processed_array




def unitary_grad_function(params, qc, branch_circuits, T, ancillas):
    qc.set_params(params[:qc.num_params])
    Ub = qc.get_unitary()
    gradb = qc.get_grad()
    params_branch_circuits = params[qc.num_params:]
    i = 0
    branch_utrys = []
    branch_grads = []
    for branch_ciruit in branch_circuits:
        branch_ciruit.set_params(params_branch_circuits[i:i + branch_ciruit.num_params])
        branch_utrys.append(branch_ciruit.get_unitary())
        branch_grads.append(branch_ciruit.get_grad())
        i += branch_ciruit.num_params

    Us = np.zeros((Ub.shape[0], Ub.shape[0]), dtype='complex')
    Ms = generate_measurement_operators(ancillas, qc.num_qudits)

    for branch_utry, M in zip(branch_utrys, Ms):
        Us += branch_utry @ M @ Ub

    ket_0 = np.array([[1], [0]])
    right = ket_0 if 0 in ancillas else I

    for i in range(1, qc.num_qudits):
        if i in ancillas:
            right = np.kron(right, ket_0)
        else:
            right = np.kron(right, I)

    output_states = computational_basis_states(len(ancillas))

    num_system = qc.num_qudits - len(ancillas)
    normalized_factor = num_system ** 2

    ks = []
    lefts = []
    for state in output_states:
        left = tensor_with_middle_state(qc.num_qudits, ancillas, state)
        lefts.append(left)
        k = left @ Us @ right
        ks.append(k)

    d_infidelity = []

    for dv_matrix0 in gradb:
        jacs = 0
        for k in ks:
            s = np.trace(np.dot(T.conj().T, k))
            jus = 0
            for branch_utry, M, left in zip(branch_utrys, Ms, lefts):
                jus += np.trace(left @ branch_utry @ M @ dv_matrix0 @ right @ T.conj().T)
            jacs += -2 * ((np.real(s) * np.real(jus) + np.imag(s) * np.imag(jus))) / normalized_factor
        d_infidelity.append(jacs)

    for branch_utry, branch_grad, M, left in zip(branch_utrys, branch_grads, Ms, lefts):
        for dv_matrix in branch_grad:
            jacs = 0
            for k in ks:
                s = np.trace(np.dot(T.conj().T, k))
                jus = np.trace(left @ dv_matrix @ M @ Ub @ right @ T.conj().T)
                jacs += -2 * ((np.real(s) * np.real(jus) + np.imag(s) * np.imag(jus))) / normalized_factor
            d_infidelity.append(jacs)

    return d_infidelity


def test_partial_trace(rho, dims, axis=0):
    """
    Takes partial trace over the subsystem defined by 'axis'
    rho: a matrix
    dims: a list containing the dimension of each subsystem
    axis: the index of the subsytem to be traced out
    (We assume that each subsystem is square)
    """
    dims_ = np.array(dims)
    # Reshape the matrix into a tensor with the following shape:
    # [dim_0, dim_1, ..., dim_n, dim_0, dim_1, ..., dim_n]
    # Each subsystem gets one index for its row and another one for its column
    reshaped_rho = rho.reshape(np.concatenate((dims_, dims_), axis=None))

    # Move the subsystems to be traced towards the end
    reshaped_rho = np.moveaxis(reshaped_rho, axis, -1)
    reshaped_rho = np.moveaxis(reshaped_rho, len(dims) + axis - 1, -1)

    # Trace over the very last row and column indices
    traced_out_rho = np.trace(reshaped_rho, axis1=-2, axis2=-1)

    # traced_out_rho is still in the shape of a tensor
    # Reshape back to a matrix
    dims_untraced = np.delete(dims_, axis)
    rho_dim = np.prod(dims_untraced)
    return traced_out_rho.reshape([rho_dim, rho_dim])


def rzz(theta):
    rzz_mat = np.array([
        [np.exp(-1j * theta / 2), 0, 0, 0],
        [0, np.exp(1j * theta / 2), 0, 0],
        [0, 0, np.exp(1j * theta / 2), 0],
        [0, 0, 0, np.exp(-1j * theta / 2)]
    ])
    return rzz_mat


def callback_function(params, qc, branch_circuits, Ts, ancillas):
    qc.set_params(params[:qc.num_params])
    Ub = qc.get_unitary()
    print('----ub-----')
    print(process_complex_matrix(Ub))

    print('---current state----')

    initial_state = [0] * 2 ** qc.num_qudits
    # initial_state[0] = 1
    for i in range(2 ** qc.num_qudits):
        initial_state[i] = np.random.random() + 1j * np.random.random()
    initial_state /= np.sqrt(np.sum(np.square(initial_state)))
    second_state = Ub @ initial_state
    print(process_complex_array(second_state))

    anc_1_index = [2, 3, 6, 7]
    # anc_1_index = [2, 3, 4, 5, 6, 7,10, 11, 12, 13,14, 15]
    anc_1_weight = sum([second_state[i] for i in anc_1_index])
    print('anc 1 weight:', anc_1_weight)

    params_branch_circuits = params[qc.num_params:]
    i = 0
    utrys = []
    for branch_ciruit in branch_circuits:  # already check all the branch circuits are correct
        branch_ciruit.set_params(params_branch_circuits[i:i + branch_ciruit.num_params])
        utrys.append(branch_ciruit.get_unitary())
        i += branch_ciruit.num_params

    Ms = generate_measurement_operators(ancillas, qc.num_qudits)  # already check all the measurement opts

    for branch_u, M, T in zip(utrys, Ms, Ts):
        distance = matrix_distance_squared(T, (branch_u @ M @ Ub))
        print('distance:', distance)

    print("Current cost:", cost_function(params, qc, branch_circuits, Ts, ancillas))



def unitary_instantiation(base_circuit, branch_circuits, ancillas, target_unitary):
    num_params = base_circuit.num_params + sum([br.num_params for br in branch_circuits])

    multi_starts = 20
    best_cost = np.inf

    for _ in range(multi_starts):
        initial_params = np.random.uniform(low=-np.pi, high=np.pi, size=num_params)
        result = minimize(cost_function, initial_params, method='BFGS',
                          jac=unitary_grad_function, args=(base_circuit, branch_circuits, target_unitary, ancillas),
                          )
        cost = cost_function(result.x, base_circuit, branch_circuits, target_unitary, ancillas)
        # print('---cost---', cost)
        if cost < 1e-10:
            best_cost = cost
            best_params = result.x
            break
    return best_cost

    
if __name__ == '__main__':

    num_qubits = 4
    qc = Circuit(num_qubits)
    for i in range(num_qubits):
        qc.append_gate(U3Gate(), i)

    # CXs = [(0, 1), (2, 3), (4, 5), (1, 2), (3, 4), (5, 6)]
    CXs = [(0, 1), (2, 3)]
    for cx in CXs:
        qc.append_gate(CXGate(), cx)
        for q in cx:
            qc.append_gate(U3Gate(), q)

    # for cx in CXs:
    #     qc.append_gate(CXGate(), cx)
    #     for q in cx:
    #         qc.append_gate(U3Gate(), q)

    # for cx in CXs:
    #     qc.append_gate(CXGate(), cx)
    #     for q in cx:
    #         qc.append_gate(U3Gate(), q)

    ancillas = [1, 2]

    num_branch_circuits = 2 ** len(ancillas)

    branch_circuits = []
    for _ in range(num_branch_circuits):
        br = Circuit(num_qubits)
        br.append_gate(U3Gate(), 0)
        br.append_gate(U3Gate(), num_qubits - 1)
        branch_circuits.append(br)

    # T = np.array([
    #     [1, 0, 0, 0],
    #     [0, 1, 0, 0],
    #     [0, 0, 0, 1],
    #     [0, 0, 1, 0]
    # ])

    target_qc = Circuit(2)
    target_qc.append_gate(CXGate(), (0, 1))
    # angles = np.random.uniform(-np.pi, np.pi, 3)
    # target_qc.append_gate(RZZGate(), (0, 1), [angles[0]])
    # target_qc.append_gate(RXXGate(), (0, 1), [angles[1]])
    # target_qc.append_gate(RXXGate(), (0, 1), [angles[2]])

    T = target_qc.get_unitary()

    num_params = qc.num_params + sum([br.num_params for br in branch_circuits])

    multi_starts = 20
    best_cost = np.inf

    for _ in range(multi_starts):
        initial_params = np.random.uniform(low=-np.pi, high=np.pi, size=num_params)
        result = minimize(cost_function, initial_params, method='BFGS',
                          jac=unitary_grad_function, args=(qc, branch_circuits, T, ancillas),
                          )
        cost = cost_function(result.x, qc, branch_circuits, T, ancillas)
        print('---cost---', cost)
        if cost < 1e-10:
            best_cost = cost
            best_params = result.x
            break
    print(best_cost)
    print(best_params)

    qc.set_params(best_params[:qc.num_params])
    Ub = qc.get_unitary()
    params_branch_circuits = best_params[qc.num_params:]
    i = 0
    branch_utrys = []
    for branch_ciruit in branch_circuits:
        branch_ciruit.set_params(params_branch_circuits[i:i + branch_ciruit.num_params])
        branch_utrys.append(branch_ciruit.get_unitary())
        i += branch_ciruit.num_params

    # qc.save('/global/homes/s/siyuan/DC_hpc/dc_circuits_cx_5anc/main.qasm')

    # for i, circ in enumerate(branch_circuits):
    #     circ.save(f'/global/homes/s/siyuan/DC_hpc/dc_circuits_cx_5anc/branch_{i}.qasm')




