# UCCSD evolution Code after OpenFermion-ProjectQ

# Copyright 2017 The OpenFermion Developers; re-used, modified and
# redistributed under the Apache License, Version 2.0.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from openfermion.transforms import jordan_wigner
from openfermion.circuits import uccsd_singlet_generator, uccsd_singlet_paramsize
import openfermion as of
import numpy as np

# import qiskit.opflow as qk_opflow
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
import qiskit.quantum_info as qk_qi
from qiskit.synthesis.evolution import LieTrotter, SuzukiTrotter

__all__ = [
    'singlet_evolution',
    'singlet_paramsize',
]

singlet_paramsize = uccsd_singlet_paramsize


def singlet_evolution(
    packed_amplitudes,
    n_qubits: int,
    n_electrons: int,
    fermion_transform=jordan_wigner,
    trotter_mode:str='suzuki',
    reps:int=2,
) -> PauliEvolutionGate:
    """\
    Create a Qiskit evolution gate for a UCCSD singlet circuit

    Args:
        packed_amplitudes(ndarray): compact array storing the unique single
            and double excitation amplitudes for a singlet UCCSD opflow.
            The ordering lists unique single excitations before double
            excitations
        n_qubits(int): number of spin-orbitals used to represent the system,
            which also corresponds to number of qubits in a non-compact map.
        n_electrons(int): number of electrons in the physical system
        fermion_transform(openfermion.transform): The transformation that
            defines the mapping from Fermions to QubitOperator
        trotter_mode(str): The Trotterization mode to use. Either 'suzuki' or
            'lie'. Default is 'suzuki'.
        reps(int): Number of repetitions for the Trotterization. Default is 2.

    Returns:
        PauliEvolutionGate: The evolution gate for the UCCSD singlet circuit
    """

  # From OpenFermion: The uccsd_singlet_generator generates a FermionOperator
  # for a UCCSD generator designed to act on a single reference state consisting
  # of n_qubits spin orbitals and n_electrons electrons, that is a spin singlet
  # operator, meaning it conserves spin.
    fermion_generator = uccsd_singlet_generator(
                            packed_amplitudes, n_qubits, n_electrons)

  # Transform generator to qubits, using only real coefficients
    qubit_generator = fermion_transform(fermion_generator)
    for key, coeff in qubit_generator.terms.items():
        qubit_generator.terms[key] = float(coeff.imag)
    qubit_generator.compress()

  # Translate the OpenFermion QubitOperators to Qiskit opflow
    pauli_strings = []
    coeffs = []
    for paulis, coeff in sorted(qubit_generator.terms.items()):
        ops = ['I']*n_qubits
        for term in paulis:
            ops[term[0]] = term[1]

        ops.reverse()

        pauli_strings.append(''.join(ops))
        coeffs.append(coeff)

    opflow = SparsePauliOp(pauli_strings, coeffs=np.array(coeffs, dtype=np.complex128))

  # Exponentiate one time step
    time = 1.
    # evolution_op = (time*opflow).exp_i()
    if trotter_mode == 'suzuki':
        synthesis = SuzukiTrotter(reps=reps)
    elif trotter_mode == 'lie' or trotter_mode == 'trotter':
        synthesis = LieTrotter(reps=reps)
    else:
        raise ValueError('Invalid trotter_mode. Must be either "suzuki" or "lie".')
    evolution_op = PauliEvolutionGate(opflow, time=time, synthesis=synthesis)

    return evolution_op
