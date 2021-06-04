import numpy as np
import scipy.sparse.linalg as scsl

from qiskit import QuantumRegister, QuantumCircuit

### Ground State Theory Method ###

def classical_eigensolver(self):
    '''Uses scipy solver to numerically compute ground energy and state
    Assumes that there is no degeneracies.'''
    
    GE, GS = scsl.eigsh(self.H_sparse, k = 1, which = 'SR')
    
    GS = GS.flatten()
    
    arg_max_coeff = np.argmax(np.abs(GS))
    ang = np.angle(GS[arg_max_coeff])
    new_GS = GS * np.exp(-1j*ang)
    normalized_GS = new_GS/np.linalg.norm(new_GS)
    
    return GE[0], normalized_GS

def initialize_GS(self):
    '''Returns qiskit circuit with initialize gate to the exact ground state'''
    
    q = QuantumRegister(self.num_qubits, 'q') # system qubits

    qc = QuantumCircuit(q)
    qubit_list = [q[ii] for ii in range(self.num_qubits)]
    qc.initialize(self.ground_state_exact, qubit_list)
    
    return  qc