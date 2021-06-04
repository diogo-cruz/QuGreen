from math import pi
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister

def green_circuit_dict(self, site):
        
    qr_a = QuantumRegister(1, 'a') # ancillary register
    qr_q = QuantumRegister(self.num_qubits, 'q') # system qubits
    
    # circuit that performs controlled-X operation
    qc_ctrX = QuantumCircuit(qr_a, qr_q)
    qc_ctrX.cnot(qr_a, qr_q[site])
    
    # circuit that performs controlled-Y operation
    qc_ctrY = QuantumCircuit(qr_a, qr_q)
    qc_ctrY.s(qr_a)
    qc_ctrY.ry(pi/2,  qr_q[site])
    qc_ctrY.cnot(qr_a, qr_q[site])
    qc_ctrY.ry(-pi/2,  qr_q[site])
    qc_ctrY.cnot(qr_a, qr_q[site])
    
    # dictionary of controlled circuits
    qc_ctr = {}
    qc_ctr['X'] = qc_ctrX
    qc_ctr['Y'] = qc_ctrY

    qc_dict = {} # initialize empty circuit dictionary  
    
    gates = ['X', 'Y'] # just to facilitate loop bellow
    
    for gate_1 in gates:
        for gate_2 in gates:
        
            qc = QuantumCircuit(qr_a, qr_q) # initialize empty circuit
            qc.h(qr_a[0])

            # some four gates
            qc += self.ground_state_circuit + qc_ctr[gate_1] + self.time_evolution_circuit + qc_ctr[gate_2]

            qc.h(qr_a[0])

            qc_dict[(gate_1, gate_2)] = qc

    return qc_dict