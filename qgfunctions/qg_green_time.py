import cvxpy as cvx
#from pyCSalgos.sl0 import SmoothedL0
import numpy as np
from qiskit import Aer
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
import qiskit

def compute_green_function_time(self, site=0, shots_per_t=1024, backend_name='qasm_simulator'):
    """
    Computes the green's function in time.
    """

    backend = Aer.get_backend(backend_name)

    assert site in self.local_green_circuit, 'The requested site circuit has not been computed.'
    circuits = self.local_green_circuit[site]

    times = self.times[site]
    n_times = self.fourier_parameters[site]['n_measurements']

    green_values = np.zeros_like(times)

    # Creates a measurement circuit.
    qubits = range(self.num_qubits+1)
    qc_a = QuantumRegister(1, 'a')
    qc_q = QuantumRegister(self.num_qubits, 'q')
    qc_c = ClassicalRegister(self.num_qubits+1, 'c')
    meas = QuantumCircuit(qc_a, qc_q, qc_c)
    meas.barrier(qubits)
    meas.measure(qubits, qubits)

    exp_value = []

    # Runs the 4 circuits to compute G(t).
    for case, circuit in circuits.items():
        
        if 'statevector' not in backend_name:
            qc = circuit + meas
        else:
            qc = circuit
        qc = [qc.bind_parameters({self.evo_time: t}) for t in times]

        if 'statevector' not in backend_name:

            # Runs array of circuits, each with an associated time t.
            job = qiskit.execute(qc, backend=backend, shots=shots_per_t)
            result = job.result()
            counts = result.get_counts()

            # Computes the expected value
            prob_0 = np.zeros(n_times)
            prob_1 = np.zeros(n_times)
            for n, count in enumerate(counts):
                for key, value in count.items():
                    if key[-1] == '0':
                        prob_0[n] += value
                    elif key[-1] == '1':
                        prob_1[n] += value
            prob_0 /= shots_per_t
            prob_1 /= shots_per_t

        else:

            result = backend.run(qc).result()
            state = np.array([result.get_statevector(qcc) for qcc in qc])

            prob_0 = np.sum(np.abs(state[:,::2])**2, axis=-1)
            prob_1 = np.sum(np.abs(state[:,1::2])**2, axis=-1)

        exp_value.append(np.real(prob_0 - prob_1))

    # Computes G(t) from the expected value of the 4 circuits.
    green_function_time = 0.5 * (-exp_value[2] + exp_value[1] - 1.j * (exp_value[0] + exp_value[3]))
    
    return green_function_time