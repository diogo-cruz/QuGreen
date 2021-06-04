from qiskit import Aer

from qiskit.circuit.library import TwoLocal
from qiskit.aqua.components.optimizers import SPSA
from qiskit.aqua.algorithms import VQE
from qiskit.aqua import QuantumInstance

import os
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.test.mock import FakeVigo
device_backend = FakeVigo()

### Ground State VQE Method

def set_vqe_options(**kwargs):
    
    vqe_options = {}
    
    if 'maxiter' in kwargs:
        vqe_options['maxiter'] = kwargs['maxiter']
    else:
        vqe_options['maxiter'] = 100
        
    if 'reps' in kwargs:
        vqe_options['n_steps'] = kwargs['reps']
    else:
        vqe_options['n_steps'] = 3
        
    if 'noise' in kwargs:
        vqe_options['noise'] = kwargs['noise']
    else:
        vqe_options['noise'] = False
        
    return vqe_options

def set_vqe_circuit(self, backend = None):    
    #Check https://qiskit.org/documentation/tutorials/algorithms/03_vqe_simulation_with_noise.html
    #seed = 170
    
    iterations = self.vqe_options['maxiter']
    #aqua_globals.random_seed = seed
    if backend is None:
        backend = 'statevector_simulator'
    backend = Aer.get_backend(backend)

    counts = []
    values = []
    stds = []
    def store_intermediate_result(eval_count, parameters, mean, std):
        counts.append(eval_count)
        values.append(mean)
        stds.append(std)

    var_form = TwoLocal(reps = self.vqe_options['n_steps'], 
                        rotation_blocks = 'ry', 
                        entanglement_blocks = 'cx', 
                        entanglement = 'linear',
                        insert_barriers = True)
    spsa = SPSA(maxiter=iterations)

    if self.vqe_options['noise']:
        os.environ['QISKIT_IN_PARALLEL'] = 'TRUE'
        device = QasmSimulator.from_backend(device_backend)
        coupling_map = device.configuration().coupling_map
        noise_model = NoiseModel.from_backend(device)
        basis_gates = noise_model.basis_gates

        qi = QuantumInstance(backend=backend,
                            coupling_map=coupling_map,
                            noise_model=noise_model)

    else:
        qi = QuantumInstance(backend=backend)

    vqe = VQE(var_form=var_form, optimizer=spsa, callback=store_intermediate_result, quantum_instance=qi)
    result = vqe.compute_minimum_eigenvalue(operator=self.H)

    return vqe.get_optimal_circuit(), vqe.optimal_params, vqe.get_optimal_vector(), vqe.get_optimal_cost()