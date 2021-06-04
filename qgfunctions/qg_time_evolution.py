from qiskit.aqua.operators import PauliTrotterEvolution, Suzuki
from qiskit.circuit import Parameter
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit import execute
from qiskit import assemble
from qiskit import Aer
from scipy.optimize import minimize
import numpy as np
from qiskit.aqua.components.optimizers import SPSA



### Trotterization ###

# remember this link: ttps://qiskit.org/documentation/tutorials/operators/01_operator_flow.html

def set_trot_options(**kwargs):
    
    trot_options = {}
    
    if 'product_formula' in kwargs:
        trot_options['product_formula'] = kwargs['product_formula']
    else:
        trot_options['product_formula'] = 'lietrotter'
        
    if 'reps' in kwargs:
        trot_options['reps'] = kwargs['reps']
    else:
        trot_options['reps'] = 1
        
    if 'order' in kwargs:
        trot_options['order'] = kwargs['order']
    else:
        trot_options['order'] = 1

    return trot_options

def set_trotter_circuit(self):
    '''Creates time evolution circuit with time as free parameter 
    according the method specified by self.trot_options'''
    
    evo_H = self.evo_H                         # this may be different from self.H
    evo_op   = (self.evo_time * evo_H).exp_i() # evolution operator
    
    if self.trot_options['product_formula'] == 'lietrotter':
        
        num_reps = self.trot_options['reps']
        
        trotterized_op = PauliTrotterEvolution(trotter_mode = 'trotter', reps = num_reps).convert(evo_op)
        qc = trotterized_op.to_circuit()
        
    elif self.trot_options['product_formula'] == 'suzuki':
        
        num_reps = self.trot_options['reps']
        order    = self.trot_options['order']
        
        trotterized_op = PauliTrotterEvolution(trotter_mode = Suzuki(order = order, reps = num_reps)).convert(evo_op)
        qc = trotterized_op.to_circuit()
        
    return qc



### VFF ###


'''
Original reference:
Gibbs, J., Gili, K., Holmes, Z., Commeau, B., Arrasmith, A., Cincio, L., Coles, P. J., & Sornborger, A. (2021). 
Long-time simulations with high fidelity on quantum hardware. 1–20. 
http://arxiv.org/abs/2102.04313
'''

def set_VFF_options(self, **kwargs):
    
    vff_options = {}
    
    if 'site' in kwargs:
        vff_options['site'] = kwargs['site']
    else:
        vff_options['site'] = 0

    if 'training_time' in kwargs:
        vff_options['training_time'] = kwargs['training_time']
    else:
        vff_options['training_time'] = 0.02 # arbitrary number

    if 'n_eig' in kwargs:
        vff_options['n_eig'] = kwargs['n_eig']
    else:
        vff_options['n_eig'] =  self.num_qubits + 1

    if 'n_layers' in kwargs:
        vff_options['n_layers'] = kwargs['n_layers']
    else:
        vff_options['n_layers'] = 2

    if 'num_shots' in kwargs:
        vff_options['num_shots'] = kwargs['num_shots']
    else:
        vff_options['num_shots'] = 30000

    if 'min_method' in kwargs:
        vff_options['min_method'] = kwargs['min_method']
    else:
        vff_options['min_method'] = 'BFGS'

    if 'maxiter' in kwargs:
        vff_options['maxiter'] = kwargs['maxiter']
    else:
        vff_options['maxiter'] = 100

    if 'tolerance' in kwargs:
        vff_options['tolerance'] = kwargs['tolerance']
    else:
        vff_options['tolerance'] = 0.0001

    if 'target_fidelity' in kwargs:
        vff_options['target_fidelity'] = kwargs['target_fidelity']
    else:
        vff_options['target_fidelity'] = 0.00001

    if 'max_reps' in kwargs:
        vff_options['max_reps'] = kwargs['max_reps']
    else:
        vff_options['max_reps'] = 5

    if 'n_diagonal_gates' in kwargs:
        vff_options['n_diagonal_gates'] = kwargs['n_diagonal_gates']
    else:
        vff_options['n_diagonal_gates'] = self.num_qubits



    return vff_options



def set_VFF_circuit(self, backend, random=True):
    '''Creates time evolution circuit with time as free parameter 
    according the method specified by self.trot_options'''
    
    LET_circuits_dict = set_LET_circuits(self) # creates LET circuits for different values of k

    optimal_params, optimized_cost = LET_optimizer(self, backend, LET_circuits_dict, random) # optimizes VFF paramerters

    # Create optimal VFF circuit
    VFF_circuit = set_VFF_ansatz(self, optimal_params)

    return VFF_circuit, optimized_cost, optimal_params


def set_LET_circuit(self, k):
    '''Loschmidt Echo Test (LET circuit) of order k.
    The circuit is of the form: 
        S . U^k . W(theta)^dag . (D(alpha)^dag)^k . W(theta) . S^dag
    where: S is state preparation circuit,
        U is single step Lie-Trotter evolution for training time 
        W(theta) is parametrized unitary
        D(alpha) is parametrized diagonal operator
    '''
    
    qr_a = QuantumRegister(1, 'a')               # ancillary register
    qr_q = QuantumRegister(self.num_qubits, 'q') # system qubits
    
    qc = QuantumCircuit(qr_a, qr_q) # initialize empty circuit
    
    
    
    # S 
    S = S_circuit(self)
    qc += S
    
    # U^k 
    Ut = set_trotter_circuit(self) # trotter options have already been set to single-step 1st order formula
    U = Ut.bind_parameters({Ut.parameters[0]: self.vff_options['training_time']}) # Ut has time as the only free parameter
    for rep in range(k):
        qc += U
    qc.barrier()
    
        
    # W(theta)^dag
    W = W_ansatz(self)
    qc += W.inverse()
    qc.barrier()
    
    # D(alpha)^dag
    qc += D_ansatz(self, k).inverse()
    qc.barrier()
    
    # W(theta)
    qc += W
    
    # S^dag
    qc += S.inverse()
    
    return qc  


def S_circuit(self):

    qr_a = QuantumRegister(1, 'a')               # ancillary register
    qr_q = QuantumRegister(self.num_qubits, 'q') # system qubits
    
    S = QuantumCircuit(qr_a, qr_q) # initialize empty circuit
    S.barrier()
    S += self.ground_state_circuit # ground state preparation part
    S.barrier()
    S.h(qr_a[0])             # particle/hole states excitation part
    S.cnot(qr_a, qr_q[self.vff_options['site']]) # particle/hole states excitation part
    S.barrier()

    return S

def W_ansatz(self):

    qr_a = QuantumRegister(1, 'a')               # ancillary register
    qr_q = QuantumRegister(self.num_qubits, 'q') # system qubits

    W = QuantumCircuit(qr_a, qr_q) # initialize empty circuit
    W += TwoLocal(num_qubits = self.num_qubits,
                  reps = self.vff_options['n_layers'], 
                  rotation_blocks = ['ry','rz'], 
                  entanglement_blocks = 'cx', 
                  entanglement = 'linear',
                  insert_barriers = True)

    return W

def W_ansatz_OLD(self):

    qr_a = QuantumRegister(1, 'a')               # ancillary register
    qr_q = QuantumRegister(self.num_qubits, 'q') # system qubits

    W = QuantumCircuit(qr_a, qr_q) # initialize empty circuit
    W.cz(4, 1)
    W.cz(4, 2)
    W.cnot(4, 1)
    W.cnot(4, 2)
    W.cnot(4, 3)
    W.h(4)

    return W

def D_ansatz(self, k):

    qr_a = QuantumRegister(1, 'a')               # ancillary register
    qr_q = QuantumRegister(self.num_qubits, 'q') # system qubits

    D = QuantumCircuit(qr_a, qr_q) # initialize empty circuit
    for qubit in range(self.vff_options['n_diagonal_gates']):
        D.rz(k * Parameter('α_{}'.format(qubit)), qr_q[qubit])

    return D

def D_ansatz_OLD(self, k):

    qr_a = QuantumRegister(1, 'a')               # ancillary register
    qr_q = QuantumRegister(self.num_qubits, 'q') # system qubits

    D = QuantumCircuit(qr_a, qr_q) # initialize empty circuit

    D.rz(k * Parameter('α_{}'.format(qubit)), 2)
    D.rz(k * Parameter('α_{}'.format(qubit)), 3)

    return D



def set_LET_circuits(self):
    '''Creates a dictionary of LET circuits for k from 1 to n_eig.
    The underlying goal is to avoid creating new circuits everytime the cost function is called.'''

    LET_circuits_dict = {k: set_LET_circuit(self, k) for k in range(1, self.vff_options['n_eig'] + 1)}

    return LET_circuits_dict



def set_VFF_ansatz(self, params):
    '''Creates VFF circuit (W . D . W^dag) with :params:'''

    alphas = params[: self.vff_options['n_diagonal_gates'] ] # the first angles are alphas
    thetas = params[self.vff_options['n_diagonal_gates'] :]  # the latter angles are thetas
    
    qr_a = QuantumRegister(1, 'a')               # ancillary register
    qr_q = QuantumRegister(self.num_qubits, 'q') # system qubits
    
    qc = QuantumCircuit(qr_a, qr_q) # initialize empty circuit
    
        
    # W(theta)
    W = W_ansatz(self)
    # set parameters to numerical values
    params_names = W.parameters
    params_size  = len(params_names)
    params_dict  = {params_names[i]: thetas[i] for i in range(params_size)}
    W = W.bind_parameters(params_dict) 
    # Add W^dag
    qc += W.inverse()
    qc.barrier()
    
    # Add D
    D = D_ansatz(self, self.evo_time / self.vff_options['training_time'])
    params_names = D.parameters
    params_size  = len(params_names) - 1  # exclude time paramter
    params_dict  = {params_names[i + 1]: alphas[i] for i in range(params_size)} 
    qc += D.bind_parameters(params_dict) 
    qc.barrier()
    
    # Add W
    qc += W
    
    return qc


def cost_function_LET(params, self, backend, LET_circuits_dict, callback):
    '''Returns cost of LET circuit with :params:'''

    cost = 1.
    
    for k in range(1, self.vff_options['n_eig'] + 1):

        var_form = LET_circuits_dict[k] # LET circuit of order k with angles as free parameters

        params_names = var_form.parameters
        params_size  = len(params_names)

        assert len(params) == params_size, 'parameters array and circuit parameters not the same size'
    
        # set parameters to numerical values
        params_dict = {params_names[i]: params[i] for i in range(params_size)}
        qc = var_form.bind_parameters(params_dict) 

        if backend == 'qasm_simulator':

            qc.measure_all()
            
            # run circuit many times to estimate the prob of zero state
            job = execute(qc,
                          backend = Aer.get_backend(backend),
                          shots = self.vff_options['num_shots']
                         )
            job_counts = job.result().get_counts()

            try:
                counts_0 = job_counts['{:0<{}d}'.format(0, self.num_qubits + 1)]
            except:
                counts_0 = 0

            prob_0   = counts_0 / self.vff_options['num_shots']
            
            
            cost += - prob_0 / self.vff_options['n_eig']

        elif backend == 'statevector_simulator':

            svsim = Aer.get_backend(backend)
            qobj = assemble(qc)    
            result = svsim.run(qobj).result() 
            out_state = result.get_statevector()

            amplitude_0 = out_state[0]
            prob_0 = np.abs(amplitude_0) ** 2

            cost += - prob_0 / self.vff_options['n_eig']

    callback.append(cost)

    return cost



def LET_optimizer(self, backend, LET_circuits_dict, random):
    '''Finds set of parameters that optimize LET circuit'''

    params_size = len(LET_circuits_dict[1].parameters) # number of parameters to optimize

    variable_bounds = []
    for var in range(self.vff_options['n_diagonal_gates']):
        variable_bounds.append((0, 6 * self.vff_options['training_time']))
    for var in range(self.vff_options['n_diagonal_gates'], params_size):
        variable_bounds.append((0, 2 * np.pi))
    callback_list = []

    # the SPSA optimizer has a distinct implementation, being a part of qiskit
    if self.vff_options['min_method'] == 'SPSA':


        spsa_optimizer = SPSA(maxiter = self.vff_options['maxiter'])

        # setting the non-variable parameters
        wrapped_cost_function = spsa_optimizer.wrap_function(function = cost_function_LET, 
                                                             args = (self, backend, LET_circuits_dict, callback_list)
                                                            )

        opt_cost = 1.
        opt_params = np.zeros(params_size)
        repetition = 0
        while opt_cost > self.vff_options['target_fidelity'] and repetition < self.vff_options['max_reps']:

            seed = np.zeros(params_size)
            seed[: self.vff_options['n_diagonal_gates']] = 6 * self.vff_options['training_time'] * np.random.random(self.vff_options['n_diagonal_gates'])
            seed[self.vff_options['n_diagonal_gates'] :] = 2 * np.pi * np.random.random(params_size - self.vff_options['n_diagonal_gates'])
            #seed = 2 * np.pi * np.random.random(params_size) # random initial guess for optimal parameters


            new_opt_params, new_opt_cost, new_nfenv = spsa_optimizer.optimize(objective_function = wrapped_cost_function,
                                                                  num_vars = params_size,
                                                                  initial_point = seed,
                                                                  variable_bounds = variable_bounds
                                                                 )

            if new_opt_cost < opt_cost:
                opt_cost = new_opt_cost
                opt_params = new_opt_params

            repetition += 1
            
        self.callback_vff = callback_list

    else:

        if random:        
            seed = np.zeros(params_size)
            seed[: self.vff_options['n_diagonal_gates']] = 6 * self.vff_options['training_time'] * np.random.random(self.vff_options['n_diagonal_gates'])
            seed[self.vff_options['n_diagonal_gates'] :] = 2 * np.pi * np.random.random(params_size - self.vff_options['n_diagonal_gates'])
        else:
            seed = self.optimal_VFF_params

        # scipy.optimize's minimization method
        out = minimize(cost_function_LET, 
                       x0      = seed,
                       args    = (self, backend, LET_circuits_dict, callback_list),
                       method  = self.vff_options['min_method'], 
                       tol     = self.vff_options['tolerance'],
                       options = {'maxiter': self.vff_options['maxiter'],
                                  'gtol': self.vff_options['tolerance'],
                                  'ftol': self.vff_options['tolerance'],
                                  'display': True}
                      )

        opt_params = out['x']
        opt_cost   = out['fun']

        self.callback_vff = callback_list


    print('opt_cost', opt_cost)


    return opt_params, opt_cost




