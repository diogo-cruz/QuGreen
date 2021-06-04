### Main Class

### Library Imports ### 

from qiskit.quantum_info import Pauli
from qiskit.aqua.operators.primitive_ops.pauli_op import PauliOp
from qiskit.aqua.operators import X, Y, Z, I
import qiskit

import scipy as sc
import numpy as np
import matplotlib.pyplot as plt

plt.rc('text',usetex=True)
plt.rc('font', family='serif', size=16)
plt.rcParams['figure.figsize'] = [10, 6]

from datetime import datetime

### Our Imports ###

from qgfunctions.qg_groundstate_theory import *
from qgfunctions.qg_groundstate_vqe import *
from qgfunctions.qg_time_evolution import *
from qgfunctions.qg_local_green import *
from qgfunctions.qg_green_time import *
from qgfunctions.qg_compressed_sensing import *


### Main Class ###

class QuGreen:
    


    def __init__(self, H):
        """Start by indicating the Model of the System."""
        
        self.H = H                           # spin/qubit Hamiltonian given in qq format
        self.H_sparse = self.H.to_spmatrix() # H in sparse matrix format
        self.num_qubits = self.H.num_qubits  # number of qubits of system
        self.local_green_circuit = {}
        self.green_function_time = {}
        self.green_time_exact = {}
        self.green_function_freq = {}
        self.fourier_parameters = {}
        self.A = {}
        self.spectral_function_exact = {}
        self.freqs = {}
        self.times = {}
        self.ri = {}            


    def set_ground_state_circuit(self, method = 'theory', backend = 'qasm_simulator', show = True, **kwargs):
        """ Creates the ground state preparation circuit according to the chosen method.
        The "theory" method first numerically computes ground state using scipy's functions and then sets up the circuit with qiskit's initialize function.
        The "VQE" method optimizes parameters of a two-local ansatz (Ry and CNOT gates).
        """
        
        self.ground_state_method = method
        
        if method == 'theory':
            
            # numerically computes ground state using scipy's solver
            # ground_energy_exact is float and ground_state_exact is 2**num_qubits sized np.array
            self.ground_energy_exact, self.ground_state_exact = classical_eigensolver(self) # assumes no degeneracies!
            
            # using qiskit's initalize function
            self.ground_state_circuit = initialize_GS(self)

        elif method == 'VQE':
            
            # params of vqe execution (sets to some default options if not specified)
            self.vqe_options = set_vqe_options(**kwargs)

            # runs vqe and returns circuit, array of params of circuit, state produced by that circuit (np.array), and energy of that state
            self.ground_state_circuit, self.optimal_VQE_params, self.ground_state_exp, self.energy_value_exp = set_vqe_circuit(self, backend)
        
        
        if show == True:

            print(self.ground_state_circuit.draw())



    def set_time_evolution_circuit(self, method = 'trotter', backend = 'statevector_simulator', random=True, show = False, **kwargs):
        """Builds time evolution circuit with "self.evo_time" as free parameter.
        The "trotter" method expands the evolution operator via trotterization.
            - Takes as options: "product_formula", "reps", "order"
        The "VFF" method trains a circuit to replicate the effect of the evolution operator on the initial state.
            - Takes as options: "site", "training_time", "n_eig", "reps", "num_shots"
        """

        # we may
        if 'evolution_hamiltonian' in kwargs:
            self.evo_H = kwargs['evolution_hamiltonian']
        else:
            self.evo_H = self.H

        # evolution time (denoted by 't' in the circuits) is a free parameter for now
        self.evo_time = Parameter('t')

        # two methods are allowed: "trotter" and "VFF", each with its options specified in kwargs
        self.time_evolution_method = method

        if method == 'trotter':

            # params of trotterization (type of formula, order, number of reps)
            self.trot_options = set_trot_options(**kwargs)

            # build trotterization circuit
            self.time_evolution_circuit = set_trotter_circuit(self)

        elif method == 'VFF': 

            assert self.ground_state_method != 'theory', 'VFF method does not accept initialize as ground state preparation scheme'

            # params of VFF (site with creation/annihilation operators, training time, number of eigenstates in initial state)
            self.vff_options = set_VFF_options(self, **kwargs)

            # VFF is trained against Trotter formula (single step Lie-Trotterby default)
            self.trot_options = set_trot_options(**kwargs)

            # build VFF circuit
            self.time_evolution_circuit, self.optimized_VFF_cost, self.optimal_VFF_params = set_VFF_circuit(self, backend, random=random)

        if show == True:

            print(self.time_evolution_circuit.draw())



    def local_green(self, site = 0,
                    show = True, **kwargs):

        # dictionary with keys ('X','X'), ('X','Y')...; and values the corresponding circuits

        self.local_green_circuit[site] = green_circuit_dict(self, site)

        if show == True:

            print('XX circuit: \n', self.local_green_circuit[site][('X','X')].draw())


    def set_fourier_parameters(self, site=0, **kwargs):

        """
        Sets the time and frequency parameters to be used for the frequency portion of the computation.
        `dw`: Equals $\Delta \omega_0 = \Delta \omega_1$.
        `w_max`: Equals $\Omega_0$.
        `t_max`: Equals $T_0$.
        `n_measurements`: Equals $m \leq n_0$.
        `dt`: Equals $\Delta t_1$.
        """

        if site not in self.fourier_parameters:
            self.fourier_parameters[site] = {'dw':None, 'w_max':None, 't_max':None, 'n_measurements':None, 'n0':None}

        self.fourier_parameters[site].update(kwargs)

        if self.fourier_parameters[site]['dw'] is None:
            self.fourier_parameters[site]['dw'] = 2*np.pi / self.fourier_parameters[site]['t_max']

        self.fourier_parameters[site]['w0'] = self.fourier_parameters[site]['dw'] * np.ceil(self.fourier_parameters[site]['w_max'] / self.fourier_parameters[site]['dw'] - 1e-6)

        ratio = 2*np.pi*self.fourier_parameters[site]['n0']/self.fourier_parameters[site]['dw']

        self.fourier_parameters[site]['t0'] = ratio / np.floor(ratio / self.fourier_parameters[site]['t_max'] +1e-6)

        #print(ratio / self.fourier_parameters[site]['t0'])
        self.fourier_parameters[site]['n1'] = int(np.around(ratio / self.fourier_parameters[site]['t0'], 0))
        self.fourier_parameters[site]['dt'] = self.fourier_parameters[site]['t0'] / self.fourier_parameters[site]['n0']
        self.fourier_parameters[site]['center_point'] = int(self.fourier_parameters[site]['n1'] // 2) # equals to previous variable `h`
        self.fourier_parameters[site]['nonzero_range'] = int(np.around(self.fourier_parameters[site]['w0']/self.fourier_parameters[site]['dw'], 0))

        self.fourier_parameters[site]['w1'] = np.pi/self.fourier_parameters[site]['dt']
        self.fourier_parameters[site]['t1'] = 2*np.pi/self.fourier_parameters[site]['dw']

        self.freqs[site] = np.linspace(-self.fourier_parameters[site]['w_max'], self.fourier_parameters[site]['w_max'], 2*self.fourier_parameters[site]['nonzero_range'], endpoint=False)

        tmax = self.fourier_parameters[site]['t_max']
        n_times = self.fourier_parameters[site]['n_measurements']
        n0 = self.fourier_parameters[site]['n0']

        if n_times < n0:
            # Chooses some points from the whole grid.
            times = np.linspace(0, tmax, n0, endpoint=False) # endpoint=False removes the 1 term from the calculations
            self.ri[site] = np.sort(np.random.choice(n0, n_times, replace=False))
            self.times[site] = times[self.ri[site]]
        elif n_times == n0:
            self.ri[site] = np.arange(n_times)
            self.times[site] = np.linspace(0, tmax, n_times)

        if self.fourier_parameters[site]['w1'] < self.fourier_parameters[site]['w0']:
            print("Invalid parameters. Omega_1 (={}) cannot be lower than Omega_0 (={}).".format(self.fourier_parameters[site]['w1'], self.fourier_parameters[site]['w0']),
                "Your time samples are probably too sparse for the Fourier transform to pick higher frequencies.",
                "Increase n0 while keeping the same number of n_measurements.")
            raise ValueError()

        elif self.fourier_parameters[site]['t1'] < self.fourier_parameters[site]['t0']:
            print("Strange choice of parameters. T_1 (={}) should not be lower than T_0 (={}).".format(self.fourier_parameters[site]['t1'], self.fourier_parameters[site]['t0']),
                "You are measuring high time values which should have no effect on the frequencies you wish to compute.",
                "If you have a uniform sample, you can apply the standard Fourier transform.",
                "If you have a non-uniform sample, there should be theoretical guarantees that L1 minimization with compressed sensing will find the right solution.")


    def run_green_circuits(self, site=0, method='theory', shots_per_t=1024, backend='qasm_simulator'):
        """
        Runs the 4 circuits previously built, in order to compute an array of values for G(t).
        """

        self.green_function_time[site] = compute_green_function_time(self, site=site, shots_per_t=shots_per_t, backend_name=backend)

        ReImPlot(self.times[site], self.green_function_time[site], x='$t$', y='$G(t)$', method='normal')


    def get_green_time_exact(self, site=0, delta=1e-4):
        """
        Computes G(t) exactly.
        """

        eigensystem = sc.linalg.eigh(self.H.to_matrix())
        energies = eigensystem[0]

        Z_array = np.array([1.])
        for n in range(site):
            Z_array = np.kron(np.array([[1,0],[0,-1]]), Z_array)

        op = np.kron(np.kron(np.eye(2**(self.num_qubits-1-site)), np.array([[0,0],[1,0]])), Z_array)
        opd = np.kron(np.kron(np.eye(2**(self.num_qubits-1-site)), np.array([[0,1],[0,0]])), Z_array)

        states = eigensystem[1].T
        En0 = energies[0]
        state0 = states[0].reshape((1,-1)).conj()

        f1, f2 = np.abs((state0@op@eigensystem[1]))**2, np.abs((state0@opd@eigensystem[1]))**2

        self.energies, self.En0, self.f1, self.f2 = energies, En0, f1, f2

        def Gt(t):
            res = -1j * np.sum(f1 * np.exp(-1j * (energies-En0)*t) + f2 * np.exp(-1j * (En0-energies) *t))
            return res

        self.green_time_exact[site] = np.array([Gt(t) for t in self.times[site]])

    def get_spectral_function_exact(self, site=0, delta=1e-4):
        """
        Computes the Spectral Function exactly.
        """

        try:
            f1, f2 = self.f1, self.f2
            energies, En0 = self.energies, self.En0
        except:
            self.get_green_time_exact(site=site, delta=delta)
            f1, f2 = self.f1, self.f2
            energies, En0 = self.energies, self.En0

        def Delta(x, a = 1e-1):
            return np.exp(-(x/a)**2)/(np.abs(a)*np.sqrt(np.pi))

        def Aw(w):
            res = np.sum(f1 * Delta(w+energies-En0) + f2 *Delta(w+En0-energies))
            return res

        self.spectral_function_exact[site] = np.array([Aw(w) for w in self.freqs[site]])




    def convert_to_spectral_function(self, site=0, method='L1', symmetric=False, force_sym=True, regularization=False, l=0.01, eps=1., show=False):

        """
        Computes an array corresponding to G(omega), and the Spectral Function A(omega).
        """

        self.green_function_freq[site], self.A[site] = compute_spectral_function(self, 
                                                                        site=site, 
                                                                        method=method, 
                                                                        symmetric=symmetric, 
                                                                        force_sym=force_sym, 
                                                                        regularization=regularization, 
                                                                        l=l, 
                                                                        eps=eps, 
                                                                        show=show)            


    def plot_spectral_function(self, site=0, save=False, also_green_freq = False):
        """
        Plots Spectral Function previously computed.
        """

        if also_green_freq:
            ReImPlot(self.freqs[site], self.green_function_freq[site], x=r'$\omega$', y=r'$G(\omega)$', method='normal')
        ReImPlot(self.freqs[site], [self.A[site](x) for x in self.freqs[site]], x=r'$\omega$', y=r'$A(\omega)$', method='noimag',save=save)


    def run_full_algorithm(self, site=0, backend='qasm_simulator', show=False):

        self.set_ground_state_circuit(method = 'theory', backend = backend, show = show, **kwargs)

        self.set_time_evolution_circuit(method = 'trotter', backend = backend, show = show, **kwargs)

        self.local_green(site=site, show = show, **kwargs)

        self.set_fourier_parameters(site=site, **kwargs)

        self.run_green_circuits(site=site, shots_per_t=1024, backend=backend)

        self.convert_to_spectral_function(site=site, method='L1', symmetric=False, force_sym=True, regularization=False, l=0.01, eps=1., show=show)

        self.plot_spectral_function(site=site)


def ReImPlot(xL, yL, x='x', y='y', method='normal', save=False):
    """
    Function to automatically plot a complex array, and save the resulting plot as a PDF.
    """
    fig = plt.figure()
    yL_real = np.real(yL)
    yL_imag = np.imag(yL)
    plt.autoscale(tight=True)
    if method == 'scatter':
        plt.scatter(xL, yL_real, label='Re', marker='.')
        plt.scatter(xL, yL_imag, label='Im', marker='.')
        plt.legend()
    elif method == 'noimag':
        plt.plot(xL, yL_real)
    else:
        plt.plot(xL, yL_real, label='Re')
        plt.plot(xL, yL_imag, label='Im')
        plt.legend()
    plt.ylabel(y)
    plt.xlabel(x)
    plt.show()
    if save:
        time = datetime.strftime(datetime.now(), '%YY%mM%dD%Hh%Mm%Ss')
        filename = time+'.pdf'
        fig.savefig(filename, bbox_inches='tight')
