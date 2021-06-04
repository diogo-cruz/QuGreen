from qiskit.aqua.operators import PauliTrotterEvolution, Suzuki
from qiskit.circuit import Parameter

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

def trotter_circuit(self):
    '''Creates time evolution circuit with time as free parameter 
    according the method specified by self.trot_options'''
    
    evo_H = self.evo_H                    # this may be different from self.H
    evo_time = Parameter('t')             # evolution time, to be set later
    self.evo_time = evo_time
    evo_op   = (evo_time * evo_H).exp_i() # evolution operator
    
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

def set_evo_time(self, time):
    
    qc = self.time_evolution_circuit          # circuit with free 't' parameter
    qc.bind_parameters({self.evo_time: time}) # set 't' parameter to time
        
    return qc