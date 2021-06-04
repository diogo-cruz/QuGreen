# QuGreen: Computing Spectral Functions using Quantum Computing

The Spectral Function is a central quantity in the study of quantum many-body systems, but it can be hard to compute. Novel methods, taking advantage of the power of quantum computers, can do this calculation more efficiently. 

However, most of these methods are either restricted to a very small number of qubits [1],[2], so not scalable, or are not suited to NISQ computers [3],[4], requiring tiny error-rates.

We will combine different techniques to calculate the Spectral Function of systems with up to 30 qubits using IBM’s quantum computers. This will constitute a new open-source tool to enable the fast calculation of Spectral Functions for small to medium-sized systems. 


# Installation

To run this package, run:

```
pip install qiskit[visualization], numpy, scipy, cvxpy, matplotlib, datetime
```

If you wish to use method `SL0`, also run
```
pip install pyCSalgos
```

# How to

Check notebook `Tutorial.ipynb` for a simple tutorial.