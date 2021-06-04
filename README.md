# QuGreen: Computing Spectral Functions using Quantum Computing
## Spectral Functions with Qiskit

The Spectral Function is a central quantity in the study of quantum many-body systems, but it can be hard to compute. Novel methods, taking advantage of the power of quantum computers, can do this calculation more efficiently. 

However, most of these methods are either restricted to a very small number of qubits [1],[2], so not scalable, or are not suited to NISQ computers [3],[4], requiring tiny error-rates.

This package combines different techniques to calculate the Spectral Function of systems with up to 30 qubits using IBM’s quantum computers. This constitutes a new open-source tool to enable the fast calculation of Spectral Functions for small to medium-sized systems.

This package consists of a Class `QuGreen` with several helpful methods (shown in `Tutorial.ipynb`). These methods call other functions, which can be found in the `qgfunctions` folder.


## Installation

To use this package, create an Anaconda environment (just for safety):

```
conda create -n QuGreen python=3
conda activate QuGreen
```

Install Qiskit:
```
pip install qiskit[visualization]
```
You also need to install [CVXPY](https://www.cvxpy.org/install/).

If you wish to use method `SL0`, also run
```
pip install pyCSalgos
```

## How-To

Check notebook `Tutorial.ipynb` for a simple tutorial.

## References
