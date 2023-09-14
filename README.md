# Berkeley Quantum Synthesis Toolkit Tutorial

The Berkeley Quantum Synthesis Toolkit (BQSKit) is a powerful and portable
quantum compiler framework. This tutorial series first explores how to use
BQSKit to compile quantum programs to efficient physical circuits for any
QPU. Future parts will dive deeper into BQSKit's advanced features, teaching
users how to design compiler workflows for their specific use cases.

## Installation and Setup

This tutorial requires a recent version of [Python](https://www.python.org/)
(3.8+) to be installed alongside the following Python packages:

- BQSKit
- Qiskit
- SciPy
- Jupyter
- OpenFermion

To install Python, visit their [download page](https://www.python.org/downloads/)
and follow the instructions for your system.
The additional packages install easily on most platforms using Python's `pip`
and `venv`:

The following lists the platform-specific instructions to set up Python
with a virtual environment and install all packages. (Upgrading pip is
always recommended but may not be necessary.)

### Linux and Macs

Create and activate a virtual environment for Python, then install:

```sh
$ python3 -m venv TUTORIAL
$ source TUTORIAL/bin/activate
(TUTORIAL) $ python -m pip install --upgrade pip
(TUTORIAL) $ python -m pip install bqskit qiskit scipy jupyter openfermion
```

### Windows

Install [Python](https://www.python.org/downloads/windows/) if not already
available on your system. On a command prompt, create and setup a virtual
environment, then install:

```sh
$ python3 -m venv TUTORIAL
$ TUTORIAL\Scripts\activate
(TUTORIAL) $ python -m pip install --upgrade pip
(TUTORIAL) $ python -m pip install bqskit qiskit scipy jupyter openfermion
```
