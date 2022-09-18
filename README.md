# Berkeley Quantum Synthesis Toolkit Tutorial

The Berkeley Quantum Synthesis Toolkit (BQSKit) is a powerful and portable
quantum compiler framework. This tutorial series first explores how to use
BQSKit to compile quantum programs to efficient physical circuits for any QPU.
Future parts wiil dive deeper into BQSKit's advanced features teaching users
how to design compiler workflows for their specific use cases.

## Installation and Setup

This tutorial requires a recent version of [Python](https://www.python.org/)
(3.8, 3.9, or 3.10) to be installed alongside the following python packages:

- BQSKit
- Qiskit
- SciPy
- Jupyter
- OpenFermion

To install Python, visit their [download page](https://www.python.org/downloads/)
and follow the instructions for your system.
The additional packages install easily on most platforms using Python's `pip` and `venv`,
with the exception of Macbooks with an M1 chip, where Anaconda
(https://anaconda.org/) should be used to be able to install BQSKit.
Since the download of Anaconda is quite large, and may trigger the subsequent
download and install of Apple's Rosetta if that was not already enabled, it is
highly recommended to pre-install these packages before the tutorial.

The following lists the platform-specific instructions to setup Python and a
virtual environment, as well as to install all packages. (Upgrading pip is
always recommended, but may not be necessary.)

### Linux and Macs with an Intel chip

Create and activate a virtual environment for Python, then install:

```sh
$ python3 -m venv TUTORIAL
$ source TUTORIAL/bin/activate
(TUTORIAL) $ python -m pip install --upgrade pip
(TUTORIAL) $ python -m pip install jupyter openfermion
(TUTORIAL) $ python -m pip install --pre 'bqskit[ext]'
```

### Macs with an M1 chip

Install [Anaconda](https://www.anaconda.com/products/individual) for Intel 64b
(ignore any warnings about the M1 not being a 64b platform). Follow the steps
to setup the conda environment for your shell (run in the Terminal app). Then
create a new conda project named "TUTORIAL":

```sh
(base) $ conda create -n TUTORIAL python=3.10
(TUTORIAL) $ conda activate TUTORIAL
(TUTORIAL) $ python -m pip install --upgrade pip
(TUTORIAL) $ python -m pip install jupyter openfermion
(TUTORIAL) $ python -m pip install --pre 'bqskit[ext]'
```

If the system asks to install Apple's Rosetta, accept the install.
The reason for requiring an Intel install is that BQSKit's dependencies
are not all yet natively supported on M1 chips.
There are also still outstanding problems with installing SciPy and NumPy from
PyPI on M1. Although these can easily be resolved by installing through a Mac
packager (such as MacPorts or Fink) instead, use of Anaconda will side-step
these installation issues as well.

### Windows

Install [Python](https://www.python.org/downloads/windows/) if not already
available on your system. On a command prompt, create and setup a virtual
environment, then install:

```sh
$ python3 -m venv TUTORIAL
$ TUTORIAL\Scripts\activate
(TUTORIAL) $ python -m pip install --upgrade pip
(TUTORIAL) $ python -m pip install jupyter openfermion
(TUTORIAL) $ python -m pip install --pre 'bqskit[ext]'
```
