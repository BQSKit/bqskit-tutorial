# Running `ft_workflows.ipynb`

This notebook needs more than the base tutorial: two Rust extensions are built
from source, and `bqskit-ft` must come from a branch rather than PyPI.

## Requirements
| --- | --- |
| **Python 3.12+** | `bqskit-ft` sets `requires-python = ">=3.12"` |
| **Rust toolchain** + **maturin** | `cyclosynth`, `rsgridsynth` and `ppms/` are Rust extensions |
| **GMP** and **MPFR** | `cyclosynth` links them through `gmp-mpfr-sys` |

> `bqskit-ft` on PyPI does **not** contain `DPFPass` or `ConvertFractionalRZPass`.
> Install it from the `algorithmic_error_dpf` branch as shown below, or the
> notebook will fail at the Dyadic Phase Fixing section.

---

## macOS

```sh
brew install python@3.12 gmp mpfr
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

cd bqskit-tutorial
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install maturin
python -m pip install bqskit "qiskit<2" scipy jupyter openfermion
```

Then run **[Shared steps](#shared-steps)** below.

## Linux (Debian / Ubuntu)

```sh
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv build-essential \
                        libgmp-dev libmpfr-dev git curl
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

cd bqskit-tutorial
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install maturin
python -m pip install bqskit "qiskit<2" scipy jupyter openfermion
```

Then run **[Shared steps](#shared-steps)** below.

## Windows

**Use WSL2 and follow the Linux instructions above.** `cyclosynth` depends on
`gmp-mpfr-sys`, which builds GMP and MPFR from source and requires a Unix-like
build environment — it does not build against the MSVC toolchain.

```powershell
wsl --install -d Ubuntu
```

Then open the Ubuntu shell and follow the Linux section.

<details>
<summary>Native Windows (not recommended, requires MSYS2/mingw)</summary>

`gmp-mpfr-sys` needs the mingw64 toolchain from [MSYS2](https://www.msys2.org/),
and Rust must target `x86_64-pc-windows-gnu` rather than the default MSVC target.
Inside an **MSYS2 MinGW64** shell:

```sh
pacman -S --needed mingw-w64-x86_64-gcc mingw-w64-x86_64-gmp \
                   mingw-w64-x86_64-mpfr make diffutils m4 git
rustup toolchain install stable-x86_64-pc-windows-gnu
rustup default stable-x86_64-pc-windows-gnu
```

Create the virtual environment from a normal PowerShell prompt:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install maturin
python -m pip install bqskit "qiskit<2" scipy jupyter openfermion
```

Run the two `maturin develop --release` steps from the MSYS2 MinGW64 shell with
the venv's `VIRTUAL_ENV` exported, and everything else from PowerShell.

This path is untested — if it fights you, use WSL2.
</details>

---

## Shared steps

With the virtual environment **activated** and the working directory set to the
repository root:

**1. `bqskit-ft` from the `algorithmic_error_dpf` branch**

```sh
git clone --branch algorithmic_error_dpf https://github.com/BQSKit/bqskit-ft.git
python -m pip install ./bqskit-ft
```

This also pulls `rsgridsynth` from git and compiles it (a few minutes).

**2. `cyclosynth`**

```sh
git clone https://github.com/mtweiden/cyclosynth.git
cd cyclosynth
maturin develop --release
cd ..
```

The first build takes roughly 10 minutes — `gmp-mpfr-sys` compiles GMP and
MPFR from source.

**3. The bundled PPM transpiler**

```sh
cd ppms
maturin develop --release
cd ..
```

**4. Launch**

```sh
jupyter lab ft_workflows.ipynb
```

Run it from the **repository root** — `cyclopass/` and `ppms/` are imported as
local packages, so a different working directory gives `ModuleNotFoundError`.

To run it headlessly instead:

```sh
jupyter nbconvert --to notebook --execute ft_workflows.ipynb --output executed.ipynb
```
---

## Checking the install

```sh
python -c "import bqskit, bqskit.ft, cyclosynth, rsgridsynth, ppm_transpiler; print('ok')"
```
