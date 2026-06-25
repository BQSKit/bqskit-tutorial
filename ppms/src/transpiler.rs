// ── All your existing types and functions, now pub ────────────────────────────
use std::fs::File;
use std::io::{self, BufWriter, Write};

use crate::tableau::{Gate1Q, PauliString, Tableau};

// Keep all your existing structs/enums/fns here, just make the ones
// needed by transpile.rs and Python pub:

pub use self::core::{transpile, write_trans, make_z_pauli};
use crate::qasm::QasmGate;
use crate::types::{TransItem, TransPauli, TransClifford, CouplingRestraint};


mod core {
    use super::*;

    // ─────────────────────────────────────────────────────────────────────────────
    // Single-qubit Clifford optimizer
    // ─────────────────────────────────────────────────────────────────────────────
    //
    // Each of the 24 single-qubit Cliffords is uniquely identified by the pair
    // (X-image, Z-image) of its conjugation action on the Pauli group, where each
    // image is one of the 6 signed non-identity Paulis {±X, ±Y, ±Z}.
    //
    // We encode a signed Pauli as a u8:
    //   bits [1:0] = Pauli type: 0=X, 1=Y, 2=Z
    //   bit  [2]   = sign: 0=+, 1=−
    //
    // A sequence of single-qubit gates is compressed by composing their actions
    // into a single (x_img, z_img) state, then looking up the minimal gate
    // sequence that produces that state from a precomputed table of all 24
    // elements of the single-qubit Clifford group.
    //
    // Minimal sequences use only {S, Sdg, SX, SXdg, X, Z}.  Of these, X and Z
    // are not emitted in the .trans format and are silently dropped.

    /// Encode a signed single-qubit Pauli as a u8.
    /// Pauli type: 0=X, 1=Y, 2=Z.  Sign bit 2: 0=+, 1=−.
    #[inline]
    fn pauli_code(pauli: u8, neg: bool) -> u8 {
        pauli | ((neg as u8) << 2)
    }

    /// Apply a single-qubit gate to a Clifford state `(x_img, z_img)` and return
    /// the updated state.  Each image is a [`pauli_code`]-encoded signed Pauli
    /// representing where the gate sends X (resp. Z) under conjugation.
    fn apply_1q_to_state(gate: Gate1Q, x_img: u8, z_img: u8) -> (u8, u8) {
        // Conjugation rules (P → G·P·G†) for each gate on the three Paulis:
        //   H:    X→Z,   Y→-Y,  Z→X
        //   S:    X→Y,   Y→-X,  Z→Z
        //   Sdg:  X→-Y,  Y→X,   Z→Z
        //   SX:   X→X,   Y→-Z,  Z→-Y
        //   SXdg: X→X,   Y→Z,   Z→Y
        //   X:    X→X,   Y→-Y,  Z→-Z
        //   Y:    X→-X,  Y→Y,   Z→-Z
        //   Z:    X→-X,  Y→-Y,  Z→Z
        //
        // For a composed state (x_img, z_img), prepending gate G updates each
        // image independently: new_x_img = G(x_img), new_z_img = G(z_img).
        fn apply_gate_to_pauli(gate: Gate1Q, p: u8) -> u8 {
            let pauli = p & 0x3;
            let neg = (p >> 2) & 1 != 0;
            let (new_pauli, extra_neg) = match gate {
                Gate1Q::H => match pauli {
                    0 => (2, false), // X→Z
                    1 => (1, true),  // Y→-Y
                    2 => (0, false), // Z→X
                    _ => unreachable!(),
                },
                Gate1Q::S => match pauli {
                    0 => (1, false), // X→Y
                    1 => (0, true),  // Y→-X
                    2 => (2, false), // Z→Z
                    _ => unreachable!(),
                },
                Gate1Q::Sdg => match pauli {
                    0 => (1, true),  // X→-Y
                    1 => (0, false), // Y→X
                    2 => (2, false), // Z→Z
                    _ => unreachable!(),
                },
                Gate1Q::SX => match pauli {
                    0 => (0, false), // X→X
                    1 => (2, true),  // Y→-Z
                    2 => (1, true),  // Z→-Y
                    _ => unreachable!(),
                },
                Gate1Q::SXdg => match pauli {
                    0 => (0, false), // X→X
                    1 => (2, false), // Y→Z
                    2 => (1, false), // Z→Y
                    _ => unreachable!(),
                },
                Gate1Q::X => match pauli {
                    0 => (0, false), // X→X
                    1 => (1, true),  // Y→-Y
                    2 => (2, true),  // Z→-Z
                    _ => unreachable!(),
                },
                Gate1Q::Y => match pauli {
                    0 => (0, true),  // X→-X
                    1 => (1, false), // Y→Y
                    2 => (2, true),  // Z→-Z
                    _ => unreachable!(),
                },
                Gate1Q::Z => match pauli {
                    0 => (0, true),  // X→-X
                    1 => (1, true),  // Y→-Y
                    2 => (2, false), // Z→Z
                    _ => unreachable!(),
                },
            };
            pauli_code(new_pauli, neg ^ extra_neg)
        }
        (apply_gate_to_pauli(gate, x_img), apply_gate_to_pauli(gate, z_img))
    }

    /// Compose a sequence of single-qubit gates into a single Clifford state
    /// `(x_img, z_img)` by applying each gate in order to the identity state
    /// `(X+, Z+)`.
    fn simulate_gate_sequence(gates: &[Gate1Q]) -> (u8, u8) {
        let mut x_img = pauli_code(0, false); // X+
        let mut z_img = pauli_code(2, false); // Z+
        for &g in gates {
            let (nx, nz) = apply_1q_to_state(g, x_img, z_img);
            x_img = nx;
            z_img = nz;
        }
        (x_img, z_img)
    }

    /// Build a lookup table mapping each of the 24 single-qubit Clifford states
    /// `(x_img, z_img)` to a minimal gate sequence that produces it.
    ///
    /// The 24 sequences cover all elements of the single-qubit Clifford group,
    /// using only gates from {I, S, Sdg, SX, SXdg, X, Z} (at most 4 gates each).
    /// The table is constructed by simulating each sequence and recording the
    /// resulting state; no two sequences produce the same state.
    pub fn build_clifford_table() -> std::collections::HashMap<(u8, u8), Vec<Gate1Q>> {
        use Gate1Q::*;
        let sequences: &[&[Gate1Q]] = &[
            &[],             // I
            &[X],            // X
            &[X, Z],         // X·Z
            &[Z],            // Z
            &[S, SX, S],     // S·SX·S  (= H up to global phase)
            &[S],            // S
            &[Sdg],          // Sdg
            &[S, X],         // S·X
            &[Sdg, X],       // Sdg·X
            &[Z, SX, S],     // Z·SX·S
            &[SX, S],        // SX·S
            &[S, SX, Z],     // S·SX·Z
            &[Z, SX, Z],     // Z·SX·Z
            &[Z, SX, Z, X],  // Z·SX·Z·X
            &[Z, SXdg],      // Z·SXdg
            &[Z, SX],        // Z·SX
            &[Sdg, SX, S],   // Sdg·SX·S
            &[Sdg, SX, Sdg], // Sdg·SX·Sdg
            &[S, SX, Sdg],   // S·SX·Sdg
            &[S, SXdg, Z],   // S·SXdg·Z
            &[S, SXdg],      // S·SXdg
            &[S, SX],        // S·SX
            &[SX, Sdg],      // SX·Sdg
            &[Z, SX, Sdg],   // Z·SX·Sdg
        ];
        let mut table = std::collections::HashMap::new();
        for seq in sequences {
            let state = simulate_gate_sequence(seq);
            table.insert(state, seq.to_vec());
        }
        table
    }

    /// Reduce a sequence of single-qubit Clifford gates on one qubit to the
    /// minimal equivalent sequence.  Returns an empty vec for the identity.
    ///
    /// The lookup table is built once per thread and cached for subsequent calls.
    fn optimize_single_qubit_sequence(gates: &[Gate1Q]) -> Vec<Gate1Q> {
        if gates.is_empty() {
            return vec![];
        }
        let state = simulate_gate_sequence(gates);
        use std::cell::RefCell;
        thread_local! {
            static TABLE: RefCell<Option<std::collections::HashMap<(u8, u8), Vec<Gate1Q>>>> =
                RefCell::new(None);
        }
        TABLE.with(|t| {
            let mut borrow = t.borrow_mut();
            if borrow.is_none() {
                *borrow = Some(build_clifford_table());
            }
            borrow.as_ref().unwrap().get(&state).cloned().unwrap_or_default()
        })
    }

    /// Reduce a sequence of Clifford gates to a minimal equivalent sequence and
    /// return the result as [`TransClifford`] items ready for the .trans output.
    ///
    /// Single-qubit gates on each qubit are accumulated independently and
    /// compressed into a minimal sequence using the 24-element Clifford group
    /// lookup.  When a 2-qubit gate is encountered, the pending single-qubit
    /// sequences on its two qubits are flushed and optimized first; the 2-qubit
    /// gate is then emitted unchanged.  Any remaining single-qubit sequences are
    /// flushed at the end.
    ///
    /// Only CX, S, Sdg, SX, and SXdg are emitted; X and Z are Pauli corrections
    /// that do not need to be scheduled and are omitted from the output.
    fn optimize_clifford_sequence(gates: &[QasmGate], n_qubits: usize) -> Vec<TransClifford> {
        let mut per_qubit: Vec<Vec<Gate1Q>> = vec![Vec::new(); n_qubits];
        let mut result: Vec<TransClifford> = Vec::new();

        fn flush_qubit(
            qubit: usize, per_qubit: &mut Vec<Vec<Gate1Q>>, result: &mut Vec<TransClifford>,
            n_qubits: usize,
        ) {
            let seq = std::mem::take(&mut per_qubit[qubit]);
            if seq.is_empty() {
                return;
            }
            for g in optimize_single_qubit_sequence(&seq) {
                let qasm_gate = QasmGate::Clifford1Q { gate: g, qubit };
                if let Some(tc) = TransClifford::from_qasm_gate(&qasm_gate, n_qubits) {
                    result.push(tc);
                }
            }
        }

        for gate in gates {
            match gate {
                QasmGate::Clifford1Q { gate: g, qubit } => {
                    per_qubit[*qubit].push(*g);
                }
                QasmGate::Clifford2Q { gate: g2, control, target } => {
                    flush_qubit(*control, &mut per_qubit, &mut result, n_qubits);
                    flush_qubit(*target, &mut per_qubit, &mut result, n_qubits);
                    let qasm_gate =
                        QasmGate::Clifford2Q { gate: *g2, control: *control, target: *target };
                    if let Some(tc) = TransClifford::from_qasm_gate(&qasm_gate, n_qubits) {
                        result.push(tc);
                    }
                }
                _ => {}
            }
        }

        for qubit in 0..n_qubits {
            flush_qubit(qubit, &mut per_qubit, &mut result, n_qubits);
        }

        result
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Transpiler
    // ─────────────────────────────────────────────────────────────────────────────

    pub fn transpile(n_qubits: usize, gates: &[QasmGate], restraint: &CouplingRestraint) -> Vec<TransItem> {
        let mut tableau = Tableau::new(n_qubits);
        let mut ops: Vec<TransItem> = Vec::new();
        let mut clifford_queue: Vec<QasmGate> = Vec::new();

        let has_t = gates.iter().any(|g| matches!(g, QasmGate::T { .. } | QasmGate::Tdg { .. }));
        if !has_t {
            eprintln!("Warning: circuit has no T gates");
        }

        // Append Z-basis measurements for any qubit not already measured.
        let mut gates_with_measurements = gates.to_vec();
        let measured: Vec<bool> = {
            let mut m = vec![false; n_qubits];
            for g in gates {
                if let QasmGate::Measure { qubit } = g {
                    m[*qubit] = true;
                }
            }
            m
        };
        for (q, &has_m) in measured.iter().enumerate() {
            if !has_m {
                gates_with_measurements.push(QasmGate::Measure { qubit: q });
            }
        }

        fn flush_clifford_queue(
            clifford_queue: &mut Vec<QasmGate>, ops: &mut Vec<TransItem>, tableau: &mut Tableau,
            n_qubits: usize,
        ) {
            let optimized = optimize_clifford_sequence(clifford_queue, n_qubits);
            for c in optimized {
                ops.push(TransItem::Clifford(c));
            }
            clifford_queue.clear();
            *tableau = Tableau::new(n_qubits);
        }

        for gate in &gates_with_measurements {
            match gate {
                QasmGate::Clifford1Q { gate: g, qubit } => {
                    clifford_queue.push(gate.clone());
                    tableau.prepend_1q_correct(*g, *qubit);
                }
                QasmGate::Clifford2Q { gate: g, control, target } => {
                    clifford_queue.push(gate.clone());
                    tableau.prepend_2q(*g, *control, *target);
                }
                QasmGate::T { qubit } => {
                    let pre_pauli = make_z_pauli(n_qubits, *qubit, false);
                    let conjugated = tableau.conjugate(&pre_pauli);
                    let compatible = restraint.accepts_pauli(&conjugated);
                    if !compatible {
                        flush_clifford_queue(&mut clifford_queue, &mut ops, &mut tableau, n_qubits);
                        let fresh_pauli = make_z_pauli(n_qubits, *qubit, false);
                        ops.push(TransItem::Pauli(TransPauli::from_pauli_string(&fresh_pauli, "T")));
                    } else {
                        ops.push(TransItem::Pauli(TransPauli::from_pauli_string(&conjugated, "T")));
                    }
                }
                QasmGate::Tdg { qubit } => {
                    // Tdg uses a negative-sign Z Pauli (−Z rotation).
                    let pre_pauli = make_z_pauli(n_qubits, *qubit, true);
                    let conjugated = tableau.conjugate(&pre_pauli);
                    let compatible = restraint.accepts_pauli(&conjugated);
                    if !compatible {
                        flush_clifford_queue(&mut clifford_queue, &mut ops, &mut tableau, n_qubits);
                        let fresh_pauli = make_z_pauli(n_qubits, *qubit, true);
                        ops.push(TransItem::Pauli(TransPauli::from_pauli_string(&fresh_pauli, "T")));
                    } else {
                        ops.push(TransItem::Pauli(TransPauli::from_pauli_string(&conjugated, "T")));
                    }
                }
                QasmGate::Measure { qubit } => {
                    let pre_pauli = make_z_pauli(n_qubits, *qubit, false);
                    let conjugated = tableau.conjugate(&pre_pauli);
                    let compatible = restraint.accepts_pauli(&conjugated);
                    if !compatible {
                        flush_clifford_queue(&mut clifford_queue, &mut ops, &mut tableau, n_qubits);
                        let fresh_pauli = make_z_pauli(n_qubits, *qubit, false);
                        ops.push(TransItem::Pauli(TransPauli::from_pauli_string(&fresh_pauli, "M")));
                    } else {
                        ops.push(TransItem::Pauli(TransPauli::from_pauli_string(&conjugated, "M")));
                    }
                }
                QasmGate::Barrier => {}
            }
        }

        ops
    }

    pub fn make_z_pauli(n: usize, q: usize, negative: bool) -> PauliString {
        let mut ps = PauliString::identity(n);
        ps.z_bits[q] = true;
        ps.sign = negative;
        ps
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Output writer
    // ─────────────────────────────────────────────────────────────────────────────

    pub fn write_trans(output_path: &str, items: &[TransItem]) -> io::Result<(usize, usize)> {
        let file = File::create(output_path)?;
        let mut writer = BufWriter::new(file);
        let mut n_ts = 0usize;
        let mut n_cliffords = 0usize;

        for item in items {
            match item {
                TransItem::Pauli(p) => {
                    writeln!(writer, "{}", p)?;
                    if p.label == "T" {
                        n_ts += 1;
                    }
                }
                TransItem::Clifford(c) => {
                    writeln!(writer, "{}", c)?;
                    n_cliffords += 1;
                }
            }
        }

        Ok((n_ts, n_cliffords))
    }
}

pub mod python {
    use super::{transpile};
    use crate::qasm::parse_qasm_str;  // new — parses from a string instead of a file
    use crate::types::{TransItem};
    use crate::types::{CouplingRestraint};
    use pyo3::prelude::*;
    use pyo3::exceptions::PyIOError;
    use std::fmt;

    // ── Python-facing TransItem ───────────────────────────────────────────────

    #[pyclass(name = "TransItem")]
    #[derive(Clone)]
    pub struct PyTransItem {
        #[pyo3(get)]
        pub is_pauli: bool,
        #[pyo3(get)]
        pub is_clifford: bool,
        #[pyo3(get)]
        pub label: String,   // "T", "M", "S", "CX", etc.
        #[pyo3(get)]
        pub sign: String,    // "+" or "-"
        #[pyo3(get)]
        pub ops: Vec<char>,  // per-qubit Pauli ops
    }

    #[pymethods]
    impl PyTransItem {
        fn __repr__(&self) -> String {
            let ops: String = self.ops.iter().collect();
            format!("TransItem(label='{}', sign='{}', ops='{}')", self.label, self.sign, ops)
        }
    }

    impl From<&TransItem> for PyTransItem {
        fn from(item: &TransItem) -> Self {
            match item {
                TransItem::Pauli(p) => PyTransItem {
                    is_pauli: true,
                    is_clifford: false,
                    label: p.label.clone(),
                    sign: format!("{}", p.sign),
                    ops: p.ops.clone(),
                },
                TransItem::Clifford(c) => PyTransItem {
                    is_pauli: false,
                    is_clifford: true,
                    label: c.name.clone(),
                    sign: "+".to_string(),
                    ops: c.ops.clone(),
                },
            }
        }
    }

    impl fmt::Display for PyTransItem {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.sign)?;
            for &c in &self.ops {
                if c == 'I' { write!(f, "_")?; } else { write!(f, "{}", c)?; }
            }
            write!(f, "<{}>", self.label)
        }
    }

    // ── Shared helpers ────────────────────────────────────────────────────────

    fn do_transpile(
        qasm: &str,
        restraint: &CouplingRestraint,
    ) -> PyResult<Vec<Vec<PyTransItem>>> {
        let (n_qubits, gates) = parse_qasm_str(qasm)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        let items = transpile(n_qubits, &gates, restraint);

        // Group into layers: each contiguous run of Paulis is one layer,
        // Cliffords between them are their own single-item layers.
        let mut layers: Vec<Vec<PyTransItem>> = Vec::new();
        let mut current_layer: Vec<PyTransItem> = Vec::new();

        for item in &items {
            match item {
                TransItem::Pauli(_) => {
                    current_layer.push(PyTransItem::from(item));
                }
                TransItem::Clifford(_) => {
                    if !current_layer.is_empty() {
                        layers.push(std::mem::take(&mut current_layer));
                    }
                    layers.push(vec![PyTransItem::from(item)]);
                }
            }
        }
        if !current_layer.is_empty() {
            layers.push(current_layer);
        }

        Ok(layers)
    }

    // ── Exposed functions ─────────────────────────────────────────────────────

    /// Transpile a QASM string using a max Pauli weight restraint.
    #[pyfunction]
    #[pyo3(signature = (input_qasm, max_width=None))]
    pub fn transpile_circuit(
        input_qasm: &str,
        max_width: Option<i32>,
    ) -> PyResult<Vec<Vec<PyTransItem>>> {
        let restraint = match max_width {
            Some(w) if w > 0 => CouplingRestraint::MaxWidth(w as usize),
            _                => CouplingRestraint::unrestricted(),
        };
        do_transpile(input_qasm, &restraint)
    }


    pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(transpile_circuit, m)?)?;
        m.add_class::<PyTransItem>()?;
        Ok(())
    }
}