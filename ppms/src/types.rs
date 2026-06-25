use std::fmt;
use crate::tableau::{Gate1Q, Gate2Q, PauliString};
use super::qasm::QasmGate;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sign {
    Plus,
    Minus,
}

impl fmt::Display for Sign {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sign::Plus  => write!(f, "+"),
            Sign::Minus => write!(f, "-"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TransPauli {
    pub sign: Sign,
    pub ops: Vec<char>,
    pub label: String,
}

impl TransPauli {
    pub fn from_pauli_string(ps: &PauliString, label: &str) -> Self {
        let sign = if ps.sign { Sign::Minus } else { Sign::Plus };
        let ops: Vec<char> = (0..ps.n).map(|q| ps.pauli_at(q)).collect();
        TransPauli { sign, ops, label: label.to_string() }
    }
}

impl fmt::Display for TransPauli {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.sign)?;
        for &c in &self.ops {
            if c == 'I' { write!(f, "_")?; } else { write!(f, "{}", c)?; }
        }
        write!(f, "<{}>", self.label)
    }
}

#[derive(Debug, Clone)]
pub struct TransClifford {
    pub ops: Vec<char>,
    pub name: String,
}

impl TransClifford {
    pub fn from_qasm_gate(gate: &QasmGate, n_qubits: usize) -> Option<Self> {
        let mut ops = vec!['_'; n_qubits];
        let name = match gate {
            QasmGate::Clifford2Q { gate: Gate2Q::CX, control, target } => {
                ops[*control] = 'Z';
                ops[*target]  = 'X';
                "CX".to_string()
            }
            QasmGate::Clifford1Q { gate: Gate1Q::S,     qubit } => { ops[*qubit] = 'Z'; "S".to_string()   }
            QasmGate::Clifford1Q { gate: Gate1Q::Sdg,   qubit } => { ops[*qubit] = 'Z'; "Sdg".to_string() }
            QasmGate::Clifford1Q { gate: Gate1Q::SX,    qubit } => { ops[*qubit] = 'X'; "SX".to_string()  }
            QasmGate::Clifford1Q { gate: Gate1Q::SXdg,  qubit } => { ops[*qubit] = 'X'; "SXdg".to_string()}
            _ => return None,
        };
        Some(TransClifford { ops, name })
    }
}

impl fmt::Display for TransClifford {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "+")?;
        for &c in &self.ops { write!(f, "{}", c)?; }
        write!(f, "<{}>", self.name)
    }
}

#[derive(Debug, Clone)]
pub enum TransItem {
    Pauli(TransPauli),
    Clifford(TransClifford),
}

impl fmt::Display for TransItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransItem::Pauli(p)    => write!(f, "{}", p),
            TransItem::Clifford(c) => write!(f, "{}", c),
        }
    }
}

use std::collections::HashSet;

// ─────────────────────────────────────────────────────────────────────────────
// Coupling graph for hardware connectivity constraints.
// ─────────────────────────────────────────────────────────────────────────────

/// An undirected coupling graph over qubits.
/// An edge (a, b) means qubits a and b can interact directly.
#[derive(Debug, Clone)]
pub struct CouplingGraph {
    edges: HashSet<(usize, usize)>,
    n_qubits: usize,
}

impl CouplingGraph {
    pub fn new(n_qubits: usize) -> Self {
        CouplingGraph { edges: HashSet::new(), n_qubits }
    }

    /// Add an undirected edge between qubits a and b.
    pub fn add_edge(&mut self, a: usize, b: usize) {
        let edge = if a < b { (a, b) } else { (b, a) };
        self.edges.insert(edge);
    }

    /// Build a fully connected coupling graph (every qubit connects to every other).
    pub fn fully_connected(n_qubits: usize) -> Self {
        let mut g = Self::new(n_qubits);
        for i in 0..n_qubits {
            for j in (i + 1)..n_qubits {
                g.add_edge(i, j);
            }
        }
        g
    }

    /// Build a linear chain: 0-1-2-..-(n-1).
    pub fn linear(n_qubits: usize) -> Self {
        let mut g = Self::new(n_qubits);
        for i in 0..(n_qubits - 1) {
            g.add_edge(i, i + 1);
        }
        g
    }

    pub fn has_edge(&self, a: usize, b: usize) -> bool {
        let edge = if a < b { (a, b) } else { (b, a) };
        self.edges.contains(&edge)
    }

    /// Check whether the induced subgraph on `qubits` is a subgraph of this
    /// coupling graph. That is, every pair of qubits in `qubits` that needs to
    /// interact must have a direct edge here.
    ///
    /// For a Pauli string, the active qubits are those with a non-identity
    /// operator. The induced subgraph check asks: is there an edge between
    /// every adjacent pair of active qubits (in qubit order)? This corresponds
    /// to checking that the Pauli product can be measured with a path on the
    /// coupling graph.
    pub fn is_pauli_compatible(&self, active_qubits: &[usize]) -> bool {
        if active_qubits.len() <= 1 {
            return true;
        }
        // Check that each consecutive pair of active qubits has an edge.
        // For a general (non-path) check you would test all pairs; using
        // consecutive pairs assumes the coupling graph is at least a path,
        // which matches typical hardware topologies.
        active_qubits.windows(2).all(|w| self.has_edge(w[0], w[1]))
    }

    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: extract active (non-identity) qubits from a PauliString
// ─────────────────────────────────────────────────────────────────────────────

fn active_qubits(ps: &PauliString) -> Vec<usize> {
    (0..ps.n).filter(|&q| ps.pauli_at(q) != 'I').collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Coupling restraint
// ─────────────────────────────────────────────────────────────────────────────

/// A restraint on which Pauli products are acceptable, either by maximum
/// weight or by compatibility with a coupling graph.
pub enum CouplingRestraint {
    MaxWidth(usize),
    Graph(CouplingGraph),
    BlockMaxWidth(usize,usize, Vec<HashSet<usize>>),
}

impl CouplingRestraint {
    /// Returns true if the Pauli string is accepted by this restraint.
    pub fn accepts_pauli(&self, ps: &PauliString) -> bool {
        match self {
            CouplingRestraint::MaxWidth(max) => ps.weight() <= *max,
            CouplingRestraint::Graph(g) => g.is_pauli_compatible(&active_qubits(ps)),
            CouplingRestraint::BlockMaxWidth(max_weight, block_max, blocks) => {
                if ps.weight()> *max_weight {
                    return false;
                }
                // Only allow connections between `block_max` or fewer blocks.
                let mut blocks_used = HashSet::new();
                for q in active_qubits(ps) {
                    for (i, block) in blocks.iter().enumerate() {
                        if block.contains(&q) {
                            blocks_used.insert(i);
                            break;
                        }
                    }
                }
                blocks_used.len() <= *block_max
            }
        }
    }

    /// Convenience: no restraint at all (always accepts).
    pub fn unrestricted() -> Self {
        CouplingRestraint::MaxWidth(usize::MAX)
    }
}