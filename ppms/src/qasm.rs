use crate::tableau::{Gate1Q, Gate2Q};

#[derive(Debug, Clone)]
pub enum QasmGate {
    Clifford1Q { gate: Gate1Q, qubit: usize },
    Clifford2Q { gate: Gate2Q, control: usize, target: usize },
    T { qubit: usize },
    Tdg { qubit: usize },
    Measure { qubit: usize },
    Barrier,
}

impl QasmGate {
    pub fn qubits(&self) -> Vec<usize> {
        match self {
            QasmGate::Clifford1Q { qubit, .. } => vec![*qubit],
            QasmGate::Clifford2Q { control, target, .. } => {
                let mut v = vec![*control, *target];
                v.sort_unstable();
                v
            }
            QasmGate::T { qubit } => vec![*qubit],
            QasmGate::Tdg { qubit } => vec![*qubit],
            QasmGate::Measure { qubit } => vec![*qubit],
            QasmGate::Barrier => vec![],
        }
    }
}

pub fn parse_qasm_str(qasm: &str) -> std::io::Result<(usize, Vec<QasmGate>)> {
    let mut n_qubits = 0usize;
    let mut gates = Vec::new();

    for line in qasm.lines() {
        let line = line.trim();
        // identical body to parse_qasm, just iterating lines of a string
        // instead of lines of a BufReader
        if line.is_empty()
            || line.starts_with("//")
            || line.starts_with("OPENQASM")
            || line.starts_with("include")
            || line.starts_with("creg")
            || line.starts_with("gate ")
            || line.starts_with('{')
            || line.starts_with('}')
            || line.starts_with("U(")
        {
            continue;
        }
        if line.starts_with("barrier") {
            gates.push(QasmGate::Barrier);
            continue;
        }
        if line.starts_with("qreg") {
            if let Some(n) = parse_qreg(line) {
                n_qubits = n;
            }
            continue;
        }
        if line.starts_with("measure") {
            if let Some(q) = parse_single_qubit_index(line) {
                gates.push(QasmGate::Measure { qubit: q });
            }
            continue;
        }
        if let Some(g) = try_parse_2q(line) { gates.push(g); continue; }
        if let Some(g) = try_parse_1q(line) { gates.push(g); continue; }
    }

    let gates = reorder_by_cycles(gates, n_qubits);
    Ok((n_qubits, gates))
}

// And refactor the original to avoid duplication:
pub fn parse_qasm(path: &str) -> std::io::Result<(usize, Vec<QasmGate>)> {
    let content = std::fs::read_to_string(path)?;
    parse_qasm_str(&content)
}

fn reorder_by_cycles(gates: Vec<QasmGate>, n_qubits: usize) -> Vec<QasmGate> {
    if n_qubits == 0 {
        return gates;
    }

    let mut next_cycle = vec![0usize; n_qubits];
    let mut cycles: Vec<Vec<(usize, QasmGate)>> = Vec::new();

    for gate in gates {
        if matches!(gate, QasmGate::Barrier) {
            let sync = next_cycle.iter().copied().max().unwrap_or(0);
            if sync >= cycles.len() {
                cycles.resize_with(sync + 1, Vec::new);
            }
            cycles[sync].push((usize::MAX, gate));
            let after = sync + 1;
            for q in next_cycle.iter_mut() {
                *q = after;
            }
            continue;
        }

        let qubits = gate.qubits();
        let cycle = qubits.iter().map(|&q| next_cycle[q]).max().unwrap_or(0);
        if cycle >= cycles.len() {
            cycles.resize_with(cycle + 1, Vec::new);
        }
        let min_q = *qubits.iter().min().unwrap();
        cycles[cycle].push((min_q, gate));
        for &q in &qubits {
            next_cycle[q] = cycle + 1;
        }
    }

    let mut result = Vec::new();
    for cycle in cycles {
        let mut sorted = cycle;
        sorted.sort_by_key(|(min_q, _)| *min_q);
        for (_, gate) in sorted {
            if !matches!(gate, QasmGate::Barrier) {
                result.push(gate);
            }
        }
    }
    result
}

fn parse_qreg(line: &str) -> Option<usize> {
    let start = line.find('[')? + 1;
    let end = line.find(']')?;
    line[start..end].parse().ok()
}

fn parse_single_qubit_index(line: &str) -> Option<usize> {
    let start = line.find('[')? + 1;
    let end = line.find(']')?;
    line[start..end].parse().ok()
}

fn parse_two_qubit_indices(line: &str) -> Option<(usize, usize)> {
    let mut indices = line.split('[').skip(1);
    let a: usize = indices.next()?.split(']').next()?.parse().ok()?;
    let b: usize = indices.next()?.split(']').next()?.parse().ok()?;
    Some((a, b))
}

fn try_parse_2q(line: &str) -> Option<QasmGate> {
    let lower = line.to_lowercase();
    if lower.starts_with("cx ") || lower.starts_with("cx\t") {
        let (a, b) = parse_two_qubit_indices(line)?;
        return Some(QasmGate::Clifford2Q { gate: Gate2Q::CX, control: a, target: b });
    }
    if lower.starts_with("cz ") || lower.starts_with("cz\t") {
        let (a, b) = parse_two_qubit_indices(line)?;
        return Some(QasmGate::Clifford2Q { gate: Gate2Q::CZ, control: a, target: b });
    }
    if lower.starts_with("swap ") || lower.starts_with("swap\t") {
        let (a, b) = parse_two_qubit_indices(line)?;
        return Some(QasmGate::Clifford2Q { gate: Gate2Q::Swap, control: a, target: b });
    }
    None
}

fn try_parse_1q(line: &str) -> Option<QasmGate> {
    let mut parts = line.splitn(2, |c: char| c.is_whitespace());
    let name = parts.next()?.trim().to_lowercase();
    let name = name.trim_end_matches(';');
    let q = parse_single_qubit_index(line)?;
    let gate = match name {
        "h"    => QasmGate::Clifford1Q { gate: Gate1Q::H,     qubit: q },
        "s"    => QasmGate::Clifford1Q { gate: Gate1Q::S,     qubit: q },
        "sdg"  => QasmGate::Clifford1Q { gate: Gate1Q::Sdg,   qubit: q },
        "sx"   => QasmGate::Clifford1Q { gate: Gate1Q::SX,    qubit: q },
        "sxdg" => QasmGate::Clifford1Q { gate: Gate1Q::SXdg,  qubit: q },
        "x"    => QasmGate::Clifford1Q { gate: Gate1Q::X,     qubit: q },
        "y"    => QasmGate::Clifford1Q { gate: Gate1Q::Y,     qubit: q },
        "z"    => QasmGate::Clifford1Q { gate: Gate1Q::Z,     qubit: q },
        "t"    => QasmGate::T   { qubit: q },
        "tdg"  => QasmGate::Tdg { qubit: q },
        _ => return None,
    };
    Some(gate)
}