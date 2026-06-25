#![allow(dead_code)]
//! Symplectic Clifford tableau for Pauli conjugation.
//!
//! An n-qubit tableau stores 2n rows: row `2*q` = image of X_q, row `2*q+1` = image of Z_q.
//!
//! Pauli encoding: I=(0,0), X=(1,0), Y=(1,1), Z=(0,1).  Sign: false=+1, true=−1.

use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// PauliString
// ─────────────────────────────────────────────────────────────────────────────

/// A Pauli string on `n` qubits with a ±1 sign.
#[derive(Debug, Clone, PartialEq)]
pub struct PauliString {
    pub n: usize,
    pub x_bits: Vec<bool>,
    pub z_bits: Vec<bool>,
    /// false = +1, true = −1
    pub sign: bool,
}

impl PauliString {
    pub fn identity(n: usize) -> Self {
        PauliString { n, x_bits: vec![false; n], z_bits: vec![false; n], sign: false }
    }

    /// Parse a string like `"+IXYZ"` or `"ZXI"`.
    pub fn from_str(s: &str) -> Self {
        let s = s.trim();
        let (sign, chars) = if s.starts_with('-') {
            (true, &s[1..])
        } else if s.starts_with('+') {
            (false, &s[1..])
        } else {
            (false, s)
        };
        let n = chars.len();
        let mut x_bits = vec![false; n];
        let mut z_bits = vec![false; n];
        for (i, c) in chars.chars().enumerate() {
            match c {
                'I' => {}
                'X' => x_bits[i] = true,
                'Y' => {
                    x_bits[i] = true;
                    z_bits[i] = true;
                }
                'Z' => z_bits[i] = true,
                _ => panic!("Invalid Pauli character '{}'", c),
            }
        }
        PauliString { n, x_bits, z_bits, sign }
    }

    pub fn pauli_at(&self, q: usize) -> char {
        match (self.x_bits[q], self.z_bits[q]) {
            (false, false) => 'I',
            (true, false) => 'X',
            (true, true) => 'Y',
            (false, true) => 'Z',
        }
    }

    /// Number of non-identity Paulis.
    pub fn weight(&self) -> usize {
        (0..self.n).filter(|&q| self.x_bits[q] || self.z_bits[q]).count()
    }

    /// Element-wise Pauli product with phase tracking (mod 4).
    /// Panics if the result has a non-Hermitian phase (±i).
    pub fn mul(&self, other: &PauliString) -> PauliString {
        let n = self.n.max(other.n);
        let mut phase: i32 = 0;
        if self.sign {
            phase += 2;
        }
        if other.sign {
            phase += 2;
        }

        let mut x_bits = vec![false; n];
        let mut z_bits = vec![false; n];

        for q in 0..n {
            let ax = if q < self.n { self.x_bits[q] } else { false };
            let az = if q < self.n { self.z_bits[q] } else { false };
            let bx = if q < other.n { other.x_bits[q] } else { false };
            let bz = if q < other.n { other.z_bits[q] } else { false };
            x_bits[q] = ax ^ bx;
            z_bits[q] = az ^ bz;
            phase += single_qubit_mul_phase(ax, az, bx, bz);
        }

        phase = phase.rem_euclid(4);
        let sign = match phase {
            0 => false,
            2 => true,
            p => panic!(
                "Non-Hermitian phase {} in Pauli product (self={}, other={})",
                p, self, other
            ),
        };
        PauliString { n, x_bits, z_bits, sign }
    }

    /// Compute `self · i · x_img · z_img`, used when conjugating Y = i·X·Z through a tableau.
    pub fn mul_with_y_phase(&self, x_img: &PauliString, z_img: &PauliString) -> PauliString {
        let n = self.n.max(x_img.n).max(z_img.n);
        let mut phase: i32 = 0;
        if self.sign {
            phase += 2;
        }
        phase += 1; // the i factor from Y = i·X·Z
        if x_img.sign {
            phase += 2;
        }
        if z_img.sign {
            phase += 2;
        }

        let mut xz_x = vec![false; n];
        let mut xz_z = vec![false; n];
        for q in 0..n {
            let ax = if q < x_img.n { x_img.x_bits[q] } else { false };
            let az = if q < x_img.n { x_img.z_bits[q] } else { false };
            let bx = if q < z_img.n { z_img.x_bits[q] } else { false };
            let bz = if q < z_img.n { z_img.z_bits[q] } else { false };
            xz_x[q] = ax ^ bx;
            xz_z[q] = az ^ bz;
            phase += single_qubit_mul_phase(ax, az, bx, bz);
        }

        let mut result_x = vec![false; n];
        let mut result_z = vec![false; n];
        for q in 0..n {
            let ax = if q < self.n { self.x_bits[q] } else { false };
            let az = if q < self.n { self.z_bits[q] } else { false };
            let bx = xz_x[q];
            let bz = xz_z[q];
            result_x[q] = ax ^ bx;
            result_z[q] = az ^ bz;
            phase += single_qubit_mul_phase(ax, az, bx, bz);
        }

        phase = phase.rem_euclid(4);
        let sign = match phase {
            0 => false,
            2 => true,
            p => panic!(
                "Non-Hermitian phase {} when conjugating Y (x_img={}, z_img={})",
                p, x_img, z_img
            ),
        };
        PauliString { n, x_bits: result_x, z_bits: result_z, sign }
    }
}

/// Phase contribution (mod 4) from multiplying two single-qubit Paulis.
/// Encoding: I=(0,0), X=(1,0), Y=(1,1), Z=(0,1).
/// Cyclic rule: X·Y=+iZ (+1), Y·Z=+iX (+1), Z·X=+iY (+1); reversed gives −i (+3).
fn single_qubit_mul_phase(ax: bool, az: bool, bx: bool, bz: bool) -> i32 {
    match (ax, az, bx, bz) {
        (true, false, true, true) => 1,  // X·Y = iZ
        (true, true, false, true) => 1,  // Y·Z = iX
        (false, true, true, false) => 1, // Z·X = iY
        (true, true, true, false) => 3,  // Y·X = -iZ
        (false, true, true, true) => 3,  // Z·Y = -iX
        (true, false, false, true) => 3, // X·Z = -iY
        _ => 0,
    }
}

impl fmt::Display for PauliString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", if self.sign { "-" } else { "+" })?;
        for q in 0..self.n {
            write!(f, "{}", self.pauli_at(q))?;
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tableau
// ─────────────────────────────────────────────────────────────────────────────

/// Symplectic Clifford tableau on `n` qubits.
///
/// `prepend` applies a gate *before* the existing tableau (matching stim's `tableau.prepend`).
#[derive(Debug, Clone)]
pub struct Tableau {
    pub n: usize,
    /// rows[2*q] = image of X_q, rows[2*q+1] = image of Z_q
    rows: Vec<PauliString>,
}

impl Tableau {
    pub fn new(n: usize) -> Self {
        let mut rows = Vec::with_capacity(2 * n);
        for q in 0..n {
            let mut xrow = PauliString::identity(n);
            xrow.x_bits[q] = true;
            rows.push(xrow);
            let mut zrow = PauliString::identity(n);
            zrow.z_bits[q] = true;
            rows.push(zrow);
        }
        Tableau { n, rows }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    /// Conjugate a Pauli string through this tableau: P → U·P·U†.
    pub fn conjugate(&self, pauli: &PauliString) -> PauliString {
        let n = self.n.max(pauli.n);
        let mut result = PauliString::identity(n);
        if pauli.sign {
            result.sign = true;
        }
        for q in 0..pauli.n {
            let (px, pz) = (pauli.x_bits[q], pauli.z_bits[q]);
            if !px && !pz {
                continue;
            }
            let x_image = self.extended_row(2 * q, n);
            let z_image = self.extended_row(2 * q + 1, n);
            if px && pz {
                // Y_q = i·X_q·Z_q
                result = result.mul_with_y_phase(&x_image, &z_image);
            } else if px {
                result = result.mul(&x_image);
            } else {
                result = result.mul(&z_image);
            }
        }
        result
    }

    /// Get row `r` zero-padded to length `n`.
    fn extended_row(&self, r: usize, n: usize) -> PauliString {
        let row = &self.rows[r];
        if row.n == n {
            return row.clone();
        }
        let mut x_bits = row.x_bits.clone();
        let mut z_bits = row.z_bits.clone();
        x_bits.resize(n, false);
        z_bits.resize(n, false);
        PauliString { n, x_bits, z_bits, sign: row.sign }
    }

    pub fn prepend_1q_correct(&mut self, gate: Gate1Q, q: usize) {
        let old_x = self.rows[2 * q].clone();
        let old_z = self.rows[2 * q + 1].clone();
        let (gx_sign, gx_x, gx_z) = gate.x_image_correct();
        let (gz_sign, gz_x, gz_z) = gate.z_image_correct();
        self.rows[2 * q] = combine_rows(&old_x, gx_x, &old_z, gx_z, gx_sign);
        self.rows[2 * q + 1] = combine_rows(&old_x, gz_x, &old_z, gz_z, gz_sign);
    }

    pub fn prepend_2q(&mut self, gate: Gate2Q, q0: usize, q1: usize) {
        let old_x0 = self.rows[2 * q0].clone();
        let old_z0 = self.rows[2 * q0 + 1].clone();
        let old_x1 = self.rows[2 * q1].clone();
        let old_z1 = self.rows[2 * q1 + 1].clone();
        let (new_x0, new_z0, new_x1, new_z1) = gate.apply(&old_x0, &old_z0, &old_x1, &old_z1);
        self.rows[2 * q0] = new_x0;
        self.rows[2 * q0 + 1] = new_z0;
        self.rows[2 * q1] = new_x1;
        self.rows[2 * q1 + 1] = new_z1;
    }
}

/// Compute the new row image after prepending a gate whose generator image is
/// `sign * (use_x ? X : I) * (use_z ? Z : I)`.  When both flags are set the
/// image is Y = i·X·Z, requiring `mul_with_y_phase`.
fn combine_rows(
    row_x: &PauliString, use_x: bool, row_z: &PauliString, use_z: bool, sign: bool,
) -> PauliString {
    let n = row_x.n;
    let mut result = PauliString::identity(n);
    result.sign = sign;
    if use_x && use_z {
        result = result.mul_with_y_phase(row_x, row_z);
    } else if use_x {
        result = result.mul(row_x);
    } else if use_z {
        result = result.mul(row_z);
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-qubit gates
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Gate1Q {
    H,
    S,
    Sdg,
    SX,
    SXdg,
    X,
    Y,
    Z,
}

impl Gate1Q {
    /// Returns `(sign_negative, x_bit, z_bit)` for the image of X under conjugation.
    pub fn x_image_correct(&self) -> (bool, bool, bool) {
        match self {
            Gate1Q::H => (false, false, true),    // X → Z
            Gate1Q::S => (false, true, true),     // X → Y
            Gate1Q::Sdg => (true, true, true),    // X → -Y
            Gate1Q::SX => (false, true, false),   // X → X
            Gate1Q::SXdg => (false, true, false), // X → X
            Gate1Q::X => (false, true, false),    // X → X
            Gate1Q::Y => (true, true, false),     // X → -X
            Gate1Q::Z => (true, true, false),     // X → -X
        }
    }

    /// Returns `(sign_negative, x_bit, z_bit)` for the image of Z under conjugation.
    pub fn z_image_correct(&self) -> (bool, bool, bool) {
        match self {
            Gate1Q::H => (false, true, false),   // Z → X
            Gate1Q::S => (false, false, true),   // Z → Z
            Gate1Q::Sdg => (false, false, true), // Z → Z
            Gate1Q::SX => (true, true, true),    // Z → -Y
            Gate1Q::SXdg => (false, true, true), // Z → Y
            Gate1Q::X => (true, false, true),    // Z → -Z
            Gate1Q::Y => (true, false, true),    // Z → -Z
            Gate1Q::Z => (false, false, true),   // Z → Z
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Gate2Q {
    CX, // control=q0, target=q1
    CZ,
    Swap,
}

impl Gate2Q {
    /// Clifford conjugation rules:
    ///   CX:   X0→X0X1, Z0→Z0,   X1→X1,  Z1→Z0Z1
    ///   CZ:   X0→X0Z1, Z0→Z0,   X1→Z0X1, Z1→Z1
    ///   SWAP: X0→X1,   Z0→Z1,   X1→X0,  Z1→Z0
    fn apply(
        &self, x0: &PauliString, z0: &PauliString, x1: &PauliString, z1: &PauliString,
    ) -> (PauliString, PauliString, PauliString, PauliString) {
        match self {
            Gate2Q::CX => (x0.mul(x1), z0.clone(), x1.clone(), z0.mul(z1)),
            Gate2Q::CZ => (x0.mul(z1), z0.clone(), z0.mul(x1), z1.clone()),
            Gate2Q::Swap => (x1.clone(), z1.clone(), x0.clone(), z0.clone()),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── PauliString::from_str ─────────────────────────────────────────────────

    #[test]
    fn pauli_string_from_str_identity() {
        let p = PauliString::from_str("III");
        assert_eq!(p.n, 3);
        assert!(!p.sign);
        assert_eq!(p.weight(), 0);
    }

    #[test]
    fn pauli_string_from_str_xyz() {
        let p = PauliString::from_str("+XYZ");
        assert_eq!(p.n, 3);
        assert!(!p.sign);
        assert_eq!(p.pauli_at(0), 'X');
        assert_eq!(p.pauli_at(1), 'Y');
        assert_eq!(p.pauli_at(2), 'Z');
    }

    #[test]
    fn pauli_string_from_str_negative() {
        let p = PauliString::from_str("-XZ");
        assert!(p.sign);
        assert_eq!(p.pauli_at(0), 'X');
        assert_eq!(p.pauli_at(1), 'Z');
    }

    #[test]
    fn pauli_string_weight() {
        let p = PauliString::from_str("+XIZIY");
        assert_eq!(p.weight(), 3);
    }

    // ── PauliString::mul ──────────────────────────────────────────────────────

    #[test]
    fn pauli_mul_xx_is_identity() {
        let x = PauliString::from_str("+X");
        let result = x.mul(&x);
        assert_eq!(result.pauli_at(0), 'I');
        assert!(!result.sign);
    }

    #[test]
    fn pauli_mul_zz_is_identity() {
        let z = PauliString::from_str("+Z");
        let result = z.mul(&z);
        assert_eq!(result.pauli_at(0), 'I');
        assert!(!result.sign);
    }

    #[test]
    fn pauli_mul_yy_is_identity() {
        let y = PauliString::from_str("+Y");
        let result = y.mul(&y);
        assert_eq!(result.pauli_at(0), 'I');
        assert!(!result.sign);
    }

    #[test]
    fn pauli_mul_zx_gives_y_with_phase() {
        // (XY)·(YX): qubit 0: X·Y=iZ, qubit 1: Y·X=-iZ → total phase 1, result=ZZ
        let xy = PauliString::from_str("+XY");
        let yx = PauliString::from_str("+YX");
        let result = xy.mul(&yx);
        assert_eq!(result.pauli_at(0), 'Z');
        assert_eq!(result.pauli_at(1), 'Z');
        assert!(!result.sign);
    }

    #[test]
    fn pauli_mul_negative_signs() {
        let neg_x = PauliString::from_str("-X");
        let pos_x = PauliString::from_str("+X");
        let result = neg_x.mul(&pos_x);
        assert_eq!(result.pauli_at(0), 'I');
        assert!(result.sign);
    }

    // ── Tableau identity ──────────────────────────────────────────────────────

    #[test]
    fn tableau_identity_conjugates_x_trivially() {
        let t = Tableau::new(3);
        let p = PauliString::from_str("+XII");
        let result = t.conjugate(&p);
        assert_eq!(result.pauli_at(0), 'X');
        assert_eq!(result.pauli_at(1), 'I');
        assert_eq!(result.pauli_at(2), 'I');
        assert!(!result.sign);
    }

    #[test]
    fn tableau_identity_conjugates_z_trivially() {
        let t = Tableau::new(2);
        let p = PauliString::from_str("+IZ");
        let result = t.conjugate(&p);
        assert_eq!(result.pauli_at(0), 'I');
        assert_eq!(result.pauli_at(1), 'Z');
        assert!(!result.sign);
    }

    #[test]
    fn tableau_identity_conjugates_y_correctly() {
        let t = Tableau::new(1);
        let y = PauliString::from_str("+Y");
        let result = t.conjugate(&y);
        assert_eq!(result.pauli_at(0), 'Y');
        assert!(!result.sign);
    }

    #[test]
    fn tableau_identity_conjugates_xyz_trivially() {
        let t = Tableau::new(3);
        let p = PauliString::from_str("+XYZ");
        let result = t.conjugate(&p);
        assert_eq!(result.pauli_at(0), 'X');
        assert_eq!(result.pauli_at(1), 'Y');
        assert_eq!(result.pauli_at(2), 'Z');
        assert!(!result.sign);
    }

    // ── H gate ────────────────────────────────────────────────────────────────

    #[test]
    fn h_swaps_x_and_z() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::H, 0);
        let x = PauliString::from_str("+X");
        let result = t.conjugate(&x);
        assert_eq!(result.pauli_at(0), 'Z');
        assert!(!result.sign);
        let z = PauliString::from_str("+Z");
        let result = t.conjugate(&z);
        assert_eq!(result.pauli_at(0), 'X');
        assert!(!result.sign);
    }

    #[test]
    fn h_twice_is_identity() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::H, 0);
        t.prepend_1q_correct(Gate1Q::H, 0);
        let x = PauliString::from_str("+X");
        let result = t.conjugate(&x);
        assert_eq!(result.pauli_at(0), 'X');
        assert!(!result.sign);
    }

    #[test]
    fn h_maps_y_to_minus_y() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::H, 0);
        let y = PauliString::from_str("+Y");
        let result = t.conjugate(&y);
        assert_eq!(result.pauli_at(0), 'Y');
        assert!(result.sign);
    }

    // ── S gate ────────────────────────────────────────────────────────────────

    #[test]
    fn s_maps_x_to_y() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::S, 0);
        let x = PauliString::from_str("+X");
        let result = t.conjugate(&x);
        assert_eq!(result.pauli_at(0), 'Y');
        assert!(!result.sign);
    }

    #[test]
    fn s_maps_z_to_z() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::S, 0);
        let z = PauliString::from_str("+Z");
        let result = t.conjugate(&z);
        assert_eq!(result.pauli_at(0), 'Z');
        assert!(!result.sign);
    }

    #[test]
    fn s_maps_y_to_minus_x() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::S, 0);
        let y = PauliString::from_str("+Y");
        let result = t.conjugate(&y);
        assert_eq!(result.pauli_at(0), 'X');
        assert!(result.sign);
    }

    // ── Sdg gate ──────────────────────────────────────────────────────────────

    #[test]
    fn sdg_maps_x_to_minus_y() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::Sdg, 0);
        let x = PauliString::from_str("+X");
        let result = t.conjugate(&x);
        assert_eq!(result.pauli_at(0), 'Y');
        assert!(result.sign);
    }

    #[test]
    fn sdg_maps_z_to_z() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::Sdg, 0);
        let z = PauliString::from_str("+Z");
        let result = t.conjugate(&z);
        assert_eq!(result.pauli_at(0), 'Z');
        assert!(!result.sign);
    }

    // ── S then Sdg = identity ─────────────────────────────────────────────────

    #[test]
    fn s_sdg_is_identity() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::S, 0);
        t.prepend_1q_correct(Gate1Q::Sdg, 0);
        let x = PauliString::from_str("+X");
        let result = t.conjugate(&x);
        assert_eq!(result.pauli_at(0), 'X');
        assert!(!result.sign);
        let z = PauliString::from_str("+Z");
        let result = t.conjugate(&z);
        assert_eq!(result.pauli_at(0), 'Z');
        assert!(!result.sign);
    }

    // ── SX gate ───────────────────────────────────────────────────────────────

    #[test]
    fn sx_maps_x_to_x() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::SX, 0);
        let x = PauliString::from_str("+X");
        let result = t.conjugate(&x);
        assert_eq!(result.pauli_at(0), 'X');
        assert!(!result.sign);
    }

    #[test]
    fn sx_maps_z_to_minus_y() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::SX, 0);
        let z = PauliString::from_str("+Z");
        let result = t.conjugate(&z);
        assert_eq!(result.pauli_at(0), 'Y');
        assert!(result.sign);
    }

    #[test]
    fn sxdg_maps_z_to_y() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::SXdg, 0);
        let z = PauliString::from_str("+Z");
        let result = t.conjugate(&z);
        assert_eq!(result.pauli_at(0), 'Y');
        assert!(!result.sign);
    }

    // ── Z gate ────────────────────────────────────────────────────────────────

    #[test]
    fn z_maps_x_to_minus_x() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::Z, 0);
        let x = PauliString::from_str("+X");
        let result = t.conjugate(&x);
        assert_eq!(result.pauli_at(0), 'X');
        assert!(result.sign);
    }

    #[test]
    fn z_maps_z_to_z() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::Z, 0);
        let z = PauliString::from_str("+Z");
        let result = t.conjugate(&z);
        assert_eq!(result.pauli_at(0), 'Z');
        assert!(!result.sign);
    }

    // ── X gate ────────────────────────────────────────────────────────────────

    #[test]
    fn x_maps_z_to_minus_z() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::X, 0);
        let z = PauliString::from_str("+Z");
        let result = t.conjugate(&z);
        assert_eq!(result.pauli_at(0), 'Z');
        assert!(result.sign);
    }

    #[test]
    fn x_maps_x_to_x() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::X, 0);
        let x = PauliString::from_str("+X");
        let result = t.conjugate(&x);
        assert_eq!(result.pauli_at(0), 'X');
        assert!(!result.sign);
    }

    // ── CX gate ───────────────────────────────────────────────────────────────

    #[test]
    fn cx_maps_x0_to_x0x1() {
        let mut t = Tableau::new(2);
        t.prepend_2q(Gate2Q::CX, 0, 1);
        let x0 = PauliString::from_str("+XI");
        let result = t.conjugate(&x0);
        assert_eq!(result.pauli_at(0), 'X');
        assert_eq!(result.pauli_at(1), 'X');
        assert!(!result.sign);
    }

    #[test]
    fn cx_maps_z1_to_z0z1() {
        let mut t = Tableau::new(2);
        t.prepend_2q(Gate2Q::CX, 0, 1);
        let z1 = PauliString::from_str("+IZ");
        let result = t.conjugate(&z1);
        assert_eq!(result.pauli_at(0), 'Z');
        assert_eq!(result.pauli_at(1), 'Z');
        assert!(!result.sign);
    }

    #[test]
    fn cx_maps_z0_to_z0() {
        let mut t = Tableau::new(2);
        t.prepend_2q(Gate2Q::CX, 0, 1);
        let z0 = PauliString::from_str("+ZI");
        let result = t.conjugate(&z0);
        assert_eq!(result.pauli_at(0), 'Z');
        assert_eq!(result.pauli_at(1), 'I');
        assert!(!result.sign);
    }

    #[test]
    fn cx_maps_x1_to_x1() {
        let mut t = Tableau::new(2);
        t.prepend_2q(Gate2Q::CX, 0, 1);
        let x1 = PauliString::from_str("+IX");
        let result = t.conjugate(&x1);
        assert_eq!(result.pauli_at(0), 'I');
        assert_eq!(result.pauli_at(1), 'X');
        assert!(!result.sign);
    }

    // ── CZ gate ───────────────────────────────────────────────────────────────

    #[test]
    fn cz_maps_x0_to_x0z1() {
        let mut t = Tableau::new(2);
        t.prepend_2q(Gate2Q::CZ, 0, 1);
        let x0 = PauliString::from_str("+XI");
        let result = t.conjugate(&x0);
        assert_eq!(result.pauli_at(0), 'X');
        assert_eq!(result.pauli_at(1), 'Z');
        assert!(!result.sign);
    }

    // ── SWAP gate ─────────────────────────────────────────────────────────────

    #[test]
    fn swap_maps_x0_to_x1() {
        let mut t = Tableau::new(2);
        t.prepend_2q(Gate2Q::Swap, 0, 1);
        let x0 = PauliString::from_str("+XI");
        let result = t.conjugate(&x0);
        assert_eq!(result.pauli_at(0), 'I');
        assert_eq!(result.pauli_at(1), 'X');
        assert!(!result.sign);
    }

    // ── Multi-qubit Y conjugation ─────────────────────────────────────────────

    #[test]
    fn identity_conjugates_multi_qubit_y() {
        let t = Tableau::new(3);
        let p = PauliString::from_str("+XYZ");
        let result = t.conjugate(&p);
        assert_eq!(result.pauli_at(0), 'X');
        assert_eq!(result.pauli_at(1), 'Y');
        assert_eq!(result.pauli_at(2), 'Z');
        assert!(!result.sign);
    }

    // ── S^4 = identity ────────────────────────────────────────────────────────

    #[test]
    fn s_four_times_is_identity() {
        let mut t = Tableau::new(1);
        for _ in 0..4 {
            t.prepend_1q_correct(Gate1Q::S, 0);
        }
        let x = PauliString::from_str("+X");
        let result = t.conjugate(&x);
        assert_eq!(result.pauli_at(0), 'X');
        assert!(!result.sign);
        let z = PauliString::from_str("+Z");
        let result = t.conjugate(&z);
        assert_eq!(result.pauli_at(0), 'Z');
        assert!(!result.sign);
    }

    // ── H·S·H = SX (up to global phase) ──────────────────────────────────────

    #[test]
    fn h_s_h_maps_z_to_x() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::H, 0);
        t.prepend_1q_correct(Gate1Q::S, 0);
        t.prepend_1q_correct(Gate1Q::H, 0);
        // H·S·H acts as SX: Z → -Y
        let z = PauliString::from_str("+Z");
        let result = t.conjugate(&z);
        assert_eq!(result.pauli_at(0), 'Y');
        assert!(result.sign);
    }

    // ── PauliString::pauli_at ─────────────────────────────────────────────────

    #[test]
    fn pauli_at_identity_qubit() {
        let ps = PauliString::from_str("+IXZ");
        assert_eq!(ps.pauli_at(0), 'I');
        assert_eq!(ps.pauli_at(1), 'X');
        assert_eq!(ps.pauli_at(2), 'Z');
    }

    #[test]
    fn pauli_at_y_qubit() {
        let ps = PauliString::from_str("+Y");
        assert_eq!(ps.pauli_at(0), 'Y');
    }

    // ── PauliString::weight ───────────────────────────────────────────────────

    #[test]
    fn weight_identity_is_zero() {
        let ps = PauliString::from_str("+III");
        assert_eq!(ps.weight(), 0);
    }

    #[test]
    fn weight_counts_non_identity() {
        let ps = PauliString::from_str("+XYZ");
        assert_eq!(ps.weight(), 3);
    }

    #[test]
    fn weight_mixed() {
        let ps = PauliString::from_str("+XIZ");
        assert_eq!(ps.weight(), 2);
    }

    // ── Tableau::len ──────────────────────────────────────────────────────────

    #[test]
    fn tableau_len_matches_n_qubits() {
        let t = Tableau::new(3);
        assert_eq!(t.len(), 3);
    }

    // ── Gate1Q images — Sdg·Y = X ────────────────────────────────────────────

    #[test]
    fn sdg_maps_y_to_x() {
        // Sdg: X→-Y, Z→Z; Y = iXZ → Sdg(Y) = i·(-Y)·Z = -i·YZ = X (up to phase)
        // Concretely: Sdg maps Y→X
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::Sdg, 0);
        let y = PauliString::from_str("+Y");
        let result = t.conjugate(&y);
        assert_eq!(result.pauli_at(0), 'X');
    }

    // ── Gate2Q::CZ — Z0 and Z1 are fixed points ──────────────────────────────

    #[test]
    fn cz_maps_z0_to_z0() {
        let mut t = Tableau::new(2);
        t.prepend_2q(Gate2Q::CZ, 0, 1);
        let z0 = PauliString::from_str("+ZI");
        let result = t.conjugate(&z0);
        assert_eq!(result.pauli_at(0), 'Z');
        assert_eq!(result.pauli_at(1), 'I');
        assert!(!result.sign);
    }

    #[test]
    fn cz_maps_z1_to_z1() {
        let mut t = Tableau::new(2);
        t.prepend_2q(Gate2Q::CZ, 0, 1);
        let z1 = PauliString::from_str("+IZ");
        let result = t.conjugate(&z1);
        assert_eq!(result.pauli_at(0), 'I');
        assert_eq!(result.pauli_at(1), 'Z');
        assert!(!result.sign);
    }

    #[test]
    fn cz_maps_x1_to_z0x1() {
        let mut t = Tableau::new(2);
        t.prepend_2q(Gate2Q::CZ, 0, 1);
        let x1 = PauliString::from_str("+IX");
        let result = t.conjugate(&x1);
        assert_eq!(result.pauli_at(0), 'Z');
        assert_eq!(result.pauli_at(1), 'X');
        assert!(!result.sign);
    }

    // ── Gate2Q::Swap — swaps both X and Z ────────────────────────────────────

    #[test]
    fn swap_maps_z0_to_z1() {
        let mut t = Tableau::new(2);
        t.prepend_2q(Gate2Q::Swap, 0, 1);
        let z0 = PauliString::from_str("+ZI");
        let result = t.conjugate(&z0);
        assert_eq!(result.pauli_at(0), 'I');
        assert_eq!(result.pauli_at(1), 'Z');
        assert!(!result.sign);
    }

    #[test]
    fn swap_maps_x1_to_x0() {
        let mut t = Tableau::new(2);
        t.prepend_2q(Gate2Q::Swap, 0, 1);
        let x1 = PauliString::from_str("+IX");
        let result = t.conjugate(&x1);
        assert_eq!(result.pauli_at(0), 'X');
        assert_eq!(result.pauli_at(1), 'I');
        assert!(!result.sign);
    }

    // ── Negative sign propagation ─────────────────────────────────────────────

    #[test]
    fn conjugate_preserves_negative_sign() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::H, 0);
        let neg_x = PauliString::from_str("-X");
        let result = t.conjugate(&neg_x);
        // H maps X→Z, so -X → -Z
        assert_eq!(result.pauli_at(0), 'Z');
        assert!(result.sign);
    }

    // ── Multi-qubit conjugation with identity tableau ─────────────────────────

    #[test]
    fn conjugate_multi_qubit_independent_qubits() {
        // Identity tableau: conjugating any Pauli returns itself
        let t = Tableau::new(3);
        let ps = PauliString::from_str("+XYZ");
        let result = t.conjugate(&ps);
        assert_eq!(result.pauli_at(0), 'X');
        assert_eq!(result.pauli_at(1), 'Y');
        assert_eq!(result.pauli_at(2), 'Z');
        assert!(!result.sign);
    }
}
