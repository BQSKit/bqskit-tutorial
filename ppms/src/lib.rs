// ── existing modules ──────────────────────────────────────────────────────────
pub mod tableau;
pub mod transpiler;
pub mod qasm;
pub mod types;

// ── Python bindings ───────────────────────────────────────────────────────────
use pyo3::prelude::*;

#[pymodule]
fn ppm_transpiler(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::transpiler::python::register(m)?;
    Ok(())
}