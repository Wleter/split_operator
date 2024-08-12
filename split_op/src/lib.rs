use pyo3::prelude::*;

use split_operator::Grid;

#[pyclass(name = "Grid")]
struct GridPy(Grid); 

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn split_op(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;

    m.add_class::<GridPy>()?;
    Ok(())
}
