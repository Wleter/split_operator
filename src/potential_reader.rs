use crate::grid::Grid;
use ndarray::{Array1, Array2};
use ndarray_npy::{read_npy, ReadNpyError};
use pyo3::prelude::*;

/// Loads potential from a file. Use case only for Ne Ocs propagation problem or for potentials with same saved data structure.
pub fn load_potential(
    path: &str,
    name: &str,
    r_grid: &Grid,
    polar_grid: &Grid,
    kx: usize,
    ky: usize,
    is_gamma: bool,
) -> Result<Array2<f64>, ReadNpyError> {
    let grid_params: Array1<f64> = read_npy(format!("{path}{name}_grid.npy"))?;

    let r_start = r_grid.nodes[0];
    let r_end = r_grid.nodes[r_grid.nodes_no - 1];
    let r_nodes_no = r_grid.nodes_no;
    let polar_nodes_no = polar_grid.nodes_no;

    if grid_params[0] != r_start
        || grid_params[1] != r_end
        || grid_params[2] != r_nodes_no as f64
        || grid_params[3] as f32 != polar_grid.nodes[0] as f32
        || grid_params[4] as f32 != polar_grid.nodes[polar_grid.nodes_no - 1] as f32
        || grid_params[5] != polar_nodes_no as f64
    {
        let code = include_str!("potential_saver.py");

        Python::with_gil(|py| -> PyResult<()> {
            let fun: Py<PyAny> = PyModule::from_code(py, code, "", "")?
                .getattr("save_potential")?
                .into();

            fun.call1(
                py,
                (
                    path,
                    format!("{name}.dat"),
                    r_start,
                    r_end,
                    r_nodes_no,
                    polar_nodes_no,
                    kx,
                    ky,
                    is_gamma,
                ),
            )?;

            Ok(())
        })
        .unwrap();
    }

    let potential: Array2<f64> = read_npy(format!("{path}{name}.npy"))?;

    Ok(potential)
}
