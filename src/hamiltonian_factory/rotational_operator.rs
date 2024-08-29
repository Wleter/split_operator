use ndarray::{Array1, Array2};
use quantum::particles::Particles;

use crate::grid::Grid;

/// Creates rotational Hamiltonian matrix for given radial and polar grids and collision parameters.
pub fn rotational_hamiltonian(
    radial_grid: &Grid,
    polar_grid: &Grid,
    collision_params: &Particles,
    rotational_const: f64,
    omega: i64
) -> Array2<f64> {
    let l: Vec<i64> = (omega.abs()..(polar_grid.nodes_no as i64 + omega.abs())).collect();
    let r = Array1::<f64>::from_vec(radial_grid.nodes.clone());

    let mut operator_matrix = Array2::<f64>::zeros((radial_grid.nodes_no, polar_grid.nodes_no));

    for i in 0..radial_grid.nodes_no {
        for j in 0..polar_grid.nodes_no {
            operator_matrix[[i, j]] = ((l[j] * (l[j] + 1)) as f64)
                * (rotational_const + 1.0 / (2.0 * collision_params.red_mass() * r[i] * r[i]));
        }
    }

    operator_matrix
}
