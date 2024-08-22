use ndarray::Array1;
use quantum::particles::Particles;
use std::f64::consts::PI;

use crate::grid::Grid;

/// Creates kinetic Hamiltonian for given grid and collision parameters.
pub fn kinetic_hamiltonian(grid: &Grid, collision_params: &Particles) -> Array1<f64> {
    let momentum_step = 2.0 * PI / (grid.nodes.last().unwrap() - grid.nodes.first().unwrap()) * (1. - 1. / grid.nodes_no as f64);
    let length: i64 = grid.nodes_no as i64;

    let momenta: Vec<f64> = (0..length / 2)
        .chain(-length / 2..0)
        .map(|x| x as f64 * momentum_step)
        .collect();

    momenta
        .into_iter()
        .map(|k| k * k / (2.0 * collision_params.red_mass()))
        .collect()
}
