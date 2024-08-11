use ndarray::Array2;
use num::complex::Complex64;

use crate::{
    grid::Grid,
    propagator::matrix_transformation::MatrixTransformation,
    special_functions::legendre_polynomials,
};

/// Creates diagonalization to Legendre polynomials eigenbasis for given polar_grid
pub fn legendre_diagonalization_operator(polar_grid: &Grid) -> MatrixTransformation {
    let l_max = polar_grid.nodes_no as i64 - 1;
    let l: Vec<i64> = (0..(l_max + 1)).collect();

    let l_grid = Grid::new_linear_countable(
        "angular_momentum",
        0.0,
        l_max as f64,
        l_max as usize + 1,
        polar_grid.dimension_no,
    );

    let mut legendre_diagonalization = MatrixTransformation::new(&polar_grid, l_grid);
    let mut transformation = Array2::<Complex64>::zeros((polar_grid.nodes_no, polar_grid.nodes_no));

    for j in 0..polar_grid.nodes_no {
        let pl = legendre_polynomials(l_max as usize, polar_grid.nodes[j].cos());
        for i in 0..polar_grid.nodes_no {
            transformation[[i, j]] = Complex64::from((l[i] as f64 + 0.5).sqrt() * pl[i]);
        }
    }
    let inverse_transformation = transformation.clone().reversed_axes();

    for j in 0..polar_grid.nodes_no {
        for i in 0..polar_grid.nodes_no {
            transformation[[i, j]] *= polar_grid.weights[j];
        }
    }

    legendre_diagonalization.set_diagonalization_matrix(transformation, inverse_transformation);

    legendre_diagonalization
}
