use ndarray::Array2;
use num::complex::Complex64;
use faer_ext::*;
use crate::propagator::state_matrix_transformation::StateMatrixTransformation;

use crate::special_functions::{associated_legendre_polynomials, normalization};
use crate::{
    grid::Grid,
    propagator::matrix_transformation::MatrixTransformation,
    special_functions::legendre_polynomials,
};

/// Creates diagonalization to Legendre polynomials eigenbasis for given polar_grid
pub fn legendre_diagonalization_operator(polar_grid: &Grid) -> MatrixTransformation {
    let l_max = polar_grid.nodes_no as i64 - 1;
    let l: Vec<i64> = (0..=l_max).collect();

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

/// Creates diagonalization to Associated Legendre polynomials eigenbasis for given polar_grid and omega value
pub fn associated_legendre_diagonalization_operator(polar_grid: &Grid, omega: isize) -> MatrixTransformation {
    let l_max = polar_grid.nodes_no as u32 - 1;
    let l: Vec<u32> = (0..=l_max).collect();

    let l_grid = Grid::new_linear_countable(
        "red_angular_momentum",
        0.0,
        l_max as f64,
        l_max as usize + 1,
        polar_grid.dimension_no,
    );

    let mut legendre_diagonalization = MatrixTransformation::new(&polar_grid, l_grid);
    let mut transformation = Array2::<Complex64>::zeros((polar_grid.nodes_no, polar_grid.nodes_no));

    for j in 0..polar_grid.nodes_no {
        let pl = associated_legendre_polynomials(l_max as usize, omega, polar_grid.nodes[j].cos());
        for i in 0..polar_grid.nodes_no {
            transformation[[i, j]] = Complex64::from(
                normalization(l[i] + omega.unsigned_abs() as u32, omega as i32) * polar_grid.weights[j].sqrt() * pl[i]
            );
        }
    }

    let mat = transformation.view().into_faer_complex().transpose(); 
    let q = mat.qr().compute_q();
    let mut transformation = q.transpose().into_ndarray_complex().to_owned();

    let mut inverse_transformation = transformation.clone().reversed_axes();

    for j in 0..polar_grid.nodes_no {
        for i in 0..polar_grid.nodes_no {
            transformation[[i, j]] *= polar_grid.weights[j].sqrt();
            inverse_transformation[[j, i]] /= polar_grid.weights[j].sqrt()
        }
    }

    legendre_diagonalization.set_diagonalization_matrix(transformation, inverse_transformation);

    legendre_diagonalization
}

/// Creates diagonalization to Associated Legendre polynomials eigenbasis for given polar_grid and omega_grid
pub fn associated_legendre_operator(polar_grid: &Grid, omega_grid: &Grid) -> StateMatrixTransformation {
    let l_max = polar_grid.nodes_no as i64 - 1;

    let l_grid = Grid::new_linear_countable(
        "red_angular_momentum",
        0.0,
        l_max as f64,
        l_max as usize + 1,
        polar_grid.dimension_no,
    );

    let mut legendre_diagonalization = StateMatrixTransformation::new(omega_grid.dimension_no, &polar_grid, l_grid);


    let mut transformations = Vec::with_capacity(omega_grid.nodes_no);
    let mut inverses = Vec::with_capacity(omega_grid.nodes_no);
    for &omega in &omega_grid.nodes {
        let transformation = associated_legendre_diagonalization_operator(polar_grid, omega as isize);
        let [transformation, inverse] = transformation.get_diagonalization_matrices();

        transformations.push(transformation);
        inverses.push(inverse);
    }

    legendre_diagonalization.set_diagonalization_matrices(transformations, inverses);

    legendre_diagonalization
}
