use ndarray::{Array, Array2, Dimension};

use crate::{grid::Grid, wave_function::WaveFunction};

/// Broadcasts two-dimensional operator to n-dimensional operator.
pub fn two_dim_into_n_dim_operator<N: Dimension>(
    example_wave_function: &WaveFunction<N>,
    hamiltonian: Array2<f64>,
    grid1: &Grid,
    grid2: &Grid,
) -> Array<f64, N> {
    let mut n_dim_operator = Array::zeros(example_wave_function.array.raw_dim());

    n_dim_operator.swap_axes(n_dim_operator.ndim() - 2, grid1.dimension_no);
    n_dim_operator.swap_axes(n_dim_operator.ndim() - 1, grid2.dimension_no);

    n_dim_operator += &hamiltonian;

    n_dim_operator.swap_axes(n_dim_operator.ndim() - 1, grid2.dimension_no);
    n_dim_operator.swap_axes(n_dim_operator.ndim() - 2, grid1.dimension_no);

    n_dim_operator
}
