use super::{n_dim_propagator::NDimPropagator, one_dim_propagator::OneDimPropagator};
use crate::{
    grid::Grid,
    time_grid::{select_step, TimeGrid, TimeStep},
    wave_function::WaveFunction,
};
use ndarray::{Array, Array1, Dimension};
use num::complex::Complex64;

/// Creates propagator from one dimensional hamiltonian acting on given [`Grid`] with given [`TimeGrid`] and [`Step`].
pub fn one_dim_into_propagator<N: Dimension>(
    example_wave_function: &WaveFunction<N>,
    hamiltonian: Array1<f64>,
    grid: &Grid,
    time: &TimeGrid,
    step: TimeStep,
) -> OneDimPropagator<N> {
    let dt = select_step(step, time);

    let mut propagator =
        OneDimPropagator::new(example_wave_function, grid.nodes_no, grid.dimension_no);
    propagator.set_operator(hamiltonian.map(|x| Complex64::exp(-Complex64::i() * x * dt)));

    propagator
}

/// Creates propagator from n dimensional hamiltonian with given [`TimeGrid`] and [`Step`].
pub fn n_dim_into_propagator<N: Dimension>(
    example_wave_function: &WaveFunction<N>,
    hamiltonian: Array<f64, N>,
    time: &TimeGrid,
    step: TimeStep,
) -> NDimPropagator<N> {
    let dt = select_step(step, time);

    let mut propagator = NDimPropagator::new(example_wave_function);
    propagator.set_operator(hamiltonian.map(|x| Complex64::exp(-Complex64::i() * x * dt)));

    propagator
}

/// Creates propagator from n dimensional complex hamiltonian with given [`TimeGrid`] and [`Step`].
pub fn complex_n_dim_into_propagator<N: Dimension>(
    example_wave_function: &WaveFunction<N>,
    hamiltonian: Array<Complex64, N>,
    time: &TimeGrid,
    step: TimeStep,
) -> NDimPropagator<N> {
    let dt = select_step(step, time);

    let mut propagator = NDimPropagator::new(example_wave_function);
    propagator.set_operator(hamiltonian.map(|x| Complex64::exp(-Complex64::i() * x * dt)));

    propagator
}
