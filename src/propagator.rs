pub mod transformation;
pub mod fft_transformation;
pub mod matrix_transformation;
pub mod n_dim_propagator;
pub mod one_dim_propagator;
pub mod propagator_factory;

use ndarray::Dimension;

use crate::{loss_checker::LossChecker, wave_function::WaveFunction};

pub trait Propagator<N: Dimension> {
    fn apply(&mut self, wave_function: &mut WaveFunction<N>);

    fn loss(&self) -> &Option<LossChecker>;

    fn loss_reset(&mut self);
}
