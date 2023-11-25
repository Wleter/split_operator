pub mod diagonalization;
pub mod fft_diagonalization;
pub mod matrix_diagonalization;
pub mod n_dim_propagator;
pub mod one_dim_propagator;
pub mod propagator_factory;

use dyn_clone::DynClone;
use ndarray::Dimension;

use crate::{loss_checker::LossChecker, wave_function::WaveFunction};

pub trait Propagator<N: Dimension>: DynClone {
    fn apply(&mut self, wave_function: &mut WaveFunction<N>);

    fn loss(&self) -> &Option<LossChecker>;

    fn loss_reset(&mut self);
}

impl<N: Dimension> Clone for Box<dyn Propagator<N>> {
    fn clone(&self) -> Self {
        dyn_clone::clone_box(&**self)
    }
}
