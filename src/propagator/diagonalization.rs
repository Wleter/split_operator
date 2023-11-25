use dyn_clone::DynClone;
use ndarray::Dimension;

use crate::wave_function::WaveFunction;

/// Trait for diagonalization of operator, transforming [`WaveFunction`] in give space and grids to operator eigenspace.
pub trait Diagonalization<N: Dimension>: DynClone {
    /// Diagonalizes given [`WaveFunction`] to operator eigenspace.
    fn diagonalize(&mut self, wave_function: &mut WaveFunction<N>);

    /// Return [`WaveFunction`] to original space.
    fn inverse_diagonalize(&mut self, wave_function: &mut WaveFunction<N>);
}

impl<N: Dimension> Clone for Box<dyn Diagonalization<N>> {
    fn clone(&self) -> Self {
        dyn_clone::clone_box(&**self)
    }
}
