use ndarray::Dimension;

use crate::wave_function::WaveFunction;

/// Trait for diagonalization of operator, transforming [`WaveFunction`] in give space and grids to operator eigenspace.
pub trait Transformation<N: Dimension> {
    /// Diagonalizes given [`WaveFunction`] to operator eigenspace.
    fn transform(&mut self, wave_function: &mut WaveFunction<N>);

    /// Return [`WaveFunction`] to original space.
    fn inverse_transform(&mut self, wave_function: &mut WaveFunction<N>);
}

/// Define whether diagonalize or inverse_diagonalize is performed first
pub enum Order {
    Normal,
    InverseFirst
}