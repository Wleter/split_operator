
use crate::wave_function::WaveFunction;

/// Trait for diagonalization of operator, transforming [`WaveFunction`] in give space and grids to operator eigenspace.
pub trait Transformation {
    /// Diagonalizes given [`WaveFunction`] to operator eigenspace.
    fn transform(&mut self, wave_function: &mut WaveFunction);

    /// Return [`WaveFunction`] to original space.
    fn inverse_transform(&mut self, wave_function: &mut WaveFunction);
}

/// Define whether diagonalize or inverse_diagonalize is performed first
pub enum Order {
    Normal,
    InverseFirst
}