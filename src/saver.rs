use dyn_clone::DynClone;
use ndarray::Dimension;

use crate::wave_function::WaveFunction;

/// Trait for monitoring the state of the wave function throughout the simulation and saving it.
pub trait Saver<N: Dimension>: DynClone {
    /// Monitor the state of the wave function.
    fn monitor(&mut self, wave_function: &mut WaveFunction<N>);

    /// Save collected data.
    fn save(&self) -> Result<(), &str>;

    /// Reset collected data
    fn reset(&mut self);
}

impl<N: Dimension> Clone for Box<dyn Saver<N>> {
    fn clone(&self) -> Self {
        dyn_clone::clone_box(&**self)
    }
}
