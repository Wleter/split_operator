use ndarray::Dimension;

use crate::wave_function::WaveFunction;

/// Trait for monitoring the state of the wave function throughout the simulation and saving it.
pub trait Saver<N: Dimension> {
    /// Monitor the state of the wave function.
    fn monitor(&mut self, wave_function: &mut WaveFunction<N>);

    /// Save collected data.
    fn save(&self) -> Result<(), &str>;

    /// Reset collected data
    fn reset(&mut self);
}
