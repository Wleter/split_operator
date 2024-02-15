use dyn_clone::DynClone;
use enum_flags::enum_flags;
use ndarray::Dimension;

use crate::{loss_checker::LossChecker, wave_function::WaveFunction};

/// Trait for controlling the wave function during propagation.
pub trait Control<N: Dimension>: DynClone {
    /// Returns the name of the control.
    fn name(&self) -> &str;
    /// Checks and controls the wave function on the first half of the time step.
    fn first_half(&mut self, wave_function: &mut WaveFunction<N>);
    /// Checks and controls the wave function on the second half of the time step.
    fn second_half(&mut self, wave_function: &mut WaveFunction<N>);

    fn loss(&self) -> &Option<LossChecker>;

    fn loss_mut(&mut self) -> &mut Option<LossChecker>;
}

impl<N: Dimension> Clone for Box<dyn Control<N>> {
    fn clone(&self) -> Self {
        dyn_clone::clone_box(&**self)
    }
}

#[repr(u8)]
#[enum_flags]
#[derive(Clone)]
pub enum Apply {
    None = 0,
    FirstHalf = 1,
    SecondHalf = 2,
}
