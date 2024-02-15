use std::marker::PhantomData;

use ndarray::Dimension;

use crate::{control::Control, loss_checker::LossChecker, wave_function::WaveFunction};

/// Controls the norm of the wave function during propagation.
/// Used when transformations of the wave function loss some of its norm due to numerical stability.
#[derive(Clone)]
pub struct LeakControl<N: Dimension> {
    norm: f64,
    phantom: PhantomData<N>,
    loss_checked: Option<LossChecker>,
}

impl<N: Dimension> LeakControl<N> {
    /// Creates new `LeakControl` with given example wave function.
    pub fn new(_example_wave_function: &WaveFunction<N>) -> Self {
        LeakControl {
            norm: 0.0,
            phantom: PhantomData,
            loss_checked: None,
        }
    }

    pub fn add_loss_checker(&mut self, loss_checker: LossChecker) {
        self.loss_checked = Some(loss_checker);
    }
}

impl<N: Dimension> Control<N> for LeakControl<N> {
    fn name(&self) -> &str {
        "LeakControl"
    }

    fn first_half(&mut self, wave_function: &mut WaveFunction<N>) {
        self.norm = wave_function.norm();

        if let Some(loss_checker) = &mut self.loss_checked {
            loss_checker.check_before(wave_function);
        }
    }

    fn second_half(&mut self, wave_function: &mut WaveFunction<N>) {
        if let Some(loss_checker) = &mut self.loss_checked {
            loss_checker.check_after(wave_function);
        }

        wave_function.normalize(self.norm);
    }

    fn loss(&self) -> &Option<LossChecker> {
        &self.loss_checked
    }

    fn loss_mut(&mut self) -> &mut Option<LossChecker> {
        &mut self.loss_checked
    }
}
