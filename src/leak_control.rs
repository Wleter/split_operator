
use crate::{control::Control, loss_checker::LossChecker, wave_function::WaveFunction};

/// Controls the norm of the wave function during propagation.
/// Used when transformations of the wave function loss some of its norm due to numerical stability.
#[derive(Clone)]
pub struct LeakControl {
    norm: f64,
    loss_checked: Option<LossChecker>,
}

impl LeakControl {
    /// Creates new `LeakControl` with given example wave function.
    pub fn new() -> Self {
        LeakControl {
            norm: 0.0,
            loss_checked: None,
        }
    }

    pub fn add_loss_checker(&mut self, loss_checker: LossChecker) {
        self.loss_checked = Some(loss_checker);
    }
}

impl Control for LeakControl {
    fn name(&self) -> &str {
        "LeakControl"
    }

    fn first_half(&mut self, wave_function: &mut WaveFunction) {
        self.norm = wave_function.norm();

        if let Some(loss_checker) = &mut self.loss_checked {
            loss_checker.check_before(wave_function);
        }
    }

    fn second_half(&mut self, wave_function: &mut WaveFunction) {
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
