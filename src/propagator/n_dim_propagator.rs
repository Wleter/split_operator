use ndarray::{ArrayD, IxDyn};
use num::complex::Complex64;

use crate::{loss_checker::LossChecker, wave_function::WaveFunction};

use super::Propagator;

#[derive(Clone)]
pub struct NDimPropagator {
    operator: ArrayD<Complex64>,
    loss_checked: Option<LossChecker>,
}

impl NDimPropagator {
    pub fn new() -> NDimPropagator {
        NDimPropagator {
            operator: ArrayD::zeros(IxDyn(&[1])),
            loss_checked: None,
        }
    }

    pub fn set_operator(&mut self, operator: ArrayD<Complex64>) {
        self.operator = operator;
    }

    pub fn add_operator(&mut self, operator: ArrayD<Complex64>) {
        assert!(operator.shape() == self.operator.shape());

        self.operator *= &operator;
    }

    fn apply_unchecked(&self, wave_function: &mut WaveFunction) {
        wave_function.change_observer.possible_norm_change = true;

        wave_function.array *= &self.operator;
    }

    pub fn set_loss_checked(&mut self, loss_checked: LossChecker) {
        self.loss_checked = Some(loss_checked);
    }
}

impl Propagator for NDimPropagator {
    fn apply(&mut self, wave_function: &mut WaveFunction) {
        if let Some(loss_checker) = &mut self.loss_checked {
            loss_checker.check_before(wave_function);
        }

        self.apply_unchecked(wave_function);

        if let Some(loss_checker) = &mut self.loss_checked {
            loss_checker.check_after(wave_function);
        }
    }

    fn loss(&self) -> &Option<LossChecker> {
        &self.loss_checked
    }

    fn loss_reset(&mut self) {
        if let Some(loss_checker) = &mut self.loss_checked {
            loss_checker.reset();
        }
    }
}
