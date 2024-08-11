use ndarray::{Array1, Axis};
use num::complex::Complex64;
use rayon::prelude::*;

use crate::{loss_checker::LossChecker, wave_function::WaveFunction};

use super::Propagator;

#[derive(Clone)]
pub struct OneDimPropagator {
    dimension_no: usize,
    operator: Array1<Complex64>,
    loss_checked: Option<LossChecker>,
}

impl OneDimPropagator {
    pub fn new(shape: usize, dimension_no: usize) -> OneDimPropagator {
        OneDimPropagator {
            dimension_no,
            operator: Array1::<Complex64>::zeros(shape),
            loss_checked: None,
        }
    }

    pub fn set_operator(&mut self, operator: Array1<Complex64>) {
        assert!(operator.shape()[0] == self.operator.shape()[0]);

        self.operator = operator;
    }

    pub fn add_operator(&mut self, operator: Array1<Complex64>) {
        assert!(operator.shape()[0] == self.operator.shape()[0]);

        self.operator *= &operator;
    }

    fn apply_unchecked(&self, wave_function: &mut WaveFunction) {
        wave_function.change_observer.possible_norm_change = true;

        wave_function
            .array
            .lanes_mut(Axis(self.dimension_no))
            .into_iter()
            .par_bridge()
            .for_each(|mut lane| lane *= &self.operator);
    }

    pub fn set_loss_checked(&mut self, loss_checked: LossChecker) {
        self.loss_checked = Some(loss_checked);
    }
}

impl Propagator for OneDimPropagator {
    #[inline(always)]
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
