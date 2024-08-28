use ndarray::{ Array2, Axis };
use num::complex::Complex64;
use rayon::prelude::*;

use crate::{loss_checker::LossChecker, wave_function::WaveFunction};

use super::Propagator;

#[derive(Clone)]
pub struct NonDiagPropagator {
    operators: Vec<Array2<Complex64>>,
    dimension_no: usize,
    loss_checked: Option<LossChecker>,
}

impl NonDiagPropagator {
    pub fn new(dimension_no: usize) -> Self {
        Self {
            operators: Vec::new(),
            dimension_no: dimension_no,
            loss_checked: None,
        }
    }

    pub fn set_operators(&mut self, operators: Vec<Array2<Complex64>>) {
        self.operators = operators;
    }

    fn apply_unchecked(&self, wave_function: &mut WaveFunction) {
        wave_function.change_observer.possible_norm_change = true;

        self.operators.iter()
            .zip(wave_function.array.lanes_mut(Axis(self.dimension_no)))
            .par_bridge()
            .into_par_iter()
            .for_each(|(op, mut lane)| lane.assign(&op.dot(&lane)));
    }

    pub fn set_loss_checked(&mut self, loss_checked: LossChecker) {
        self.loss_checked = Some(loss_checked);
    }
}

impl Propagator for NonDiagPropagator {
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
