use std::marker::PhantomData;

use ndarray::{Array, Dimension};
use num::complex::Complex64;

use crate::{loss_checker::LossChecker, wave_function::WaveFunction};

use super::Propagator;

#[derive(Clone)]
pub struct NDimPropagator<N: Dimension> {
    operator: Array<Complex64, N>,
    phantom: PhantomData<N>,
    loss_checked: Option<LossChecker>,
}

impl<N: Dimension> NDimPropagator<N> {
    pub fn new(example_wave_function: &WaveFunction<N>) -> NDimPropagator<N> {
        NDimPropagator {
            operator: Array::<Complex64, N>::zeros(example_wave_function.array.raw_dim()),
            phantom: PhantomData,
            loss_checked: None,
        }
    }

    pub fn set_operator(&mut self, operator: Array<Complex64, N>) {
        assert!(operator.shape() == self.operator.shape());

        self.operator = operator;
    }

    pub fn add_operator(&mut self, operator: Array<Complex64, N>) {
        assert!(operator.shape() == self.operator.shape());

        self.operator *= &operator;
    }

    fn apply_unchecked(&self, wave_function: &mut WaveFunction<N>) {
        wave_function.change_observer.possible_norm_change = true;

        wave_function.array *= &self.operator;
    }

    pub fn set_loss_checked(&mut self, loss_checked: LossChecker) {
        self.loss_checked = Some(loss_checked);
    }
}

impl<N: Dimension> Propagator<N> for NDimPropagator<N> {
    fn apply(&mut self, wave_function: &mut WaveFunction<N>) {
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
