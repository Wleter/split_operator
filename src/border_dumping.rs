use std::f64::consts::PI;

use ndarray::Array1;
use num::complex::Complex64;

use crate::{
    control::Control,
    grid::Grid,
    loss_checker::LossChecker,
    propagator::{one_dim_propagator::OneDimPropagator, Propagator},
    wave_function::WaveFunction,
};

pub fn dumping_end(mask_width: f64, mask_end: f64, grid: &Grid) -> Array1<Complex64> {
    let r_max = grid.nodes.last().unwrap();

    let dumping = grid
        .nodes
        .clone()
        .into_iter()
        .map(|x| {
            if x < r_max - mask_width {
                Complex64::from(1.0)
            } else if x > r_max - mask_end {
                Complex64::from(0.0)
            } else {
                Complex64::from((PI / 2.0 * (r_max - x) / mask_width).sin())
            }
        })
        .collect::<Vec<Complex64>>();

    Array1::from(dumping)
}

#[derive(Clone)]
pub struct BorderDumping {
    operator: OneDimPropagator,
    loss_checked: Option<LossChecker>,
}

impl BorderDumping {
    pub fn new(mask: Array1<Complex64>, grid: &Grid) -> Self {
        let mut operator = OneDimPropagator::new(mask.len(), grid.dimension_no);
        operator.set_operator(mask);

        BorderDumping {
            operator,
            loss_checked: None,
        }
    }

    pub fn add_loss_checker(&mut self, loss_checker: LossChecker) {
        self.loss_checked = Some(loss_checker);
    }
}

impl Control for BorderDumping {
    fn name(&self) -> &str {
        "BorderDumping"
    }

    fn first_half(&mut self, wave_function: &mut WaveFunction) {
        if let Some(loss_checker) = &mut self.loss_checked {
            loss_checker.check_before(wave_function);
        }

        self.operator.apply(wave_function);

        if let Some(loss_checker) = &mut self.loss_checked {
            loss_checker.check_after(wave_function);
        }
    }

    fn second_half(&mut self, wave_function: &mut WaveFunction) {
        if let Some(loss_checker) = &mut self.loss_checked {
            loss_checker.check_before(wave_function);
        }

        self.operator.apply(wave_function);

        if let Some(loss_checker) = &mut self.loss_checked {
            loss_checker.check_after(wave_function);
        }
    }

    fn loss(&self) -> &Option<LossChecker> {
        &self.loss_checked
    }

    fn loss_mut(&mut self) -> &mut Option<LossChecker> {
        &mut self.loss_checked
    }
}
