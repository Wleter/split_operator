use crate::{loss_saver::LossSaver, time_grid::TimeGrid, wave_function::WaveFunction};

/// Checks the loss of norm of the wave function.
/// `LossChecker` is used to check loss of norm of the wave function during the use of `Propagator` on wave function if needed.
/// It also stores the cumulative loss of norm during checks.
#[derive(Clone)]
pub struct LossChecker {
    pub name: String,
    loss: f64,
    current_norm: f64,
    loss_saver: Option<LossSaver>,
}

impl LossChecker {
    /// Creates new `LossChecker` with given name.
    pub fn new(name: &str) -> LossChecker {
        LossChecker {
            name: name.to_string(),
            loss: 0.0,
            current_norm: 1.0,
            loss_saver: None,
        }
    }

    pub fn new_with_saver(
        name: &str,
        frames_no: usize,
        filename: String,
        time_grid: &TimeGrid,
    ) -> LossChecker {
        LossChecker {
            name: name.to_string(),
            loss: 0.0,
            current_norm: 1.0,
            loss_saver: Some(LossSaver::new(filename, frames_no, time_grid)),
        }
    }

    /// Return the cumulative loss of norm from checks.
    pub fn loss(&self) -> f64 {
        if let Some(loss_saver) = &self.loss_saver {
            loss_saver.save()
        }

        self.loss
    }

    /// Check the norm of the wave function before possible norm change.
    pub fn check_before(&mut self, wave_function: &mut WaveFunction) {
        self.current_norm = wave_function.norm();
    }

    /// Check the norm of the wave function after possible norm change.
    pub fn check_after(&mut self, wave_function: &mut WaveFunction) {
        let new_norm = wave_function.norm();
        self.loss += self.current_norm - new_norm;

        if let Some(loss_saver) = &mut self.loss_saver {
            loss_saver.monitor(self.loss);
        }

        self.current_norm = new_norm;
    }

    /// Reset the cumulative loss of norm.
    pub fn reset(&mut self) {
        self.loss = 0.0;
    }
}
