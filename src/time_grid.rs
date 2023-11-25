use num::complex::Complex64;

/// Time grid for the propagation of the wave function.
/// - `step` is the time in au for each step.
/// - `step_no` is the number of steps in the propagation.
#[derive(Clone, Default)]
pub struct TimeGrid {
    pub step: f64,
    pub step_no: usize,
    pub im_time: bool,
}

/// Enum for the type of step in the split-operator method. Available options are:
/// - `Full` for a full step.
/// - `Half` for a half step.
pub enum TimeStep {
    Full,
    Half,
}

/// Select the step size from [`TimeGrid`] for the propagation.
pub fn select_step(step: TimeStep, time: &TimeGrid) -> Complex64 {
    let time_step = match step {
        TimeStep::Full => time.step,
        TimeStep::Half => time.step / 2.0,
    };

    if time.im_time {
        -time_step * Complex64::i()
    } else {
        Complex64::from(time_step)
    }
}
