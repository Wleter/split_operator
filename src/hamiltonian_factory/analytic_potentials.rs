/// Returns value of dispersion potential on given distance `r` for potential with dispersion `n` power distance and strength `cn`.
pub fn dispersion(r: f64, n: i32, cn: f64) -> f64 {
    let value = cn * r.powi(n);

    if value.is_nan() {
        f64::INFINITY
    } else {
        value
    }
}

/// Returns value of Lennard-Jones potential on given distance `r` for potential with parameters `d6`, `r6`.
pub fn lennard_jones(r: f64, d6: f64, r6: f64) -> f64 {
    let r_ratio = (r6 / r).powi(6);
    let value = d6 * r_ratio * (r_ratio - 2.0);

    if value.is_nan() {
        f64::INFINITY
    } else {
        value
    }
}

pub fn harmonic(r: f64, r0: f64, mass: f64, omega: f64) -> f64 {
    let value = 0.5 * mass * omega.powi(2) * (r - r0).powi(2);

    if value.is_nan() {
        f64::INFINITY
    } else {
        value
    }
}
