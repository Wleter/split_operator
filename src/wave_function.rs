use ndarray::{Array, Array1, Axis, Dimension, IxDyn, Zip};
use num::complex::Complex64;

use crate::change_observer::ChangeObserver;
use crate::grid::Grid;

/// Struct to hold information about a wave function on actual time step.
/// It contains the wave function array in the representation of grids.
/// `change_observer` is used to observe possible norm and grid changes during the propagation.
#[derive(Clone, Default)]
pub struct WaveFunction<N: Dimension> {
    pub array: Array<Complex64, N>,
    pub grids: Vec<Grid>,

    pub change_observer: ChangeObserver,

    /// Array of weights for calculating wave function norm.
    weight_amplitude_array: Array<Complex64, N>,
}

impl<N: Dimension> WaveFunction<N> {
    /// Creates new wave function from wave function array and grids.
    pub fn new(wave_function_array: Array<Complex64, N>, grids: Vec<Grid>) -> WaveFunction<N> {
        let mut weight_amplitude_array: Array<Complex64, N> =
            Array::ones(wave_function_array.dim());

        for axis in 0..wave_function_array.ndim() {
            weight_amplitude_array
                .lanes_mut(Axis(axis))
                .into_iter()
                .for_each(|mut lane| {
                    lane.assign(
                        &(&lane
                            * Array::from(grids[axis].weights.to_vec())
                                .mapv(|x| Complex64::from(x.sqrt()))),
                    );
                });
        }

        let change_observer = ChangeObserver::new(&grids);

        WaveFunction {
            array: wave_function_array,
            grids,
            change_observer,
            weight_amplitude_array,
        }
    }

    /// Updates the weight amplitude array from current grids.
    fn update_weight_amplitude_array(&mut self) {
        self.weight_amplitude_array = Array::ones(self.array.dim());

        for axis in 0..self.array.ndim() {
            self.weight_amplitude_array
                .lanes_mut(Axis(axis))
                .into_iter()
                .for_each(|mut lane| {
                    lane.assign(
                        &(&lane
                            * Array::from(self.grids[axis].weights.to_vec())
                                .mapv(|x| Complex64::from(x.sqrt()))),
                    );
                });
        }
    }

    /// Returns the norm of the wave function.
    pub fn norm(&mut self) -> f64 {
        if self.change_observer.possible_norm_change == false {
            return self.change_observer.last_norm();
        }

        if self.change_observer.has_grid_changed(&self.grids) {
            self.update_weight_amplitude_array();
            self.change_observer.observe_grid(&self.grids);
        }

        let norm = Zip::from(&self.array)
            .and(&self.weight_amplitude_array)
            .fold(0.0, |acc, x, y| acc + x.norm_sqr() * y.norm_sqr());

        self.change_observer.observe_norm(norm);

        norm
    }

    pub fn dot(&mut self, other: &mut Self) -> Complex64 {
        let norm_1 = self.norm();
        let norm_2 = other.norm();

        assert!(self.weight_amplitude_array == other.weight_amplitude_array);

        let dot_prod = Zip::from(&self.array)
            .and(&other.array)
            .and(&self.weight_amplitude_array)
            .fold(Complex64::new(0.0, 0.0), |acc, x, y, w1| {
                acc + x * y.conj() * w1.norm_sqr()
            });

        dot_prod / (norm_1 * norm_2).sqrt()
    }

    /// Sets the norm of the wave function to `new_norm`.
    pub fn normalize(&mut self, new_norm: f64) {
        let norm = self.norm();
        self.array *= Complex64::from((new_norm / norm).sqrt());

        self.change_observer.observe_norm(new_norm);
    }

    /// Returns the density of the wave function on actual `grids`.
    pub fn density(&mut self) -> Array<f64, N> {
        let density_vec: Vec<f64> = self.array.iter().map(|x| x.norm_sqr()).collect();

        let density = Array::from_shape_vec(self.array.raw_dim(), density_vec).unwrap();

        density
    }

    /// Return the density of the wave function on actual `grids` along given `axis`.
    pub fn state_density(&mut self, axis: usize) -> Array1<f64> {
        if self.change_observer.has_grid_changed(&self.grids) {
            self.update_weight_amplitude_array();
            self.change_observer.observe_grid(&self.grids);
        }

        let density = self.density();

        if density.ndim() == 1 {
            return density.into_dimensionality().unwrap();
        }

        density
            .into_dimensionality::<IxDyn>()
            .unwrap()
            .axis_iter_mut(Axis(axis))
            .zip(
                self.weight_amplitude_array
                    .view()
                    .into_dimensionality::<IxDyn>()
                    .unwrap()
                    .axis_iter(Axis(axis)),
            )
            .map(|(lane, weight_lane)| {
                lane.iter()
                    .zip(weight_lane.iter())
                    .map(|(x, w)| x * w.norm_sqr())
                    .sum()
            })
            .collect()
    }
}

/// Returns value of a gaussian distribution with momentum `momentum` and position `x0` with width `sigma` at position `x`.
pub fn gaussian_distribution(x: f64, x0: f64, sigma: f64, momentum: f64) -> Complex64 {
    (-((x - x0) / (2.0 * sigma)).powi(2) - Complex64::i() * (x - x0) * momentum).exp()
}
