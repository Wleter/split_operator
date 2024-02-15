use std::{
    f64::consts::PI,
    fmt::{Display, Formatter},
};

use crate::{
    border_dumping::{dumping_end, BorderDumping},
    control::Apply,
    grid::Grid,
    hamiltonian_factory::{
        analytic_potentials::dispersion, kinetic_operator::kinetic_hamiltonian,
        legendre_diagonalization::legendre_diagonalization_operator,
        rotational_operator::rotational_hamiltonian,
    },
    leak_control::LeakControl,
    loss_checker::LossChecker,
    propagation::Propagation,
    propagator::{
        fft_diagonalization::FFTDiagonalization,
        propagator_factory::{
            complex_n_dim_into_propagator, n_dim_into_propagator, one_dim_into_propagator,
        },
    },
    special_functions::gauss_legendre_quad,
    time_grid::{TimeGrid, TimeStep},
    wave_function::{gaussian_distribution, WaveFunction},
    wave_function_saver::{StateSaver, WaveFunctionSaver},
};
use ndarray::{Array1, Array2, Ix2};
use num::complex::Complex64;
use quantum::{particle_factory::{create_atom, create_molecule}, particles::Particles, units::energy_units::{Energy, Kelvin}};
use scilib::math::polynomial::Poly;

use crate::potential_reader::load_potential;

/// Struct representing a Ne-OCS system for propagation using the split-operator method.
/// It heavily uses the [`Propagation<N>`] struct from `split_operator` crate that implements general split-operator method.
/// The system is built using the builder pattern.
/// Building the split-operator method is done by appending operations in the order they should be performed,
/// last appended operation should be the central one with full step.
#[derive(Clone, Default)]
pub struct NeOcs {
    propagation: Propagation<Ix2>,
    pub collision_params: Particles,
    r_grid: Grid,
    polar_grid: Grid,
    wave_function: WaveFunction<Ix2>,
    angular_grid: Grid,

    xpi_position: Option<usize>,
    potential_bsigma_position: Option<usize>,
    ang_kinetic_position: Option<usize>,
    scalings: [Scaling; 4],
}

impl NeOcs {
    /// Sets the time grid for the propagation with the given time step `time_step` and number of steps `steps_no`.
    pub fn set_time_grid(&mut self, time_step: f64, steps_no: usize) {
        self.propagation.set_time_grid(TimeGrid {
            step: time_step,
            step_no: steps_no,
            im_time: false
        });
    }

    /// Sets the radial grid for the propagation with the given start `r_start` and end `r_stop` points and number of points `r_points_no`.
    pub fn set_radial_grid(&mut self, r_start: f64, r_end: f64, r_points_no: usize) {
        self.r_grid = Grid::new_linear_continuos("radial", r_start, r_end, r_points_no, 0);
    }

    /// Sets the polar grid for the propagation with the given number of points.
    pub fn set_polar_grid(&mut self, polar_points_no: usize) {
        let legendre_quad = gauss_legendre_quad(polar_points_no);

        let polar_grid = Grid::new_custom(
            "polar",
            legendre_quad.nodes.iter().map(|x| x.acos()).collect(),
            legendre_quad.weights,
            1,
        );

        let angular_grid = Grid::new_linear_countable(
            "angular_momentum",
            0.0,
            (polar_points_no - 1) as f64,
            polar_points_no,
            1,
        );

        self.polar_grid = polar_grid;
        self.angular_grid = angular_grid;
    }

    /// Sets parameters of collision such as energy, initial angular momentum `j_init`, initial projection of angular momentum `omega_init` and total angular momentum `J_tot`.
    pub fn set_collision_params(
        &mut self,
        energy_kelvin: f64,
        j_init: usize,
        omega_init: usize,
        j_tot: usize,
    ) {
        let energy = Energy(energy_kelvin, Kelvin);
        let ne = create_atom("Ne").unwrap();
        let ocs = create_molecule("OCS").unwrap();
        self.collision_params = Particles::new_pair(ne, ocs, energy);
        self.collision_params.internals
            .insert_value("j_init", j_init as f64)
            .insert_value("omega_init", omega_init as f64)
            .insert_value("j_total", j_tot as f64);
    }

    /// Sets the initial wave function as a wave packet with the given position `r0` and dispersion `r_sigma` and current set collision parameters.
    pub fn set_wave_function(&mut self, r0: f64, r_sigma: f64) {
        let mut wave_function_array =
            Array2::<Complex64>::ones((self.r_grid.nodes_no, self.polar_grid.nodes_no));

        let momentum =
            (2.0 * self.collision_params.red_mass() * self.collision_params.internals.get_value("energy")).sqrt();

        let r_init = self
            .r_grid
            .nodes
            .iter()
            .map(|x| gaussian_distribution(*x, r0, r_sigma, momentum))
            .collect::<Vec<Complex64>>();

        let polar_init = self
            .polar_grid
            .nodes
            .iter()
            .map(|x| {
                Poly::gen_legendre(
                    self.collision_params.internals.get_value("j_init") as usize,
                    self.collision_params.internals.get_value("omega_init") as isize,
                )
                .compute(x.cos())
            })
            .collect::<Vec<f64>>();

        for i in 0..self.r_grid.nodes_no {
            for j in 0..self.polar_grid.nodes_no {
                wave_function_array[[i, j]] = r_init[i] * polar_init[j];
            }
        }

        let mut wave_function = WaveFunction::new(
            wave_function_array,
            vec![self.r_grid.clone(), self.polar_grid.clone()],
        );
        wave_function.normalize(1.0);

        self.wave_function = wave_function;
        self.propagation
            .set_wave_function(self.wave_function.clone());
    }

    /// Sets the wave function to propagate the same as set using `set_wave_function`.
    pub fn reset_wave_function(&mut self) {
        assert!(self.wave_function.norm() == 1.0);
        self.propagation
            .set_wave_function(self.wave_function.clone());
    }

    // Append to the propagation the kinetic operator.
    pub fn set_kinetic_operator(&mut self) {
        let kinetic_operator = kinetic_hamiltonian(&self.r_grid, &self.collision_params);
        let kinetic_propagator = one_dim_into_propagator(
            &self.wave_function,
            kinetic_operator,
            &self.r_grid,
            &self.propagation.time_grid(),
            TimeStep::Full,
        );
        self.propagation
            .add_propagator(Box::new(kinetic_propagator));
    }

    /// Append to the propagation or replace existing angular kinetic operator.
    pub fn set_angular_operator(&mut self) {
        let rot_const = self.collision_params.particle_mut("OCS").unwrap().internals.get_value("rot_const");

        let angular_operator = rotational_hamiltonian(
            &self.r_grid,
            &self.polar_grid,
            &self.collision_params,
            rot_const,
        );

        let angular_propagator = n_dim_into_propagator(
            &self.wave_function,
            angular_operator,
            &self.propagation.time_grid(),
            TimeStep::Half,
        );

        if self.ang_kinetic_position == None {
            self.propagation
                .add_propagator(Box::new(angular_propagator));
            self.ang_kinetic_position = Some(self.propagation.operations_len() - 1)
        } else {
            self.propagation.replace_propagator(
                Box::new(angular_propagator),
                self.ang_kinetic_position.unwrap(),
            );
        }
    }

    /// Append to the propagation transformation to the angular momentum space.
    pub fn set_ang_momentum_space_transf(&mut self) {
        let angular_momentum_transformation =
            legendre_diagonalization_operator(&self.wave_function, &self.polar_grid);

        self.propagation
            .add_diagonalization(Box::new(angular_momentum_transformation), true);
    }

    /// Append to the propagation transformation to the momentum space.
    pub fn set_momentum_space_transf(&mut self) {
        let momentum_transformation =
            FFTDiagonalization::new(&self.wave_function, &self.r_grid, "polar_momentum");

        self.propagation
            .add_diagonalization(Box::new(momentum_transformation), true);
    }

    /// Set the scaling for intermolecular potential. See `Scaling` enum to see the options.
    pub fn set_potential_scaling(&mut self, potential_scaling: Scaling) {
        self.scalings[0] = potential_scaling;
    }

    /// Set the scaling for xpi gamma potential. See `Scaling` enum to see the options.
    pub fn set_xpi_scaling(&mut self, xpi_scaling: Scaling) {
        self.scalings[1] = xpi_scaling;
    }

    /// Set the scaling for bsigma gamma potential. See `Scaling` enum to see the options.
    pub fn set_bsigma_scaling(&mut self, bsigma_scaling: Scaling) {
        self.scalings[2] = bsigma_scaling;
    }

    /// Set the scaling for api gamma potential. See `Scaling` enum to see the options.
    pub fn set_api_scaling(&mut self, api_scaling: Scaling) {
        self.scalings[3] = api_scaling;
    }

    /// Append to operations or replace existing propagation the XPi gamma potential. `loss_check` determines if the loss from operator should be observed.
    /// Potential scaling can be set using beforehand `set_potential_scalings`.
    pub fn set_xpi_gamma(&mut self, loss_check: bool) {
        let path = std::env::current_dir()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();

        let mut xpi_gamma = load_potential(
            &path,
            "/potentials/XPi_gamma",
            &self.r_grid,
            &self.polar_grid,
            5,
            3,
            true,
        )
        .unwrap();
        xpi_gamma = self.scalings[1].scale(xpi_gamma, &self.r_grid, &self.polar_grid);

        let xpi_gamma = xpi_gamma.map(|x| -Complex64::i() * x / 2.0);

        let mut xpi_propagator = complex_n_dim_into_propagator(
            &self.wave_function,
            xpi_gamma,
            &self.propagation.time_grid(),
            TimeStep::Half,
        );
        if loss_check {
            xpi_propagator.set_loss_checked(LossChecker::new("xpi"));
        }

        if self.xpi_position == None {
            self.propagation.add_propagator(Box::new(xpi_propagator));
            self.xpi_position = Some(self.propagation.operations_len() - 1)
        } else {
            self.propagation
                .replace_propagator(Box::new(xpi_propagator), self.xpi_position.unwrap());
        }
    }

    /// Append to operations or replace existing propagation the BSigma gamma potential. `loss_check` determines if the loss from operator should be observed.
    /// Potentials scalings can be set using beforehand `set_potential_scalings`.
    pub fn set_potential_and_bsigma(&mut self, loss_check: bool) {
        let path = std::env::current_dir()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();

        let j_total = self.collision_params.internals.get_value("j_total");
        let omega_init = self.collision_params.internals.get_value("omega_init");

        let centrifugal_potential = Array1::<f64>::from_vec(
            self.r_grid
                .nodes
                .clone()
                .into_iter()
                .map(|x| {
                    dispersion(
                        x,
                        -2,
                        (j_total * (j_total + 1.0) - 2.0 * omega_init * omega_init)
                            / (2.0 * self.collision_params.red_mass()),
                    )
                })
                .collect(),
        );

        let mut raw_potential = load_potential(
            &path,
            "/potentials/potential",
            &self.r_grid,
            &self.polar_grid,
            5,
            5,
            false,
        )
        .unwrap();
        raw_potential = self.scalings[0].scale(raw_potential, &self.r_grid, &self.polar_grid);

        let mut bsigma_gamma = load_potential(
            &path,
            "/potentials/BSigma_gamma",
            &self.r_grid,
            &self.polar_grid,
            5,
            5,
            true,
        )
        .unwrap();
        bsigma_gamma = self.scalings[2].scale(bsigma_gamma, &self.r_grid, &self.polar_grid);

        let mut api_gamma = load_potential(
            &path,
            "/potentials/APi_gamma",
            &self.r_grid,
            &self.polar_grid,
            5,
            5,
            true,
        )
        .unwrap();
        api_gamma = self.scalings[3].scale(api_gamma, &self.r_grid, &self.polar_grid);

        let mut potential = Array2::<Complex64>::zeros(raw_potential.raw_dim());
        for i in 0..potential.raw_dim()[0] {
            for j in 0..potential.raw_dim()[1] {
                potential[[i, j]] = raw_potential[[i, j]] + centrifugal_potential[[i]]
                    - Complex64::i() * (bsigma_gamma[[i, j]] + api_gamma[[i, j]]) / 2.0;
            }
        }

        let mut potential_bsigma = complex_n_dim_into_propagator(
            &self.wave_function,
            potential,
            &self.propagation.time_grid(),
            TimeStep::Half,
        );

        if loss_check {
            potential_bsigma.set_loss_checked(LossChecker::new("bsigma"));
        }

        if self.potential_bsigma_position == None {
            self.propagation.add_propagator(Box::new(potential_bsigma));
            self.potential_bsigma_position = Some(self.propagation.operations_len() - 1)
        } else {
            self.propagation.replace_propagator(
                Box::new(potential_bsigma),
                self.potential_bsigma_position.unwrap(),
            );
        }
    }

    /// Appends to operations dumping border that erases wave function at the end of the radial grid.
    pub fn set_dumping_border(&mut self) {
        let mask_width = 5.0;
        let mask_end = 1.0;
        let mask = dumping_end(mask_width, mask_end, &self.r_grid);
        let dumping = BorderDumping::new(mask, &self.wave_function, &self.r_grid);

        self.propagation
            .add_control(Box::new(dumping), Apply::SecondHalf);
    }

    /// Appends to operations leak control that will preserve the norm of the wave function after appending.
    /// Useful for numerical instabilities such as fft transformations.
    pub fn set_leak_control(&mut self) {
        let leak_control = LeakControl::new(&self.wave_function);

        self.propagation
            .add_control(Box::new(leak_control), Apply::FirstHalf | Apply::SecondHalf);
    }

    /// Appends to operations the saver for the wave function in spatial grids.
    /// `frames_no` determines how many frames will be saved.
    /// Using `save_animations` at the end will save this animation.
    pub fn set_wave_animation(&mut self, frames_no: usize, prefix: &str) {
        let path = std::env::current_dir()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();

        let name = format!("/data/{prefix}_wave_animation");
        let wave_function_saver = WaveFunctionSaver::new(
            path.clone(),
            name,
            &self.propagation.time_grid(),
            &self.r_grid,
            &self.polar_grid,
            frames_no,
            &self.wave_function,
        );

        self.propagation
            .add_saver(Box::new(wave_function_saver), true);
    }

    /// Appends to operations the saver for the angular momentum density of wave function.
    /// `frames_no` determines how many frames will be saved.
    /// Using `save_animations` at the end will save this animation.
    pub fn set_legendre_animation(&mut self, frames_no: usize, prefix: &str) {
        let path = std::env::current_dir()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();

        let name = format!("/data/{prefix}_angular_animation");

        let angular_momentum_saver = StateSaver::new(
            path.clone(),
            name,
            &self.propagation.time_grid(),
            &self.angular_grid,
            frames_no,
            &self.wave_function,
        );

        self.propagation
            .add_saver(Box::new(angular_momentum_saver), true);
    }

    /// Propagate the initial wave function set by `set_wave_function` using appended operations
    /// for duration set by `set_time_grid`.
    pub fn propagate(&mut self) {
        self.propagation.propagate();
    }

    /// Print losses from operators that have `loss_check` set to true.
    pub fn print_losses(&mut self) {
        self.propagation.print_losses();
    }

    /// Get losses from operators that have `loss_check` set to true.
    pub fn get_losses(&mut self) -> Vec<f64> {
        self.propagation.get_losses()
    }

    /// Save animations set by `set_wave_animation` and `set_legendre_animation`.
    pub fn save_animations(&mut self) {
        self.propagation.savers_save();
    }

    /// Clear all appended operations.
    pub fn reset_operations(&mut self) {
        self.xpi_position = None;
        self.potential_bsigma_position = None;
        self.ang_kinetic_position = None;

        self.propagation.reset_operations();
    }

    /// Clear all appended savers - in this case animations.
    pub fn reset_savers_state(&mut self) {
        self.propagation.reset_savers_state();
    }

    /// Clear all losses from operators that have `loss_check` set to true.
    pub fn reset_losses(&mut self) {
        self.propagation.reset_losses();
    }
}

/// Enum responsible for scaling potentials. Available options are:
/// - `None` - no scaling
/// - `Factor(f64)` - scaling by a constant factor
/// - `Function(fn(f64, f64, f64) -> f64)` - scaling by a function of radius and angle with free parameters in Vec
#[derive(Clone)]
pub enum Scaling {
    None,
    Factor(f64),
    Function(Functions),
}

impl Scaling {
    /// Scale given potential.
    pub fn scale(&self, potential: Array2<f64>, r_grid: &Grid, polar_grid: &Grid) -> Array2<f64> {
        match self {
            Scaling::None => potential,
            Scaling::Factor(factor) => potential * (*factor),
            Scaling::Function(function) => {
                let scaling_array =
                    Array2::from_shape_fn((r_grid.nodes_no, polar_grid.nodes_no), |(i, j)| {
                        function.eval(&r_grid.nodes[i], &polar_grid.nodes[j])
                    });
                potential * &scaling_array
            }
        }
    }
}

impl Default for Scaling {
    fn default() -> Self {
        Scaling::None
    }
}

impl Display for Scaling {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Scaling::None => write!(f, "None"),
            Scaling::Factor(factor) => write!(f, "{factor}"),
            Scaling::Function(function) => {
                write!(f, "{function}")
            }
        }
    }
}

/// Enum with functions used for transforming potentials.
#[derive(Clone)]
pub enum Functions {
    /// Cosine function with scaling.
    CosTheta(f64),
    /// Stripe masking with start and end being in degrees.
    ThetaStripes(f64, f64),
}

impl Functions {
    /// Evaluate function at given radius and angle.
    pub fn eval(&self, _r: &f64, theta: &f64) -> f64 {
        match self {
            Functions::CosTheta(scaling) => 1.0 + scaling * theta.cos(),
            Functions::ThetaStripes(start_deg, end_deg) => {
                if start_deg / 180.0 * PI < *theta && end_deg / 180.0 * PI > *theta {
                    1.0
                } else {
                    0.000001
                }
            }
        }
    }
}

impl Display for Functions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Functions::CosTheta(scaling) => write!(f, "cos_theta_scaling_{scaling}"),
            Functions::ThetaStripes(start, end) => write!(f, "stripe_{start}_{end}"),
        }
    }
}
