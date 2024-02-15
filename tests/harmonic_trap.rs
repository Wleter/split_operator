#[cfg(test)]
pub mod harmonic_trap {
    use ndarray::{Array1, Ix1};
    use num::complex::Complex64;
    use quantum::{particle_factory::create_atom, particles::Particles, units::energy_units::{Energy, Kelvin}};
    use split_operator::{
        control::Apply,
        grid::Grid,
        hamiltonian_factory::{
            analytic_potentials::harmonic, kinetic_operator::kinetic_hamiltonian,
        },
        leak_control::LeakControl,
        propagation::Propagation,
        propagator::{
            fft_diagonalization::FFTDiagonalization, propagator_factory::one_dim_into_propagator,
        },
        time_grid::{TimeGrid, TimeStep},
        wave_function::{gaussian_distribution, WaveFunction},
        wave_function_saver::StateSaver,
    };

    #[allow(dead_code)]
    #[derive(Clone)]
    pub struct HarmonicTrap {
        propagation: Propagation<Ix1>,
        wave_function: WaveFunction<Ix1>,
        grid: Grid,
    }

    impl HarmonicTrap {
        pub fn new(is_imaginary_time: bool, name: &str) -> Self {
            let mut propagation = Propagation::default();

            let grid = Grid::new_linear_continuos("space", -4.0, 4.0, 256, 0);
            let mut wave_function_array = Array1::<Complex64>::zeros(256);
            for (i, x) in wave_function_array.iter_mut().enumerate() {
                *x = gaussian_distribution(grid.nodes[i], 2.0, 0.2, 0.0);
            }
            let mut wave_function = WaveFunction::new(wave_function_array, vec![grid.clone()]);
            wave_function.normalize(1.0);
            propagation.set_wave_function(wave_function.clone());

            let time_grid = TimeGrid {
                step: 3.0,
                step_no: 1000,
                im_time: is_imaginary_time
            };
            propagation.set_time_grid(time_grid.clone());

            let collision_params = Particles::new_pair(
                create_atom("Li6").unwrap(),
                create_atom("Li7").unwrap(),
                Energy(1000.0, Kelvin),
            );

            let mut potential = Array1::<f64>::zeros(256);
            for (i, x) in potential.iter_mut().enumerate() {
                *x = harmonic(grid.nodes[i], 0.0, collision_params.red_mass(), 0.001)
            }
            let path = std::env::current_dir().unwrap();
            let path = path.to_str().unwrap();
            ndarray_npy::write_npy(
                format!("{path}/tests/test_data/{name}_potential.npy"),
                &potential,
            )
            .unwrap();

            let potential_propagator = one_dim_into_propagator(
                &wave_function,
                potential,
                &grid,
                &time_grid,
                TimeStep::Half
            );

            let fft_transform = FFTDiagonalization::new(&wave_function, &grid, "momentum");
            let kinetic_hamiltonian = kinetic_hamiltonian(&grid, &collision_params);
            let kinetic_propagator = one_dim_into_propagator(
                &wave_function,
                kinetic_hamiltonian,
                &grid,
                &time_grid,
                TimeStep::Full
            );

            let saver = StateSaver::new(
                format!("{path}/tests/test_data/"),
                name.to_string(),
                &time_grid,
                &grid,
                50,
                &wave_function,
            );

            propagation.add_saver(Box::new(saver), true);
            propagation.add_control(
                Box::new(LeakControl::new(&wave_function)),
                Apply::FirstHalf | Apply::SecondHalf,
            );
            propagation.add_propagator(Box::new(potential_propagator));
            propagation.add_diagonalization(Box::new(fft_transform), true);
            propagation.add_propagator(Box::new(kinetic_propagator));

            Self {
                propagation,
                wave_function,
                grid,
            }
        }

        pub fn propagate(&mut self) {
            self.propagation.propagate();
        }

        pub fn save(&mut self) {
            self.propagation.savers_save();
        }
    }
}
