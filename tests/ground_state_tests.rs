pub mod harmonic_trap;

#[cfg(test)]
mod ground_state_tests {
    use std::time::Instant;

    use ndarray::{Array1, Ix1};
    use num::complex::Complex64;
    use quantum::{
        particle_factory::create_atom,
        particles::Particles,
        units::energy_units::{Energy, Kelvin},
    };
    use split_operator::{
        control::Apply,
        grid::Grid,
        hamiltonian_factory::{
            analytic_potentials::lennard_jones, kinetic_operator::kinetic_hamiltonian,
        },
        leak_control::LeakControl,
        loss_checker::LossChecker,
        propagation::Propagation,
        propagator::{
            fft_diagonalization::FFTDiagonalization, propagator_factory::one_dim_into_propagator,
        },
        time_grid::{TimeGrid, TimeStep},
        wave_function::{gaussian_distribution, WaveFunction},
        wave_function_saver::StateSaver,
    };

    #[test]
    fn propagation_test() {
        let mut propagation = OneChannelPropagation::new();

        let start = Instant::now();
        propagation.propagate();
        let elapsed = start.elapsed();
        println!("Elapsed time: {:?}", elapsed);

        propagation.save();
    }

    #[allow(dead_code)]
    #[derive(Clone)]
    struct OneChannelPropagation {
        propagation: Propagation<Ix1>,
        wave_function: WaveFunction<Ix1>,
        grid: Grid,
    }

    impl OneChannelPropagation {
        pub fn new() -> Self {
            let mut propagation = Propagation::<Ix1>::default();

            let x_start = 8.0;
            let x_end = 20.0;
            let x_no = 512;
            let grid = Grid::new_linear_continuos("space", x_start, x_end, x_no, 0);

            let collision_params = Particles::new_pair(
                create_atom("Li6").unwrap(),
                create_atom("Li7").unwrap(),
                Energy(1e-7, Kelvin),
            );

            let momentum = (2.0
                * collision_params.red_mass()
                * collision_params.internals.get_value("energy"))
            .sqrt();
            let mut wave_function_array = Array1::<Complex64>::zeros(x_no);
            for (i, x) in grid.nodes.iter().enumerate() {
                wave_function_array[i] = gaussian_distribution(*x, 14.0, 2.0, momentum);
            }
            let mut wave_function = WaveFunction::new(wave_function_array, vec![grid.clone()]);
            wave_function.normalize(1.0);
            propagation.set_wave_function(wave_function.clone());

            let time_grid = TimeGrid {
                step: 50.0,
                step_no: 1000,
                im_time: true,
            };
            propagation.set_time_grid(time_grid.clone());

            let r6 = 9.7;
            let d6 = 0.0003;
            let mut potential_array = Array1::<f64>::zeros(x_no);
            for (i, x) in grid.nodes.iter().enumerate() {
                potential_array[i] = lennard_jones(*x, d6, r6);
            }
            let path = std::env::current_dir().unwrap();
            let path = path.to_str().unwrap();
            ndarray_npy::write_npy(
                format!("{path}/tests/test_data/lj_ground_space_potential.npy"),
                &potential_array,
            )
            .unwrap();

            let potential_propagator = one_dim_into_propagator(
                &wave_function,
                potential_array,
                &grid,
                &time_grid,
                TimeStep::Half,
            );

            let kinetic_array = kinetic_hamiltonian(&grid, &collision_params);
            let kinetic_propagator = one_dim_into_propagator(
                &wave_function,
                kinetic_array,
                &grid,
                &time_grid,
                TimeStep::Full,
            );
            let fft_transform = FFTDiagonalization::new(&wave_function, &grid, "momentum");

            let name = "lj_ground_space".to_string();
            let wave_function_saver = StateSaver::new(
                format!("{path}/tests/test_data/"),
                name,
                &time_grid,
                &grid,
                120,
                &wave_function,
            );

            let mut leak_control = LeakControl::new(&wave_function);
            leak_control.add_loss_checker(LossChecker::new("leak control"));

            propagation.add_control(Box::new(leak_control), Apply::FirstHalf | Apply::SecondHalf);
            propagation.add_saver(Box::new(wave_function_saver), true);
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

            println!("Mean energy: {}", self.propagation.mean_energy());
        }

        pub fn save(&mut self) {
            self.propagation.savers_save();
        }
    }
}
