#[cfg(test)]
mod harmonic_tests {
    use std::time::Instant;

    use ndarray::{Array1, ArrayD, IxDyn};
    use num::complex::Complex64;
    use quantum::{particle::Particle, particles::Particles, units::{energy_units::{Energy, Kelvin}, mass_units::{Dalton, Mass}, Unit}};
    use split_operator::{control::Apply, grid::Grid, hamiltonian_factory::kinetic_operator::kinetic_hamiltonian, leak_control::LeakControl, propagation::{OperationStack, Propagation}, propagator::{fft_transformation::FFTTransformation, propagator_factory::one_dim_into_propagator, transformation::Order}, time_grid::{TimeGrid, TimeStep}, wave_function::{gaussian_distribution, WaveFunction}, wave_function_saver::StateSaver};

    #[test]
    fn propagation_test() {
        let mut propagation = Propagation::default();

        let collision_params = Particles::new_pair(
            Particle::new("au", Mass(12.0, Dalton)),
            Particle::new("au", Mass(12.0, Dalton)),
            Energy(3000.0, Kelvin),
        );

        let momentum = (2.0 * collision_params.red_mass() * 3000. * Kelvin::TO_AU_MUL).sqrt();

        let grid = Grid::new_linear_continuos("space", 0.0, 30.0, 512, 0);
        let mut wave_function_array = ArrayD::<Complex64>::zeros(IxDyn(&[512]));
        for (i, x) in wave_function_array.iter_mut().enumerate() {
            *x = gaussian_distribution(grid.nodes[i], 15.0, 0.5, momentum);
        }
        let mut wave_function = WaveFunction::new(wave_function_array, vec![grid.clone()]);
        wave_function.normalize(1.0);
        propagation.set_wave_function(wave_function.clone());

        let time_grid = TimeGrid {
            step: 50.0,
            step_no: 500,
            im_time: false,
        };
        propagation.set_time_grid(time_grid.clone());

        println!("{}", momentum);
        println!("should travel: {}", momentum / collision_params.red_mass() * time_grid.step * time_grid.step_no as f64);

        let potential = Array1::<f64>::zeros(512);

        let potential_propagator = one_dim_into_propagator(
            potential,
            &grid,
            &time_grid,
            TimeStep::Half,
        );

        let fft_transform = FFTTransformation::new(&grid, "momentum");
        let kinetic_hamiltonian = kinetic_hamiltonian(&grid, &collision_params);
        let kinetic_propagator = one_dim_into_propagator(
            kinetic_hamiltonian,
            &grid,
            &time_grid,
            TimeStep::Full,
        );

        let saver = StateSaver::new(
            format!("tests/test_data/free_wave"),
            &time_grid,
            &grid,
            50,
        );

        let mut operation_stack = OperationStack::new();
        operation_stack.add_saver(Box::new(saver), Apply::FirstHalf);
        operation_stack.add_control(
                Box::new(LeakControl::new()),
                Apply::FirstHalf | Apply::SecondHalf
            );
        operation_stack.add_propagator(Box::new(potential_propagator));
        operation_stack.add_transformation(Box::new(fft_transform), Order::Normal);
        operation_stack.add_propagator(Box::new(kinetic_propagator));

        propagation.set_operation_stack(operation_stack);

        let start = Instant::now();
        propagation.propagate();
        let elapsed = start.elapsed();
        println!("Elapsed time: {:?}", elapsed);

        propagation.savers_save();
    }
}
