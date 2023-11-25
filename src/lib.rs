pub mod border_dumping;
pub mod change_observer;
pub mod control;
pub mod grid;
pub mod hamiltonian_factory;
pub mod leak_control;
pub mod loss_checker;
pub mod propagation;
pub mod propagator;
pub mod saver;
pub mod special_functions;
pub mod time_grid;
pub mod wave_function;
pub mod wave_function_saver;
pub mod loss_saver;

pub mod ne_ocs_propagation;
pub mod potential_reader;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
