use std::cell::RefCell;

use ndarray::Dimension;

use crate::{
    control::{Apply, Control},
    propagator::{diagonalization::Diagonalization, Propagator},
    saver::Saver,
    time_grid::TimeGrid,
    wave_function::WaveFunction,
};

/// Enum of all operations that can be performed during step in propagation.
#[derive(Clone)]
enum Operations<N: Dimension> {
    Propagator(RefCell<Box<dyn Propagator<N>>>),
    Diagonalization(RefCell<Box<dyn Diagonalization<N>>>, bool),
    Saver(RefCell<Box<dyn Saver<N>>>, bool),
    Control(RefCell<Box<dyn Control<N>>>, Apply),
}

/// Struct that is used to perform the propagation of the wave function using Split-operator method in N dimensional space.
/// Building the split-operator method is done by appending operations in the order they should be performed,
/// last appended operation should be the central one with full step.
/// There are 4 types of operations:
/// 1. Propagator - operator that implement [`Propagation<N>`].
/// 2. Diagonalization - operator that transform basis of `wave_function` and implement [`Diagonalization<N>`].
/// 3. Saver - saves states of `wave_function` during propagation and implement [`Saver<N>`].
/// 4. Control - other operations that control `wave_function` during propagation and implement [`Control<N>`].
///
/// With supplied [`WaveFunction<N>`] and [`TimeGrid`] using setters, propagation is performed by calling `propagate` method.
/// For example implementation of Propagation see `NeOcs` struct and `Animation` that builds NeOcs and propagate it.
#[derive(Clone, Default)]
pub struct Propagation<N: Dimension> {
    wave_function: WaveFunction<N>,
    time_grid: TimeGrid,
    operations: Vec<Operations<N>>,
}

impl<N: Dimension> Propagation<N> {
    /// Creates new `Propagation<N>` with supplied `WaveFunction<N>` and `TimeGrid`.
    pub fn new(wave_function: WaveFunction<N>, time_grid: TimeGrid) -> Self {
        Propagation {
            wave_function,
            time_grid,
            operations: Vec::new(),
        }
    }

    /// Returns actual number of operations to be performed during one step in propagation.
    pub fn operations_len(&self) -> usize {
        self.operations.len()
    }

    /// Appends `Propagator<N>` to the end of the operations.
    pub fn add_propagator(&mut self, propagator: Box<dyn Propagator<N>>) {
        self.operations
            .push(Operations::Propagator(RefCell::new(propagator)));
    }

    /// Replaces `Propagator` that is in operations vector at index `index`.
    pub fn replace_propagator(&mut self, propagator: Box<dyn Propagator<N>>, index: usize) {
        self.operations[index] = Operations::Propagator(RefCell::new(propagator));
    }

    /// Appends `Diagonalization<N>` to the end of the operations.
    pub fn add_diagonalization(
        &mut self,
        diagonalization: Box<dyn Diagonalization<N>>,
        diagonalization_first: bool,
    ) {
        self.operations.push(Operations::Diagonalization(
            RefCell::new(diagonalization),
            diagonalization_first,
        ));
    }

    /// Appends `Saver<N>` to the end of the operations.
    pub fn add_saver(&mut self, saver: Box<dyn Saver<N>>, first_half: bool) {
        self.operations
            .push(Operations::Saver(RefCell::new(saver), first_half));
    }

    /// Appends `Control<N>` to the end of the operations. `config` is used to determine when `Control<N>` should be applied.
    /// - 1 - apply on first half of the time step
    /// - 2 - apply on second half of the time step
    /// - 3 - apply on both halves of the time step
    pub fn add_control(&mut self, control: Box<dyn Control<N>>, config: Apply) {
        self.operations
            .push(Operations::Control(RefCell::new(control), config));
    }

    /// Sets new `WaveFunction<N>` to be used in propagation.
    pub fn set_wave_function(&mut self, wave_function: WaveFunction<N>) {
        self.wave_function = wave_function;
    }

    /// Sets new `TimeGrid` to be used in propagation. Be aware that appended operations might have depended on previous `time_step` in `TimeGrid`.
    pub fn set_time_grid(&mut self, time_grid: TimeGrid) {
        self.time_grid = time_grid;
    }

    /// Returns reference to `TimeGrid` used in propagation.
    pub fn time_grid(&self) -> &TimeGrid {
        &self.time_grid
    }

    /// Resets all `Saver<N>` states in operations vector.
    pub fn reset_savers_state(&mut self) {
        for saver in &self.operations {
            if let Operations::Saver(saver, _) = saver {
                saver.borrow_mut().reset();
            }
        }
    }

    /// Clears all operations
    pub fn reset_operations(&mut self) {
        self.operations.clear();
    }

    /// Resets all losses that were observed by `Propagator<N>` with enabled loss checking.
    pub fn reset_losses(&mut self) {
        for op in &mut self.operations {
            match op {
                Operations::Propagator(propagator) => {
                    propagator.borrow_mut().loss_reset();
                }
                _ => {}
            }
        }
    }

    /// Performs one step in propagation.
    fn step(&mut self) {
        for op in &mut self.operations {
            match op {
                Operations::Propagator(propagator) => {
                    propagator.borrow_mut().apply(&mut self.wave_function);
                }
                Operations::Diagonalization(diagonalization, diagonalization_first) => {
                    if *diagonalization_first {
                        diagonalization
                            .borrow_mut()
                            .diagonalize(&mut self.wave_function);
                    } else {
                        diagonalization
                            .borrow_mut()
                            .inverse_diagonalize(&mut self.wave_function);
                    }
                }
                Operations::Saver(saver, first_half) => {
                    if *first_half {
                        saver.borrow_mut().monitor(&mut self.wave_function)
                    };
                }
                Operations::Control(control, config) => {
                    if *config & Apply::FirstHalf != Apply::None {
                        control.borrow_mut().first_half(&mut self.wave_function);
                    }
                }
            }
        }

        for op in &mut self.operations.iter().rev().skip(1) {
            match op {
                Operations::Propagator(propagator) => {
                    propagator.borrow_mut().apply(&mut self.wave_function);
                }
                Operations::Diagonalization(diagonalization, diagonalization_first) => {
                    if *diagonalization_first {
                        diagonalization
                            .borrow_mut()
                            .inverse_diagonalize(&mut self.wave_function);
                    } else {
                        diagonalization
                            .borrow_mut()
                            .diagonalize(&mut self.wave_function);
                    }
                }
                Operations::Saver(saver, first_half) => {
                    if !first_half {
                        saver.borrow_mut().monitor(&mut self.wave_function)
                    };
                }
                Operations::Control(control, config) => {
                    if *config & Apply::SecondHalf != Apply::None {
                        control.borrow_mut().second_half(&mut self.wave_function);
                    }
                }
            }
        }
    }

    /// Performs propagation of the `wave_function` for time given by `TimeGrid`.
    pub fn propagate(&mut self) {
        for _ in 0..self.time_grid.step_no {
            self.step();
        }
    }

    /// Prints losses that were observed by `Propagator<N>` with enabled loss checking.
    pub fn print_losses(&mut self) {
        for op in &mut self.operations {
            match op {
                Operations::Propagator(propagator) => {
                    let propagator_borrowed = propagator.borrow_mut();
                    let loss_checker = propagator_borrowed.loss();
                    if let Some(loss) = loss_checker {
                        println!("{} loss: {}", loss.name, loss.loss());
                    }
                }
                Operations::Control(control, _) => {
                    let control_borrowed = control.borrow_mut();
                    let loss_checker = control_borrowed.loss();
                    if let Some(loss) = loss_checker {
                        println!("{} loss: {}", loss.name, loss.loss());
                    }
                }
                _ => {}
            }
        }
    }

    /// Returns losses that were observed by `Propagator<N>` with enabled loss checking.
    pub fn get_losses(&mut self) -> Vec<f64> {
        let mut losses = Vec::new();
        for op in &mut self.operations {
            match op {
                Operations::Propagator(propagator) => {
                    let propagator_borrowed = propagator.borrow_mut();
                    let loss_checker = propagator_borrowed.loss();
                    if let Some(loss) = loss_checker {
                        losses.push(loss.loss());
                    }
                }
                _ => {}
            }
        }
        losses
    }

    /// Saves states of `wave_function` during propagation observed by all `Saver<N>`.
    pub fn savers_save(&mut self) {
        for op in &mut self.operations {
            match op {
                Operations::Saver(saver, _) => {
                    saver.borrow_mut().save().unwrap();
                }
                _ => {}
            }
        }
    }

    pub fn wave_function(&self) -> &WaveFunction<N> {
        &self.wave_function
    }

    pub fn mean_energy(&mut self) -> f64 {
        if self.time_grid.im_time == true {
            match &mut self.operations[0] {
                Operations::Control(control, _) => {
                    assert!(control.borrow().name() == "LeakControl", "For imaginary time propagation first control must be LeakControl.");
                    let mut borrowed_control = control.borrow_mut();
                    let loss = borrowed_control.loss_mut();

                    if let Some(loss) = loss {
                        loss.reset()
                    } else {
                        panic!("Leak control must have loss checker to get mean energy for imaginary time.")
                    }
                },
                _ => panic!("")
            }
            self.step();
            
            let decay = match &self.operations[0] {
                Operations::Control(control, _) => {
                    let mut borrowed_control = control.borrow_mut();
                    let loss = borrowed_control.loss_mut();
                    
                    if let Some(loss) = loss {
                        loss.loss()
                    } else {
                        panic!("Leak control must have loss checker to get mean energy.")
                    }

                },
                _ => panic!("")
            };
            
            let energy = - (self.wave_function.norm() - decay).ln() / self.time_grid.step;
            energy
        } else {
            let mut wave_before = self.wave_function.clone();
            self.step();
            let mut wave_after = self.wave_function.clone();

            let energy = - wave_after.dot(&mut wave_before).arg() / self.time_grid.step;
            energy
        }
    }
}
