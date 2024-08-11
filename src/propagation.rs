use std::cell::RefCell;

use crate::{
    control::{Apply, Control},
    propagator::{transformation::{Transformation, Order}, Propagator},
    saver::Saver,
    time_grid::TimeGrid,
    wave_function::WaveFunction,
};

/// Enum of all operations that can be performed during step in propagation.
enum Operations {
    Propagator(RefCell<Box<dyn Propagator>>),
    Transformation(RefCell<Box<dyn Transformation>>, Order),
    Saver(RefCell<Box<dyn Saver>>, Apply),
    Control(RefCell<Box<dyn Control>>, Apply),
}

/// Operation stack defining split operator propagation step
/// There are 4 types of operations:
/// 1. Propagator - operator that implement [`Propagation`].
/// 2. Transformation - operator that transform basis of `wave_function` and implement [`Diagonalization`].
/// 3. Saver - saves states of `wave_function` during propagation and implement [`Saver`].
/// 4. Control - other operations that control `wave_function` during propagation and implement [`Control`].
#[derive(Default)]
pub struct OperationStack {
    stack: Vec<Operations>,
}

impl OperationStack {
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
        }
    }

    /// Returns actual number of operations to be performed during one step in propagation.
    pub fn operations_len(&self) -> usize {
        self.stack.len()
    }

    /// Appends `Propagator` to the end of the operations.
    pub fn add_propagator(mut self, propagator: Box<dyn Propagator>) -> Self {
        self.stack.push(Operations::Propagator(RefCell::new(propagator)));
        self
    }

    /// Appends `Diagonalization` to the end of the operations.
    /// `order` is used to define the order of the transformations performed.
    pub fn add_transformation(mut self, transformation: Box<dyn Transformation>, order: Order) -> Self {
        self.stack.push(Operations::Transformation(RefCell::new(transformation), order));
        self
    }

    /// Appends `Saver` to the end of the operations. 
    /// `apply` is used to define when `Saver` should be applied.
    pub fn add_saver(mut self, saver: Box<dyn Saver>, apply: Apply) -> Self {
        assert!(apply != Apply::FirstHalf & Apply::SecondHalf);

        self.stack.push(Operations::Saver(RefCell::new(saver), apply));
        self
    }

    /// Appends `Control` to the end of the operations. `apply` is used to define when `Control` should be applied.
    pub fn add_control(mut self, control: Box<dyn Control>, apply: Apply) -> Self {
        self.stack.push(Operations::Control(RefCell::new(control), apply));
        self
    }
}

/// Struct that is used to perform the propagation of the wave function using Split-operator method in N dimensional space.
/// Building the split-operator method is done by appending operations in the order they should be performed,
/// last appended operation should be the central one with full step.
///
/// With supplied [`WaveFunction`] and [`TimeGrid`] using setters, propagation is performed by calling `propagate` method.
/// For example implementation of Propagation see `NeOcs` struct and `Animation` that builds NeOcs and propagate it.
#[derive(Default)]
pub struct Propagation {
    wave_function: WaveFunction,
    time_grid: TimeGrid,
    operation_stack: OperationStack,
}

impl Propagation {
    /// Creates new `Propagation` with supplied `WaveFunction` and `TimeGrid`.
    pub fn new(wave_function: WaveFunction, time_grid: TimeGrid, operation_stack: OperationStack) -> Self {
        Propagation {
            wave_function,
            time_grid,
            operation_stack,
        }
    }

    /// Sets new `WaveFunction` to be used in propagation.
    pub fn set_wave_function(&mut self, wave_function: WaveFunction) {
        self.wave_function = wave_function;
    }

    /// Sets new `TimeGrid` to be used in propagation. Be aware that appended operations might have depended on previous `time_step` in `TimeGrid`.
    pub fn set_time_grid(&mut self, time_grid: TimeGrid) {
        self.time_grid = time_grid;
    }

    /// Sets new `OperationStack` defining operations during propagation.
    pub fn set_operation_stack(&mut self, operation_stack: OperationStack) {
        self.operation_stack = operation_stack;
    }

    /// Returns reference to `TimeGrid` used in propagation.
    pub fn time_grid(&self) -> &TimeGrid {
        &self.time_grid
    }

    /// Returns reference to `WaveFunction` used in propagation.
    pub fn wave_function(&self) -> &WaveFunction {
        &self.wave_function
    }

    /// Resets all `Saver` states in operations vector.
    pub fn reset_savers_state(&mut self) {
        for op in &self.operation_stack.stack {
            if let Operations::Saver(saver, _) = op {
                saver.borrow_mut().reset();
            }
        }
    }

    /// Resets all losses that were observed by `Propagator` with enabled loss checking.
    pub fn reset_losses(&mut self) {
        for op in &mut self.operation_stack.stack {
            if let Operations::Propagator(propagator) = op {
                propagator.borrow_mut().loss_reset();
            }
        }
    }

    /// Performs one step in propagation.
    fn step(&mut self) {
        for op in &mut self.operation_stack.stack {
            match op {
                Operations::Propagator(propagator) => {
                    propagator.borrow_mut().apply(&mut self.wave_function);
                }
                Operations::Transformation(transformation, order) => {
                    match order {
                        Order::Normal => transformation.borrow_mut().transform(&mut self.wave_function),
                        Order::InverseFirst => transformation.borrow_mut().inverse_transform(&mut self.wave_function),
                    }
                }
                Operations::Saver(saver, apply) => {
                    if *apply & Apply::FirstHalf != Apply::None {
                        saver.borrow_mut().monitor(&mut self.wave_function)
                    };
                }
                Operations::Control(control, apply) => {
                    if *apply & Apply::FirstHalf != Apply::None {
                        control.borrow_mut().first_half(&mut self.wave_function);
                    }
                }
            }
        }

        for op in &mut self.operation_stack.stack.iter().rev().skip(1) {
            match op {
                Operations::Propagator(propagator) => {
                    propagator.borrow_mut().apply(&mut self.wave_function);
                }
                Operations::Transformation(transformation, order) => {
                    match order {
                        Order::Normal => transformation.borrow_mut().inverse_transform(&mut self.wave_function),
                        Order::InverseFirst => transformation.borrow_mut().transform(&mut self.wave_function),
                    }
                }
                Operations::Saver(saver, apply) => {
                    if *apply & Apply::FirstHalf != Apply::None {
                        saver.borrow_mut().monitor(&mut self.wave_function)
                    };
                }
                Operations::Control(control, apply) => {
                    if *apply & Apply::SecondHalf != Apply::None {
                        control.borrow_mut().second_half(&mut self.wave_function);
                    }
                }
            }
        }
    }

    /// Performs propagation of the `wave_function` for the time given by `TimeGrid`.
    pub fn propagate(&mut self) {
        for _ in 0..self.time_grid.step_no {
            self.step();
        }
    }

    /// Prints losses that were observed by `Propagator` with enabled loss checking.
    pub fn print_losses(&mut self) {
        for op in &mut self.operation_stack.stack {
            match op {
                Operations::Propagator(propagator) => {
                    let borrowed = propagator.borrow();
                    let loss_checker = borrowed.loss();
                    if let Some(loss) = loss_checker {
                        println!("{} loss: {}", loss.name, loss.loss());
                    }
                }
                Operations::Control(control, _) => {
                    let borrowed = control.borrow();
                    let loss_checker = borrowed.loss();
                    if let Some(loss) = loss_checker {
                        println!("{} loss: {}", loss.name, loss.loss());
                    }
                }
                _ => {}
            }
        }
    }

    /// Returns losses that were observed by `Propagator` with enabled loss checking.
    pub fn get_losses(&mut self) -> Vec<f64> {
        let mut losses = Vec::new();
        for op in &mut self.operation_stack.stack {
            match op {
                Operations::Propagator(propagator) => {
                    let borrowed = propagator.borrow();
                    let loss_checker = borrowed.loss();
                    if let Some(loss) = loss_checker {
                        losses.push(loss.loss());
                    }
                }
                _ => {}
            }
        }
        losses
    }

    /// Saves states of `wave_function` during propagation observed by all `Saver`.
    pub fn savers_save(&mut self) {
        for op in &mut self.operation_stack.stack {
            match op {
                Operations::Saver(saver, _) => {
                    saver.borrow_mut().save().unwrap();
                }
                _ => {}
            }
        }
    }

    pub fn mean_energy(&mut self) -> f64 {
        if self.time_grid.im_time == true {
            match &self.operation_stack.stack[0] {
                Operations::Control(control, _) => {
                    assert!(
                        control.borrow_mut().name() == "LeakControl",
                        "For imaginary time propagation first control must be LeakControl."
                    );
                    let mut borrowed = control.borrow_mut();
                    let loss = borrowed.loss_mut();

                    if let Some(loss) = loss {
                        loss.reset()
                    } else {
                        panic!("Leak control must have loss checker to get mean energy for imaginary time.")
                    }
                }
                _ => panic!(""),
            }
            self.step();

            let decay = match &self.operation_stack.stack[0] {
                Operations::Control(control, _) => {
                    let mut borrowed = control.borrow_mut();
                    let loss = borrowed.loss_mut();

                    if let Some(loss) = loss {
                        loss.loss()
                    } else {
                        panic!("Leak control must have loss checker to get mean energy.")
                    }
                }
                _ => panic!(""),
            };

            let energy = -(self.wave_function.norm() - decay).ln() / self.time_grid.step;
            energy
        } else {
            let mut wave_before = self.wave_function.clone();
            self.step();
            let mut wave_after = self.wave_function.clone();

            let energy = -wave_after.dot(&mut wave_before).arg() / self.time_grid.step;
            energy
        }
    }
}
