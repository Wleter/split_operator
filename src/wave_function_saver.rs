use std::marker::PhantomData;

use ndarray::{s, Array, Array1, Array2, Array3, Dimension};
use ndarray_npy::write_npy;

use crate::{grid::Grid, saver::Saver, time_grid::TimeGrid, wave_function::WaveFunction};

/// Saves density of a wave function that is in 2d space during propagation.
#[derive(Clone)]
pub struct WaveFunctionSaver<N: Dimension> {
    path: String,
    name: String,
    current_frame: usize,
    frames_no: usize,
    time_grid: TimeGrid,
    x_grid: Grid,
    y_grid: Grid,
    data_array: Array3<f64>,
    wf: PhantomData<N>,
}

impl<N: Dimension> WaveFunctionSaver<N> {
    /// Creates new `WaveFunctionSaver` with given path, name, time grid, x grid, y grid, frames number and example wave function.
    pub fn new(
        path: String,
        name: String,
        time_grid: &TimeGrid,
        x_grid: &Grid,
        y_grid: &Grid,
        frames_no: usize,
        _example_wave_function: &WaveFunction<N>,
    ) -> WaveFunctionSaver<N> {
        WaveFunctionSaver {
            path,
            name,
            current_frame: 0,
            frames_no,
            time_grid: time_grid.clone(),
            x_grid: x_grid.clone(),
            y_grid: y_grid.clone(),
            data_array: Array::zeros((x_grid.nodes_no, y_grid.nodes_no, frames_no)),
            wf: PhantomData,
        }
    }
}

impl<N: Dimension> Saver<N> for WaveFunctionSaver<N> {
    fn monitor(&mut self, wave_function: &mut WaveFunction<N>) {
        if wave_function.array.ndim() != 2 {
            panic!("Wave function must be 2d for now");
        }

        let frequency = self.time_grid.step_no / self.frames_no;

        if self.current_frame % frequency == 0 && self.current_frame / frequency < self.frames_no {
            let density = wave_function.density();

            let density2d: Array2<f64> = density
                .into_shape((self.x_grid.nodes_no, self.y_grid.nodes_no))
                .unwrap();

            self.data_array
                .slice_mut(s![.., .., self.current_frame / frequency])
                .assign(&density2d);
        }

        self.current_frame += 1;
    }

    fn save(&self) -> Result<(), &str> {
        let path = [self.path.clone(), self.name.clone()].join("");

        let result = write_npy(path.clone() + ".npy", &self.data_array);
        if result.is_err() {
            return Err("Failed to save wave function");
        }

        let x_grid: Array1<f64> = Array::from_vec(self.x_grid.nodes.clone());
        let result = write_npy(path.clone() + "_x_grid.npy", &x_grid);
        if result.is_err() {
            return Err("Failed to save r grid");
        }

        let y_grid: Array1<f64> = Array::from_vec(self.y_grid.nodes.clone());
        let result = write_npy(path.clone() + "_y_grid.npy", &y_grid);
        if result.is_err() {
            return Err("Failed to save theta grid");
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.data_array = Array::zeros((self.x_grid.nodes_no, self.y_grid.nodes_no, self.frames_no))
    }
}

/// Saves density of a wave function on given dimension during propagation.
#[derive(Clone)]
pub struct StateSaver<N: Dimension> {
    path: String,
    name: String,
    current_frame: usize,
    frames_no: usize,
    time_grid: TimeGrid,
    state_grid: Grid,
    data_array: Array2<f64>,
    wf: PhantomData<N>,
}

impl<N: Dimension> StateSaver<N> {
    /// Creates new `StateSaver` with given path, name, time grid, state grid, frames number and example wave function.
    pub fn new(
        path: String,
        name: String,
        time_grid: &TimeGrid,
        state_grid: &Grid,
        frames_no: usize,
        _example_wave_function: &WaveFunction<N>,
    ) -> StateSaver<N> {
        StateSaver {
            path,
            name,
            current_frame: 0,
            frames_no,
            time_grid: time_grid.clone(),
            state_grid: state_grid.clone(),
            data_array: Array::zeros((state_grid.nodes_no, frames_no)),
            wf: PhantomData,
        }
    }
}

impl<N: Dimension> Saver<N> for StateSaver<N> {
    fn monitor(&mut self, wave_function: &mut WaveFunction<N>) {
        let frequency = self.time_grid.step_no / self.frames_no;

        if self.current_frame % frequency == 0 && self.current_frame / frequency < self.frames_no {
            let state = wave_function.state_density(self.state_grid.dimension_no);

            self.data_array
                .slice_mut(s![.., self.current_frame / frequency])
                .assign(&state);
        }

        self.current_frame += 1;
    }

    fn save(&self) -> Result<(), &str> {
        let path = [self.path.clone(), self.name.clone()].join("");

        let result = write_npy(path.clone() + ".npy", &self.data_array);
        if result.is_err() {
            return Err("Failed to save wave function");
        }

        let state_grid: Array1<f64> = Array::from_vec(self.state_grid.nodes.clone());
        let result = write_npy(
            path.clone() + "_" + &self.state_grid.name.clone() + "_grid.npy",
            &state_grid,
        );
        if result.is_err() {
            return Err("Failed to save state grid");
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.data_array = Array::zeros((self.state_grid.nodes_no, self.frames_no));
    }
}
