use std::{f64::consts::PI, marker::PhantomData, sync::Arc};

use crate::{grid::Grid, wave_function::WaveFunction};

use super::diagonalization::Diagonalization;
use ndarray::{Axis, Dimension, Zip};
use num::complex::Complex64;
use rustfft::{Fft, FftPlanner};

/// Diagonalization to operator eigenspace using Fourier transformation.
#[derive(Clone)]
pub struct FFTDiagonalization<N: Dimension> {
    dimension_no: usize,
    dimension_size: usize,

    fft: Box<Arc<dyn Fft<f64>>>,
    ifft: Box<Arc<dyn Fft<f64>>>,

    pub grid_transformation: Grid,

    phantom: PhantomData<N>,
}

impl<N: Dimension> FFTDiagonalization<N> {
    /// Creates new [`FFTDiagonalization`] along given grid that transforms this grid into new grid with name `transformed_grid_name`.
    pub fn new(
        _example_wave_function: &WaveFunction<N>,
        grid: &Grid,
        transformed_grid_name: &str,
    ) -> FFTDiagonalization<N> {
        let fft = FftPlanner::new().plan_fft_forward(grid.nodes_no);
        let ifft = FftPlanner::new().plan_fft_inverse(grid.nodes_no);

        let momentum_step = 2.0 * PI / (grid.nodes.last().unwrap() - grid.nodes.first().unwrap());
        let length: i64 = grid.nodes_no as i64;
        let momenta: Vec<f64> = (0..length / 2)
            .chain(-length / 2..0)
            .map(|x| x as f64 * momentum_step)
            .collect();

        let mut weights: Vec<f64> = vec![momentum_step; momenta.len()];
        weights[length as usize / 2 - 1] *= 0.5;
        weights[length as usize / 2] *= 0.5;

        let grid = Grid::new_custom(transformed_grid_name, momenta, weights, grid.dimension_no);

        FFTDiagonalization {
            dimension_no: grid.dimension_no,
            dimension_size: grid.nodes_no,
            fft: Box::new(fft),
            ifft: Box::new(ifft),
            grid_transformation: grid,
            phantom: PhantomData,
        }
    }
}

impl<N: Dimension> Diagonalization<N> for FFTDiagonalization<N> {
    #[inline(always)]
    fn diagonalize(&mut self, wave_function: &mut WaveFunction<N>) {
        wave_function.grids[self.dimension_no].swap(&mut self.grid_transformation);
        wave_function.change_observer.possible_norm_change = true;

        let dimension_size_sqrt = (self.dimension_size as f64).sqrt();

        Zip::from(wave_function.array.lanes_mut(Axis(self.dimension_no))).par_for_each(
            |mut lane| {
                let mut temp = lane.to_vec();
                self.fft.process(&mut temp);

                lane.iter_mut().zip(temp.iter()).for_each(|(dest, src)| {
                    *dest = *src / dimension_size_sqrt;
                });
            },
        )
    }

    #[inline(always)]
    fn inverse_diagonalize(&mut self, wave_function: &mut WaveFunction<N>) {
        wave_function.grids[self.dimension_no].swap(&mut self.grid_transformation);
        wave_function.change_observer.possible_norm_change = true;

        let dimension_size_sqrt = Complex64::from((self.dimension_size as f64).sqrt());

        Zip::from(wave_function.array.lanes_mut(Axis(self.dimension_no))).par_for_each(
            |mut lane| {
                let mut temp = lane.to_vec();
                self.ifft.process(&mut temp);

                lane.iter_mut().zip(temp.iter()).for_each(|(dest, src)| {
                    *dest = *src / dimension_size_sqrt;
                });
            },
        )
    }
}
