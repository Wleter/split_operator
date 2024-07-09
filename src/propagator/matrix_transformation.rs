use std::marker::PhantomData;

use crate::{grid::Grid, wave_function::WaveFunction};

use super::transformation::Transformation;
use ndarray::{Array2, Axis, Dimension, Ix2, Zip};
use num::complex::Complex64;

/// Diagonalization to operator eigenspace using matrix transformation.
#[derive(Clone)]
pub struct MatrixTransformation<N: Dimension> {
    dimension_no: usize,
    dimension_size: usize,

    transformation: Array2<Complex64>,
    inverse_transformation: Array2<Complex64>,

    pub grid_transformation: Grid,

    phantom: PhantomData<N>,
}

impl<N: Dimension> MatrixTransformation<N> {
    /// Creates new [`MatrixDiagonalization`] along given grid that transforms this grid into new grid `grid_transformation`.
    pub fn new(
        example_wave_function: &WaveFunction<N>,
        grid: &Grid,
        grid_transformation: Grid,
    ) -> MatrixTransformation<N> {
        assert!(grid.dimension_no < example_wave_function.array.ndim());
        assert!(grid.nodes_no == example_wave_function.array.shape()[grid.dimension_no]);

        MatrixTransformation {
            dimension_no: grid.dimension_no,
            dimension_size: grid.nodes_no,
            transformation: Array2::zeros((grid.nodes_no, grid.nodes_no)),
            inverse_transformation: Array2::zeros((grid.nodes_no, grid.nodes_no)),
            grid_transformation,
            phantom: PhantomData,
        }
    }

    pub fn set_diagonalization_matrix(
        &mut self,
        transformation: Array2<Complex64>,
        inverse_transformation: Array2<Complex64>,
    ) {
        assert!(
            transformation.shape()[0] == self.dimension_size
                && transformation.shape()[1] == self.dimension_size
        );
        assert!(
            inverse_transformation.shape()[0] == self.dimension_size
                && inverse_transformation.shape()[1] == self.dimension_size
        );

        self.transformation = transformation;
        self.inverse_transformation = inverse_transformation;
    }
}

impl<N: Dimension> Transformation<N> for MatrixTransformation<N> {
    #[inline(always)]
    fn transform(&mut self, wave_function: &mut WaveFunction<N>) {
        wave_function.grids[self.dimension_no].swap(&mut self.grid_transformation);
        wave_function.change_observer.possible_norm_change = true;

        Zip::from(wave_function.array.lanes_mut(Axis(self.dimension_no)))
            .par_for_each(|mut lane| lane.assign(&self.transformation.dot(&lane)));
    }

    #[inline(always)]
    fn inverse_transform(&mut self, wave_function: &mut WaveFunction<N>) {
        wave_function.grids[self.dimension_no].swap(&mut self.grid_transformation);
        wave_function.change_observer.possible_norm_change = true;

        Zip::from(wave_function.array.lanes_mut(Axis(self.dimension_no)))
            .par_for_each(|mut lane| lane.assign(&self.inverse_transformation.dot(&lane)));
    }
}

/// 2D equivalent of `MatrixDiagonalization`.
/// consider unstable specialization using 'default fn' in trait implementation
#[derive(Clone)]
pub struct MatrixDiagonalization2D {
    dimension_no: usize,
    dimension_size: usize,

    transformation: Array2<Complex64>,
    inverse_transformation: Array2<Complex64>,

    pub grid_transformation: Grid,
}

impl MatrixDiagonalization2D {
    pub fn new(
        example_wave_function: &WaveFunction<Ix2>,
        grid: &Grid,
        grid_transformation: Grid,
    ) -> MatrixDiagonalization2D {
        assert!(grid.dimension_no < example_wave_function.array.ndim());
        assert!(grid.nodes_no == example_wave_function.array.shape()[grid.dimension_no]);

        MatrixDiagonalization2D {
            dimension_no: grid.dimension_no,
            dimension_size: grid.nodes_no,
            transformation: Array2::zeros((grid.nodes_no, grid.nodes_no)),
            grid_transformation,
            inverse_transformation: Array2::zeros((grid.nodes_no, grid.nodes_no)),
        }
    }

    pub fn set_diagonalization_matrix(
        &mut self,
        transformation: Array2<Complex64>,
        inverse_transformation: Array2<Complex64>,
    ) {
        assert!(
            transformation.shape()[0] == self.dimension_size
                && transformation.shape()[1] == self.dimension_size
        );
        assert!(
            inverse_transformation.shape()[0] == self.dimension_size
                && inverse_transformation.shape()[1] == self.dimension_size
        );

        self.transformation = transformation;
        self.inverse_transformation = inverse_transformation;
    }
}

impl Transformation<Ix2> for MatrixDiagonalization2D {
    fn transform(&mut self, wave_function: &mut WaveFunction<Ix2>) {
        wave_function.grids[self.dimension_no].swap(&mut self.grid_transformation);
        wave_function.change_observer.possible_norm_change = true;

        if self.dimension_no == 0 {
            wave_function.array = self.transformation.dot(&wave_function.array);
        }

        if self.dimension_no == 1 {
            wave_function.array = wave_function.array.dot(&self.transformation.t());
        }
    }

    fn inverse_transform(&mut self, wave_function: &mut WaveFunction<Ix2>) {
        wave_function.grids[self.dimension_no].swap(&mut self.grid_transformation);
        wave_function.change_observer.possible_norm_change = true;

        if self.dimension_no == 0 {
            wave_function.array = self.inverse_transformation.dot(&wave_function.array);
        }

        if self.dimension_no == 1 {
            wave_function.array = wave_function.array.dot(&self.inverse_transformation.t());
        }
    }
}
