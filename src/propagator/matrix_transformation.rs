use crate::{grid::Grid, wave_function::WaveFunction};

use super::transformation::Transformation;
use ndarray::{Array2, Axis, Zip};
use num::complex::Complex64;

/// Diagonalization to operator eigenspace using matrix transformation.
#[derive(Clone)]
pub struct MatrixTransformation {
    dimension_no: usize,
    dimension_size: usize,

    transformation: Array2<Complex64>,
    inverse_transformation: Array2<Complex64>,

    pub grid_transformation: Grid,
}

impl MatrixTransformation {
    /// Creates new [`MatrixDiagonalization`] along given grid that transforms this grid into new grid `grid_transformation`.
    pub fn new(
        grid: &Grid,
        grid_transformation: Grid,
    ) -> Self {

        MatrixTransformation {
            dimension_no: grid.dimension_no,
            dimension_size: grid.nodes_no,
            transformation: Array2::zeros((grid.nodes_no, grid.nodes_no)),
            inverse_transformation: Array2::zeros((grid.nodes_no, grid.nodes_no)),
            grid_transformation,
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

    pub fn get_diagonalization_matrices(self) -> [Array2<Complex64>; 2] {
        [self.transformation, self.inverse_transformation]
    }
}

impl Transformation for MatrixTransformation {
    #[inline(always)]
    fn transform(&mut self, wave_function: &mut WaveFunction) {
        wave_function.grids[self.dimension_no].swap(&mut self.grid_transformation);
        wave_function.change_observer.possible_norm_change = true;

        Zip::from(wave_function.array.lanes_mut(Axis(self.dimension_no)))
            .par_for_each(|mut lane| lane.assign(&self.transformation.dot(&lane)));
    }

    #[inline(always)]
    fn inverse_transform(&mut self, wave_function: &mut WaveFunction) {
        wave_function.grids[self.dimension_no].swap(&mut self.grid_transformation);
        wave_function.change_observer.possible_norm_change = true;

        Zip::from(wave_function.array.lanes_mut(Axis(self.dimension_no)))
            .par_for_each(|mut lane| lane.assign(&self.inverse_transformation.dot(&lane)));
    }
}
