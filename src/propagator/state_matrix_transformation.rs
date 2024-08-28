use crate::{grid::Grid, wave_function::WaveFunction};

use super::transformation::Transformation;
use ndarray::{Array2, Axis, Zip};
use num::complex::Complex64;

/// Diagonalization to operator eigenspace using matrix transformation.
#[derive(Clone)]
pub struct StateMatrixTransformation {
    dimension_no: usize,
    dimension_no_dependent: usize,

    transformations: Vec<Array2<Complex64>>,
    inverse_transformations: Vec<Array2<Complex64>>,

    pub grid_transformation: Grid,
}

impl StateMatrixTransformation {
    /// Creates new [`StateMatrixDiagonalization`] along given grid that transforms this grid into new grid `grid_transformation`.
    pub fn new(dimension_no_dependent: usize,
        grid: &Grid,
        grid_transformation: Grid,
    ) -> Self {
        assert!(dimension_no_dependent > grid.dimension_no, "Dependent dimension should have higher dimension index");

        StateMatrixTransformation {
            dimension_no: grid.dimension_no,
            dimension_no_dependent,
            transformations: Vec::new(),
            inverse_transformations: Vec::new(),
            grid_transformation,
        }
    }

    pub fn set_diagonalization_matrices(
        &mut self,
        transformations: Vec<Array2<Complex64>>,
        inverse_transformations: Vec<Array2<Complex64>>,
    ) {
        self.transformations = transformations;
        self.inverse_transformations = inverse_transformations;
    }
}

impl Transformation for StateMatrixTransformation {
    #[inline(always)]
    fn transform(&mut self, wave_function: &mut WaveFunction) {
        wave_function.grids[self.dimension_no].swap(&mut self.grid_transformation);
        wave_function.change_observer.possible_norm_change = true;

        self.transformations.iter()
            .zip(wave_function.array.axis_iter_mut(Axis(self.dimension_no_dependent)))
            .for_each(|(t, mut array)| {
                Zip::from(array.lanes_mut(Axis(self.dimension_no)))
                    .par_for_each(|mut lane| lane.assign(&t.dot(&lane)))
            }
        )
    }

    #[inline(always)]
    fn inverse_transform(&mut self, wave_function: &mut WaveFunction) {
        wave_function.grids[self.dimension_no].swap(&mut self.grid_transformation);
        wave_function.change_observer.possible_norm_change = true;

        self.inverse_transformations.iter()
            .zip(wave_function.array.axis_iter_mut(Axis(self.dimension_no_dependent)))
            .for_each(|(t, mut array)| {
                Zip::from(array.lanes_mut(Axis(self.dimension_no)))
                    .par_for_each(|mut lane| lane.assign(&t.dot(&lane)))
            }
        )
    }
}
