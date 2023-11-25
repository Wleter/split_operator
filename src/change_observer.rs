use crate::grid::Grid;

/// Observes changes in the wave function such as possible norm change and grid change.
/// Used inside `WaveFunction` to help calculate the norm of the wave function.
/// Propagators and diagonalizations change `possible_norm_change` to true when they act on a wave function.
#[derive(Clone, Default)]
pub struct ChangeObserver {
    last_grid_names: Vec<String>,

    last_norm: f64,
    pub possible_norm_change: bool,
}

impl ChangeObserver {
    /// Creates new `ChangeObserver` with given grids.
    pub fn new(grids: &Vec<Grid>) -> Self {
        ChangeObserver {
            last_grid_names: grids.into_iter().map(|x| x.name.clone()).collect(),
            last_norm: 1.0,
            possible_norm_change: true,
        }
    }

    /// Observes current new grids.
    pub fn observe_grid(&mut self, new_grids: &Vec<Grid>) {
        self.last_grid_names = new_grids.into_iter().map(|x| x.name.clone()).collect();
    }

    /// Observes current new norm.
    pub fn observe_norm(&mut self, new_norm: f64) {
        self.last_norm = new_norm;
        self.possible_norm_change = false;
    }

    /// Returns last observed norm.
    pub fn last_norm(&self) -> f64 {
        self.last_norm
    }

    /// Returns true if grid has changed since last observation using `observe_grid`.
    pub fn has_grid_changed(&self, grids: &Vec<Grid>) -> bool {
        self.last_grid_names
            .iter()
            .zip(grids.into_iter().map(|x| x.name.clone()))
            .any(|(x, y)| x != &y)
    }
}
