/// General one dimensional grid. It is used to create a grid for a specific dimension.
/// The grid contains:
/// - `name`: name of the grid
/// - `dimension_no`: number of the dimension in the whole n-dimensional grid, starting from 0
/// - `nodes_no`: number of nodes in the grid
/// - `nodes`: vector of nodes
/// - `weights`: vector of weights used for calculating integral.
///
/// Grid can be created using methods:
/// - `new_linear_continuos`: creates a grid with linearly spaced nodes and weights associated to continuous space
/// - `new_linear_countable`: creates a grid with linearly spaced nodes and weights associated to countable space
/// - `new_custom`: creates a grid with given custom nodes and weights
#[derive(Clone, Default)]
pub struct Grid {
    pub name: String,
    pub dimension_no: usize,
    pub nodes_no: usize,
    pub nodes: Vec<f64>,
    pub weights: Vec<f64>,
}

impl Grid {
    /// Creates a new grid with linearly spaced nodes and weights associated to continuous space.
    pub fn new_linear_continuos(
        name: &str,
        start_position: f64,
        end_position: f64,
        nodes_no: usize,
        dimension_no: usize,
    ) -> Grid {
        let step = (end_position - start_position) / (nodes_no as f64 - 1.0);

        let nodes = (0..nodes_no as usize)
            .map(|i| start_position + step * (i as f64))
            .collect();

        let mut weights = vec![1.0 * step; nodes_no as usize];
        weights[0] = 0.5 * step;
        weights[nodes_no - 1] = 0.5 * step;

        Grid {
            name: name.to_string(),
            dimension_no,
            nodes_no,
            nodes,
            weights,
        }
    }

    /// Creates a new grid with linearly spaced nodes and weights associated to countable space.
    pub fn new_linear_countable(
        name: &str,
        start_position: f64,
        end_position: f64,
        nodes_no: usize,
        dimension_no: usize,
    ) -> Grid {
        let step = (end_position - start_position) / (nodes_no as f64 - 1.0);

        let nodes = (0..nodes_no as usize)
            .map(|i| start_position + step * (i as f64))
            .collect();

        let weights = vec![1.0 * step; nodes_no as usize];

        Grid {
            name: name.to_string(),
            dimension_no,
            nodes_no,
            nodes,
            weights,
        }
    }

    /// Creates a new grid with given custom nodes and weights.
    pub fn new_custom(name: &str, nodes: Vec<f64>, weights: Vec<f64>, dimension_no: usize) -> Grid {
        Grid {
            name: name.to_string(),
            dimension_no,
            nodes_no: nodes.len(),
            nodes,
            weights,
        }
    }

    /// Swaps two grids.
    pub fn swap(&mut self, other: &mut Grid) {
        std::mem::swap(self, other);
    }
}
