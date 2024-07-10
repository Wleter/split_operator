use gauss_quad::GaussLegendre;

/// Returns the gauss legendre quadrature with nodes in ascending order.
pub fn gauss_legendre_quad(points_no: usize) -> GaussLegendre {
    let mut gauss_legendre = GaussLegendre::init(points_no);

    let mut nodes_with_weights: Vec<(f64, f64)> = gauss_legendre
        .nodes
        .iter()
        .zip(gauss_legendre.weights.iter())
        .map(|(&node, &weight)| (node, weight))
        .collect();

    nodes_with_weights.sort_by(|(a_node, _), (b_node, _)| b_node.partial_cmp(a_node).unwrap());

    gauss_legendre.nodes = nodes_with_weights.iter().map(|(node, _)| *node).collect();

    gauss_legendre.weights = nodes_with_weights
        .iter()
        .map(|(_, weight)| *weight)
        .collect();

    gauss_legendre
}

/// Returns the legendre polynomials up to order `j` at `x`.
pub fn legendre_polynomials(j: usize, x: f64) -> Vec<f64> {
    let mut p = vec![0.0; j + 1];

    p[0] = 1.0;
    p[1] = x;

    for i in 2..=j {
        p[i] = ((2 * i - 1) as f64 / i as f64) * x * p[i - 1] - ((i - 1) as f64 / i as f64) * p[i - 2];
    }

    p
}
