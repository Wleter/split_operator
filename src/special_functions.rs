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

/// Returns the associated legendre polynomials up to `j` at `x`.
pub fn associated_legendre_polynomials(j: usize, m: isize, x: f64) -> Vec<f64> {
    let m_u = m.unsigned_abs();
    if m == 0 {
        return legendre_polynomials(j, x)
    }

    let mut p = vec![0.0; j + 1];

    p[m_u] = (-1.0f64).powi(m as i32) * double_factorial(2 * m_u - 1) * (1. - x * x).powf(m_u as f64 / 2.0);

    if m < 0 {
        p[m_u] *= negate_m(m_u as u32, m_u as i32);
        p[m_u+1] = x * p[m_u]
    } else {
        p[m_u+1] = x * (2. * m_u as f64 + 1.) * p[m_u]
    }

    for i in m_u+2..=j {
        let l = i - 1;
        
        p[i] = (((2 * l + 1) as f64) * x * p[i - 1] - (l as f64 + m as f64) * p[i - 2]) / (l as f64 - m as f64 + 1.);
    }

    p
}

pub fn double_factorial(n: usize) -> f64 {
    assert!(n < 100);
    if n == 0 {
        return 1.;
    }

    let mut value = n as f64;
    for k in (2..n-1).rev().step_by(2) {
        value *= k as f64
    }

    value
}

fn negate_m(l: u32, m: i32) -> f64 {
    assert!(m < 50);

    if m > 0 {
        let mut value = (-1.0f64).powi(m);
        let min = l as i32 - m;
        let max = l as i32 + m;

        for k in min..max {
            value /= k as f64 + 1.
        }

        value
    } else if m < 0 {
        let mut value = (-1.0f64).powi(m);
        let min = l as i32 + m;
        let max = l as i32 - m;

        for k in min..max {
            value *= k as f64 + 1.
        }

        value
    } else {
        1.0
    }
}

pub fn normalization(l: u32, m: i32) -> f64 {
    assert!(m < 50);

    let norm2 = if m > 0 {
        let mut value = 1.0;
        let min = l as i32 - m;
        let max = l as i32 + m;

        for k in min..max {
            value /= k as f64 + 1.;
            if value.is_nan() || value < 0.{
                println!("{} {} {}", l, m, k)
            }
        }

        value
    } else if m < 0 {
        let mut value = 1.0;
        let min = l as i32 + m;
        let max = l as i32 - m;

        for k in min..max {
            value *= k as f64 + 1.
        }

        value
    } else {
        1.0
    };

    (norm2 * (2.0 * l as f64 + 1.)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn associated() {
        let values = associated_legendre_polynomials(150, -5, 1.0);
        println!("{:?}", values);

        let values = associated_legendre_polynomials(150, -5, -1.0);
        println!("{:?}", values);
        
        let values = associated_legendre_polynomials(150, -5, 0.0);
        println!("{:?}", values);
    }
}