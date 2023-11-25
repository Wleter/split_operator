use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array, Array2, Array3, Axis, Dimension, Zip};
use rayon::prelude::*;

fn assign_iter<N: Dimension>(a: &mut Array<f64, N>, b: &Array2<f64>, axis: usize) {
    a.lanes_mut(Axis(axis))
        .into_iter()
        .par_bridge()
        .for_each(|mut lane| lane.assign(&b.dot(&lane)));
}

fn assign_zip<N: Dimension>(a: &mut Array<f64, N>, b: &Array2<f64>, axis: usize) {
    Zip::from(a.lanes_mut(Axis(axis))).par_for_each(|mut lane| lane.assign(&b.dot(&lane)));
}

fn lane_map(c: &mut Criterion) {
    let a = (0..1024 * 160 * 3)
        .map(|x| (x as f64) / 100.0)
        .collect::<Vec<f64>>();

    let m = Array2::<f64>::from_shape_vec((160, 160), a[0..160 * 160].to_vec()).unwrap();

    let matrix = Array3::<f64>::from_shape_vec((1024, 160, 3), a).unwrap();

    c.bench_function("assign", |c| {
        c.iter(|| {
            let _d = assign_iter(&mut matrix.clone(), &m, 1);
        })
    });

    c.bench_function("assign_zip", |c| {
        c.iter(|| {
            let _d = assign_zip(&mut matrix.clone(), &m, 1);
        })
    });
}

criterion_group!(benches, lane_map);
criterion_main!(benches);
