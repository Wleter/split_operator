use std::{sync::Arc, time::Duration};

use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array, Array1, Array3, Axis, Dimension, Zip};
use num::complex::Complex64;
use rayon::prelude::*;
use rustfft::{Fft, FftPlanner};

fn assign_iter<N: Dimension>(
    a: &mut Array<Complex64, N>,
    fft: &Box<Arc<dyn Fft<f64>>>,
    axis: usize,
) {
    let dim_size_sqrt = (a.raw_dim()[axis] as f64).sqrt();

    a.lanes_mut(Axis(axis))
        .into_iter()
        .par_bridge()
        .for_each(|mut lane| {
            let mut temp = lane.to_vec();
            fft.process(&mut temp);

            lane.iter_mut().zip(temp.iter()).for_each(|(dest, src)| {
                *dest = *src / dim_size_sqrt;
            });
        });
}

fn assign_zip<N: Dimension>(
    a: &mut Array<Complex64, N>,
    fft: &Box<Arc<dyn Fft<f64>>>,
    axis: usize,
) {
    let dim_size_sqrt = (a.raw_dim()[axis] as f64).sqrt();

    Zip::from(a.lanes_mut(Axis(axis))).par_for_each(|mut lane| {
        let mut temp = lane.to_vec();
        fft.process(&mut temp);

        lane.iter_mut().zip(temp.iter()).for_each(|(dest, src)| {
            *dest = *src / dim_size_sqrt;
        });
    })
}

fn assign_double_zip<N: Dimension>(
    a: &mut Array<Complex64, N>,
    fft: &Box<Arc<dyn Fft<f64>>>,
    axis: usize,
) {
    let dim_size_sqrt = (a.raw_dim()[axis] as f64).sqrt();

    Zip::from(a.lanes_mut(Axis(axis))).par_for_each(|lane| {
        let mut temp = lane.to_vec();
        fft.process(&mut temp);

        Zip::from(lane).and(&temp).for_each(|dest, src| {
            *dest = *src / dim_size_sqrt;
        });
    })
}

fn assign_iter_with<N: Dimension>(
    a: &mut Array<Complex64, N>,
    fft: &Box<Arc<dyn Fft<f64>>>,
    axis: usize,
) {
    let dim_size_sqrt = (a.raw_dim()[axis] as f64).sqrt();
    let temp = Array1::<Complex64>::zeros(a.raw_dim()[axis]);

    Zip::from(a.lanes_mut(Axis(axis)))
        .into_par_iter()
        .for_each_with(temp, |temp, lane| {
            temp.assign(&lane.0);
            fft.process(temp.as_slice_mut().unwrap());
            // Put back everything in the lane
            Zip::from(lane.0)
                .and(temp)
                .for_each(|l, t| *l = *t / dim_size_sqrt);
        });
}

fn fft_lane_map(c: &mut Criterion) {
    let a = (0..1024 * 160 * 3)
        .map(|x| Complex64::from((x as f64) / 100.0))
        .collect::<Vec<Complex64>>();

    let fft = Box::new(FftPlanner::new().plan_fft_forward(1024));

    let matrix = Array3::<Complex64>::from_shape_vec((1024, 160, 3), a).unwrap();

    c.bench_function("assign_iter", |c| {
        c.iter(|| {
            let _d = assign_iter(&mut matrix.clone(), &fft, 0);
        })
    });

    c.bench_function("assign_zip", |c| {
        c.iter(|| {
            let _d = assign_zip(&mut matrix.clone(), &fft, 0);
        })
    });

    c.bench_function("assign_double_zip", |c| {
        c.iter(|| {
            let _d = assign_double_zip(&mut matrix.clone(), &fft, 0);
        })
    });

    c.bench_function("assign_iter_with", |c| {
        c.iter(|| {
            let _d = assign_iter_with(&mut matrix.clone(), &fft, 0);
        })
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = fft_lane_map
);
criterion_main!(benches);
