use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{arr1, Array3};
use num::complex::Complex64;
use split_operator::{
    grid::Grid,
    propagation::Propagation,
    propagator::{one_dim_propagator::OneDimPropagator, Propagator},
    time_grid::TimeGrid,
    wave_function::WaveFunction,
};

fn dynamic_dispatch_benchmark(c: &mut Criterion) {
    let wf_array = Array3::<Complex64>::ones((100, 100, 100));
    let grid_x = Grid::new_linear_countable("x", 0.0, 1.0, 100, 0);
    let grid_y = Grid::new_linear_countable("y", 0.0, 1.0, 100, 1);
    let grid_z = Grid::new_linear_countable("z", 0.0, 1.0, 100, 2);

    let mut wf = WaveFunction::new(wf_array.clone(), vec![grid_x, grid_y, grid_z]);

    let mut unboxed = OneDimPropagator::new(&wf, 100, 0);
    unboxed.set_operator(arr1(&[Complex64::from(2.0); 100]));

    let mut boxed = Box::new(OneDimPropagator::new(&wf, 100, 0));
    boxed.set_operator(arr1(&[Complex64::from(2.0); 100]));

    let mut boxed2 = Box::new(OneDimPropagator::new(&wf, 100, 0));
    boxed2.set_operator(arr1(&[Complex64::from(2.0); 100]));
    let time_grid = TimeGrid {
        step: 1.0,
        step_no: 1,
        im_time: false,
    };
    let mut propagation = Propagation::new(wf.clone(), time_grid);
    propagation.add_propagator(boxed2);

    c.bench_function("boxed propagator", |c| {
        c.iter(|| {
            unboxed.apply(&mut wf);
        })
    });

    c.bench_function("unboxed propagator", |c| {
        c.iter(|| {
            boxed.apply(&mut wf);
        })
    });

    c.bench_function("boxed refcell propagator in propagation", |c| {
        c.iter(|| {
            propagation.propagate();
        })
    });
}

criterion_group!(benches, dynamic_dispatch_benchmark);
criterion_main!(benches);
