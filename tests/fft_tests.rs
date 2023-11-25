#[cfg(test)]
mod fft_tests {
    use ndarray::arr2;
    use num::complex::Complex64;
    use split_operator::{
        grid::Grid,
        propagator::{diagonalization::Diagonalization, fft_diagonalization::FFTDiagonalization},
        wave_function::WaveFunction,
    };

    #[test]
    fn test_fft_diagonalization() {
        let grid1 = Grid::new_linear_countable("a", 0.0, 1.0, 4, 0);
        let grid2 = Grid::new_linear_countable("b", 0.0, 1.0, 4, 1);
        let wf_array = arr2(&[
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
            [
                Complex64::new(5.0, 0.0),
                Complex64::new(6.0, 0.0),
                Complex64::new(7.0, 0.0),
                Complex64::new(8.0, 0.0),
            ],
            [
                Complex64::new(9.0, 0.0),
                Complex64::new(10.0, 0.0),
                Complex64::new(11.0, 0.0),
                Complex64::new(12.0, 0.0),
            ],
            [
                Complex64::new(13.0, 0.0),
                Complex64::new(14.0, 0.0),
                Complex64::new(15.0, 0.0),
                Complex64::new(16.0, 0.0),
            ],
        ]);

        let mut wf = WaveFunction::new(wf_array.clone(), vec![grid1.clone(), grid2.clone()]);
        let mut fft_diag = FFTDiagonalization::new(&wf, &grid1, "a_fft");

        fft_diag.diagonalize(&mut wf);
        let wf_array_transformed = wf.array.clone();
        fft_diag.inverse_diagonalize(&mut wf);

        let transformed_array = arr2(&[
            [
                Complex64::new(14.0, 0.0),
                Complex64::new(16.0, 0.0),
                Complex64::new(18.0, 0.0),
                Complex64::new(20.0, 0.0),
            ],
            [
                Complex64::new(-4.0, 4.0),
                Complex64::new(-4.0, 4.0),
                Complex64::new(-4.0, 4.0),
                Complex64::new(-4.0, 4.0),
            ],
            [
                Complex64::new(-4.0, 0.0),
                Complex64::new(-4.0, 0.0),
                Complex64::new(-4.0, 0.0),
                Complex64::new(-4.0, 0.0),
            ],
            [
                Complex64::new(-4.0, -4.0),
                Complex64::new(-4.0, -4.0),
                Complex64::new(-4.0, -4.0),
                Complex64::new(-4.0, -4.0),
            ],
        ]);

        assert_eq!(wf_array, wf.array);
        assert_eq!(transformed_array, wf_array_transformed)
    }

    #[test]
    fn test_transformed_norm() {
        let grid1 = Grid::new_linear_countable("a", 0.0, 1.0, 4, 0);
        let grid2 = Grid::new_linear_countable("b", 0.0, 1.0, 4, 1);
        let wf_array = arr2(&[
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
            [
                Complex64::new(5.0, 0.0),
                Complex64::new(6.0, 0.0),
                Complex64::new(7.0, 0.0),
                Complex64::new(8.0, 0.0),
            ],
            [
                Complex64::new(9.0, 0.0),
                Complex64::new(10.0, 0.0),
                Complex64::new(11.0, 0.0),
                Complex64::new(12.0, 0.0),
            ],
            [
                Complex64::new(13.0, 0.0),
                Complex64::new(14.0, 0.0),
                Complex64::new(15.0, 0.0),
                Complex64::new(16.0, 0.0),
            ],
        ]);

        let mut wf = WaveFunction::new(wf_array, vec![grid1.clone(), grid2.clone()]);
        let mut fft_diag = FFTDiagonalization::new(&wf, &grid1, "a_fft");

        let norm1 = wf.norm();
        fft_diag.diagonalize(&mut wf);
        let norm2 = wf.norm();
        fft_diag.inverse_diagonalize(&mut wf);
        let norm3 = wf.norm();

        println!("{}", norm1);
        println!("{}", norm2);
        println!("{}", norm3);

        assert_eq!(norm1, norm3);
        assert_eq!(norm1, norm2);
    }
}
