pub mod harmonic_trap;

#[cfg(test)]
mod harmonic_tests {
    use std::time::Instant;

    use crate::harmonic_trap::harmonic_trap::HarmonicTrap;

    #[test]
    fn propagation_test() {
        let mut propagation = HarmonicTrap::new(false, "harmonic");

        let start = Instant::now();
        propagation.propagate();
        let elapsed = start.elapsed();
        println!("Elapsed time: {:?}", elapsed);

        propagation.save();
    }

    #[test]
    fn propagation_im_time() {
        let mut propagation = HarmonicTrap::new(true, "harmonic_imaginary");

        let start = Instant::now();
        propagation.propagate();
        let elapsed = start.elapsed();
        println!("Elapsed time: {:?}", elapsed);

        propagation.save();
    }
}
