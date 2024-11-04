use std::{fs::File, io::Write};

use crate::time_grid::TimeGrid;

#[derive(Clone)]
pub struct LossSaver {
    pub name: String,
    losses: Vec<f64>,
    current_frame: usize,
    frames_no: usize,
    total_frames: usize,
    step: f64,
    times: Vec<f64>,
}

impl LossSaver {
    pub fn new(name: String, frames_no: usize, time_grid: &TimeGrid) -> LossSaver {
        LossSaver {
            name,
            losses: Vec::with_capacity(frames_no),
            current_frame: 0,
            frames_no,
            total_frames: time_grid.step_no,
            step: time_grid.step,
            times: Vec::with_capacity(frames_no),
        }
    }

    pub fn monitor(&mut self, loss: f64) {
        // It is monitored two times during the propagation step.
        let frequency = 2 * self.total_frames / self.frames_no;

        if self.current_frame % frequency == 0 && self.current_frame / frequency < self.frames_no {
            self.losses.push(loss);
            self.times.push((self.current_frame as f64 + 1.0) / 2. * self.step);
        }

        self.current_frame += 1;
    }

    pub fn save(&self) {
        let mut buf = String::new();
        buf.push_str(&format!("time\tlosses for {}\n", self.name));
        for (time, loss) in self.times.iter().zip(self.losses.iter()) {
            buf.push_str(&format!("{}\t{}\n", time, loss));
        }

        let path = std::env::current_dir()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();

        let mut file = File::create(format!("{path}/{}.dat", self.name)).unwrap();
        file.write_all(buf.as_bytes()).unwrap();
    }
}
