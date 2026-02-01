use super::{BenchmarkFn, RunResult};
use globalsearch::observers::Observer;
use globalsearch::oqnlp::OQNLP;
use globalsearch::problem::Problem;
use globalsearch::types::{EvaluationError, OQNLPParams};
use ndarray::{Array1, Array2};
use std::time::{Duration, Instant};

pub struct Griewank;

impl BenchmarkFn for Griewank {
    fn name(&self) -> &str {
        "Griewank"
    }

    fn run(&self, dim: usize, seed: u64) -> RunResult {
        let problem = GriewankProblem { dim };
        let params = OQNLPParams { seed, ..OQNLPParams::default() };

        let observer = Observer::new().with_stage1_tracking().with_stage2_tracking().with_timing();
        let mut optimizer =
            OQNLP::new(problem, params).expect("Failed to create OQNLP").add_observer(observer);

        let start = Instant::now();
        let solution_set = std::hint::black_box(optimizer.run()).expect("OQNLP run failed");
        let duration = start.elapsed();

        let obs = optimizer.observer().unwrap();
        let stage1_duration = obs
            .stage1_final()
            .and_then(|s| s.total_time())
            .map(Duration::from_secs_f64)
            .unwrap_or(Duration::ZERO);
        let stage2_duration = obs
            .stage2()
            .and_then(|s| s.total_time())
            .map(Duration::from_secs_f64)
            .unwrap_or(Duration::ZERO);

        let best_sol = solution_set.best_solution().expect("No solutions found");
        let obj = best_sol.objective;

        RunResult {
            success: obj < 1e-4,
            runtime: duration,
            stage1_runtime: stage1_duration,
            stage2_runtime: stage2_duration,
            best_obj: obj,
            solution_set_size: solution_set.len(),
        }
    }
}

fn griewank_local(x: &[f64]) -> f64 {
    let sum = x.iter().map(|&xi| xi.powi(2)).sum::<f64>() / 4000.0;
    let prod =
        x.iter().enumerate().map(|(i, &xi)| (xi / ((i + 1) as f64).sqrt()).cos()).product::<f64>();
    sum - prod + 1.0
}

#[derive(Clone)]
struct GriewankProblem {
    dim: usize,
}

impl Problem for GriewankProblem {
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        Ok(griewank_local(x.as_slice().unwrap()))
    }

    fn variable_bounds(&self) -> Array2<f64> {
        let mut bounds = Array2::zeros((self.dim, 2));
        for i in 0..self.dim {
            bounds[[i, 0]] = -600.0 + 1.0;
            bounds[[i, 1]] = 600.0 + 1.0;
        }
        bounds
    }
}
