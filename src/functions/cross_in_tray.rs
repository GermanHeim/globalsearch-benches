use super::{BenchmarkFn, RunResult};
use argmin_testfunctions::cross_in_tray;
use globalsearch::observers::Observer;
use globalsearch::oqnlp::OQNLP;
use globalsearch::problem::Problem;
use globalsearch::types::{EvaluationError, OQNLPParams};
use ndarray::{Array1, Array2};
use std::time::{Duration, Instant};

pub struct CrossInTray;

impl BenchmarkFn for CrossInTray {
    fn name(&self) -> &str {
        "CrossInTray"
    }

    fn supported_dims(&self, _default_dims: &[usize]) -> Vec<usize> {
        vec![2]
    }

    fn run(&self, _dim: usize, seed: u64) -> RunResult {
        let problem = CrossInTrayProblem;
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

        // Global min is -2.06261
        RunResult {
            success: (obj - (-2.06261)).abs() < 1e-4,
            runtime: duration,
            stage1_runtime: stage1_duration,
            stage2_runtime: stage2_duration,
            best_obj: obj,
            solution_set_size: solution_set.len(),
        }
    }
}

#[derive(Clone)]
struct CrossInTrayProblem;

impl Problem for CrossInTrayProblem {
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        let arr: [f64; 2] = x.as_slice().and_then(|s| s.try_into().ok()).ok_or_else(|| {
            EvaluationError::InvalidInput { reason: "Expected 2D point".to_string() }
        })?;
        Ok(cross_in_tray(&arr))
    }

    fn variable_bounds(&self) -> Array2<f64> {
        let mut bounds = Array2::zeros((2, 2));
        for i in 0..2 {
            bounds[[i, 0]] = -10.0;
            bounds[[i, 1]] = 10.0;
        }
        bounds
    }
}
