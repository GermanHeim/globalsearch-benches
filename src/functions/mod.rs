use std::time::Duration;

pub mod ackley;
pub mod cross_in_tray;
pub mod griewank;
pub mod levy;
pub mod rastrigin;
pub mod rosenbrock;
pub mod six_hump_camel;

pub struct RunResult {
    pub success: bool,
    pub runtime: Duration,
    pub stage1_runtime: Duration,
    pub stage2_runtime: Duration,
    pub best_obj: f64,
    pub solution_set_size: usize,
}

pub trait BenchmarkFn: Send + Sync {
    fn name(&self) -> &str;
    fn run(&self, dim: usize, seed: u64) -> RunResult;
    fn supported_dims(&self, default_dims: &[usize]) -> Vec<usize> {
        default_dims.to_vec()
    }
}
