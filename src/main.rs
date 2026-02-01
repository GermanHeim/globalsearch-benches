use clap::Parser;
use functions::{
    BenchmarkFn, ackley::Ackley, cross_in_tray::CrossInTray, griewank::Griewank, levy::Levy,
    rastrigin::Rastrigin, rosenbrock::Rosenbrock, six_hump_camel::SixHumpCamel,
};
use plotly::common::{ErrorData, ErrorType, Mode, Title, Visible};
use plotly::layout::{Axis, GridPattern, Layout, LayoutGrid};
use plotly::{Plot, Scatter};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;

mod functions;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Benchmark function to run (all if not specified)
    #[arg(short, long)]
    function: Option<String>,

    /// Specific dimension to run (runs default set 10, 50, 100 if not specified)
    #[arg(short, long)]
    dim: Option<usize>,

    /// Number of runs per dimension
    #[arg(short, long, default_value_t = 20)]
    runs: usize,

    /// Save current stats to a JSON file
    #[arg(long)]
    save_json: Option<String>,

    /// Load baseline stats from a JSON file to compare against
    #[arg(long)]
    load_baseline: Option<String>,
}

#[derive(Serialize, Deserialize, Clone)]
struct StatPoint {
    dim: usize,
    success_rate: f64,
    avg_runtime_sec: f64,
    std_runtime_sec: f64,
    avg_stage1_sec: f64,
    avg_stage2_sec: f64,
    avg_solution_set_size: f64,
    std_solution_set_size: f64,
    avg_best_obj: f64,
}

#[derive(Serialize, Deserialize)]
struct AllStats {
    // Map function name to list of StatPoints
    data: std::collections::HashMap<String, Vec<StatPoint>>,
}

fn main() {
    let cli = Cli::parse();

    let all_functions: Vec<Box<dyn BenchmarkFn>> = vec![
        Box::new(Rosenbrock),
        Box::new(Rastrigin),
        Box::new(Ackley),
        Box::new(Griewank),
        Box::new(Levy),
        Box::new(SixHumpCamel),
        Box::new(CrossInTray),
    ];

    let functions_to_run: Vec<&Box<dyn BenchmarkFn>> = if let Some(name) = &cli.function {
        all_functions.iter().filter(|f| f.name().to_lowercase() == name.to_lowercase()).collect()
    } else {
        all_functions.iter().collect()
    };

    let default_dims = if let Some(d) = cli.dim {
        vec![d]
    } else {
        // Default dimensions
        vec![10, 50, 100]
    };

    let mut current_run_stats = AllStats { data: std::collections::HashMap::new() };

    for func in functions_to_run {
        println!("Running benchmark for: {}", func.name());
        let mut stats: Vec<StatPoint> = Vec::new();

        let func_dims = func.supported_dims(&default_dims);

        for &dim in &func_dims {
            println!("  Dimension: {}", dim);
            let mut runtimes = Vec::new();
            let mut stage1_runtimes = Vec::new();
            let mut stage2_runtimes = Vec::new();
            let mut solution_set_sizes = Vec::new();
            let mut successes = 0;
            let mut best_objs = Vec::new();

            for i in 0..cli.runs {
                let seed = i as u64 * 702983;
                let res = func.run(dim, seed);

                runtimes.push(res.runtime.as_secs_f64());
                stage1_runtimes.push(res.stage1_runtime.as_secs_f64());
                stage2_runtimes.push(res.stage2_runtime.as_secs_f64());
                solution_set_sizes.push(res.solution_set_size as f64);
                best_objs.push(res.best_obj);
                if res.success {
                    successes += 1;
                }
            }

            let success_rate = successes as f64 / cli.runs as f64;
            let avg_runtime = mean(&runtimes);
            let std_runtime = std_dev(&runtimes, avg_runtime);
            let avg_sol_size = mean(&solution_set_sizes);
            let std_sol_size = std_dev(&solution_set_sizes, avg_sol_size);
            let avg_obj = mean(&best_objs);

            println!(
                "    SR: {:.2}, Avg T: {:.4}s, Avg SolSize: {:.1}",
                success_rate, avg_runtime, avg_sol_size
            );

            stats.push(StatPoint {
                dim,
                success_rate,
                avg_runtime_sec: avg_runtime,
                std_runtime_sec: std_runtime,
                avg_stage1_sec: mean(&stage1_runtimes),
                avg_stage2_sec: mean(&stage2_runtimes),
                avg_solution_set_size: avg_sol_size,
                std_solution_set_size: std_sol_size,
                avg_best_obj: avg_obj,
            });
        }

        current_run_stats.data.insert(func.name().to_string(), stats.clone());
    }

    // Save results if requested
    if let Some(path) = &cli.save_json {
        let file = File::create(path).expect("Failed to create output JSON file");
        serde_json::to_writer_pretty(file, &current_run_stats).expect("Failed to write JSON");
        println!("Saved stats to {}", path);
    }

    // Load baseline if requested and generate plots
    let baseline_stats = if let Some(path) = &cli.load_baseline {
        let file = File::open(path).expect("Failed to open baseline JSON file");
        let reader = BufReader::new(file);
        let loaded: AllStats =
            serde_json::from_reader(reader).expect("Failed to parse baseline JSON");
        println!("Loaded baseline stats from {}", path);
        Some(loaded)
    } else {
        None
    };

    // Generate plots (comparing if baseline exists)
    for (func_name, current_stats) in &current_run_stats.data {
        let baseline = baseline_stats.as_ref().and_then(|b| b.data.get(func_name));
        generate_plots(func_name, current_stats, baseline);
    }
}

fn mean(data: &[f64]) -> f64 {
    let sum: f64 = data.iter().sum();
    sum / data.len() as f64
}

fn std_dev(data: &[f64], mean: f64) -> f64 {
    let variance = data
        .iter()
        .map(|value| {
            let diff = mean - *value;
            diff * diff
        })
        .sum::<f64>()
        / data.len() as f64;
    variance.sqrt()
}

fn generate_plots(func_name: &str, current: &[StatPoint], baseline: Option<&Vec<StatPoint>>) {
    let _ = std::fs::create_dir_all("plots");

    let x_vals: Vec<usize> = current.iter().map(|s| s.dim).collect();

    let mut plot = Plot::new();

    let layout = Layout::new()
        .title(Title::with_text(format!("{} Benchmarks", func_name)))
        .grid(LayoutGrid::new().rows(3).columns(1).pattern(GridPattern::Independent))
        // Success Rate
        .x_axis(Axis::new().title(Title::with_text("Dimension")))
        .y_axis(Axis::new().title(Title::with_text("Success Rate")))
        // Runtime
        .x_axis2(Axis::new().title(Title::with_text("Dimension")))
        .y_axis2(Axis::new().title(Title::with_text("Time (s)")))
        // Solution Size
        .x_axis3(Axis::new().title(Title::with_text("Dimension")))
        .y_axis3(Axis::new().title(Title::with_text("Solution Set Size")))
        .height(1200);

    plot.set_layout(layout);

    let current_sr: Vec<f64> = current.iter().map(|s| s.success_rate).collect();
    plot.add_trace(
        Scatter::new(x_vals.clone(), current_sr)
            .name("Current SR")
            .mode(Mode::LinesMarkers)
            .x_axis("x")
            .y_axis("y"),
    );

    if let Some(base) = baseline {
        let base_sr: Vec<f64> = base.iter().map(|s| s.success_rate).collect();
        plot.add_trace(
            Scatter::new(x_vals.clone(), base_sr)
                .name("Baseline SR")
                .mode(Mode::LinesMarkers)
                .x_axis("x")
                .y_axis("y"),
        );
    }

    let current_rt: Vec<f64> = current.iter().map(|s| s.avg_runtime_sec).collect();
    let current_std_rt: Vec<f64> = current.iter().map(|s| s.std_runtime_sec).collect();
    let current_s1: Vec<f64> = current.iter().map(|s| s.avg_stage1_sec).collect();
    let current_s2: Vec<f64> = current.iter().map(|s| s.avg_stage2_sec).collect();

    plot.add_trace(
        Scatter::new(x_vals.clone(), current_rt)
            .name("Current Total RT")
            .mode(Mode::LinesMarkers)
            .error_y(ErrorData::new(ErrorType::Data).array(current_std_rt))
            .x_axis("x2")
            .y_axis("y2"),
    );

    plot.add_trace(
        Scatter::new(x_vals.clone(), current_s1)
            .name("Current Stage 1 RT")
            .mode(Mode::LinesMarkers)
            .visible(Visible::LegendOnly)
            .x_axis("x2")
            .y_axis("y2"),
    );
    plot.add_trace(
        Scatter::new(x_vals.clone(), current_s2)
            .name("Current Stage 2 RT")
            .mode(Mode::LinesMarkers)
            .visible(Visible::LegendOnly)
            .x_axis("x2")
            .y_axis("y2"),
    );

    if let Some(base) = baseline {
        let base_rt: Vec<f64> = base.iter().map(|s| s.avg_runtime_sec).collect();
        let base_std_rt: Vec<f64> = base.iter().map(|s| s.std_runtime_sec).collect();
        plot.add_trace(
            Scatter::new(x_vals.clone(), base_rt)
                .name("Baseline Total RT")
                .mode(Mode::LinesMarkers)
                .error_y(ErrorData::new(ErrorType::Data).array(base_std_rt))
                .x_axis("x2")
                .y_axis("y2"),
        );

        let base_s1: Vec<f64> = base.iter().map(|s| s.avg_stage1_sec).collect();
        let base_s2: Vec<f64> = base.iter().map(|s| s.avg_stage2_sec).collect();
        plot.add_trace(
            Scatter::new(x_vals.clone(), base_s1)
                .name("Baseline Stage 1 RT")
                .mode(Mode::LinesMarkers)
                .visible(Visible::LegendOnly)
                .x_axis("x2")
                .y_axis("y2"),
        );
        plot.add_trace(
            Scatter::new(x_vals.clone(), base_s2)
                .name("Baseline Stage 2 RT")
                .mode(Mode::LinesMarkers)
                .visible(Visible::LegendOnly)
                .x_axis("x2")
                .y_axis("y2"),
        );
    }

    let current_sz: Vec<f64> = current.iter().map(|s| s.avg_solution_set_size).collect();
    let current_std_sz: Vec<f64> = current.iter().map(|s| s.std_solution_set_size).collect();
    plot.add_trace(
        Scatter::new(x_vals.clone(), current_sz)
            .name("Current SolSize")
            .mode(Mode::LinesMarkers)
            .error_y(ErrorData::new(ErrorType::Data).array(current_std_sz))
            .x_axis("x3")
            .y_axis("y3"),
    );

    if let Some(base) = baseline {
        let base_sz: Vec<f64> = base.iter().map(|s| s.avg_solution_set_size).collect();
        let base_std_sz: Vec<f64> = base.iter().map(|s| s.std_solution_set_size).collect();
        plot.add_trace(
            Scatter::new(x_vals.clone(), base_sz)
                .name("Baseline SolSize")
                .mode(Mode::LinesMarkers)
                .error_y(ErrorData::new(ErrorType::Data).array(base_std_sz))
                .x_axis("x3")
                .y_axis("y3"),
        );
    }

    let filename = format!("plots/{}_benchmark.html", func_name.to_lowercase());
    plot.write_html(filename);
}
