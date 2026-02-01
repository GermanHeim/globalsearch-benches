# GlobalSearch-rs Benches

A benchmarking and visualization suite for the [`globalsearch-rs`](https://crates.io/crates/globalsearch) Rust crate. This tool allows you to measure performance, compare algorithm versions, and visualize the behavior of different implementations.

## Features

- **Multi-attribute Performance Tracking**: Measures Success Rate (SR), Total Runtime, Stage-specific timing (Stage 1 vs. Stage 2), and Solution Set size.
- **A/B Version Comparison**: Automatically swap and compare your current `src` against a `src-new` implementation to validate performance improvements.
- **2D Population Visualization**: Generate contour plots with population overlays for 2D problems to visualize Stage 1 coverage and convergence.
- **Standard Benchmark Functions**: Integrated with `argmin_testfunctions` and custom implementations for classic optimization problems.

## Commands

To run the commands, this suite relies on being in the root directory of `globalsearch-rs`. You can get started by running:

```bash
cd globalsearch-rs
git clone https://github.com/GermanHeim/globalsearch-benches.git
```

### Performance Benchmarking

Run the standard benchmark suite:
```bash
cargo run --release --bin globalsearch-benches
```

### Full Comparison Benchmarking

Run a comparison between the current library and a new version (requires a `src-new` folder in the root directory of `globalsearch-rs`):
```bash
cargo run --release --bin compare
```

### 2D Population Analysis

Generate 2D landscape visualization files of Stage 1 generation:
```bash
cargo run --bin visualize
```

## Configuration & Arguments

The main runner supports several CLI flags:
- `--runs <N>`: Number of stochastic runs per dimension (default: 20).
- `--dim <D>`: Run a specific dimension instead of the default set.
- `--function <NAME>`: Run a specific benchmark function.
- `--save-json <PATH>`: Save results to a JSON file for later comparison.
- `--load-baseline <PATH>`: Load a previous JSON result to compare against.

## Core Components

### 1. Performance Runner (`main.rs`)

The core engine that runs standard benchmarks. It generates HTML reports with Plotly charts showing how metrics scale with problem dimensionality (10D, 50D, 100D).

### 2. Comparison Tool (`compare.rs`)

Automates the process of testing algorithm changes:
1. Runs a "baseline" benchmark on the current source.
2. Swaps the `src` directory with `src-new`.
3. Runs the benchmark again and generates comparative plots (Current vs. Baseline).
4. Restores the original directory structure.

### 3. Population Visualizer (`visualize.rs`)

Focuses on the stochastic nature of GlobalSearch. It runs multiple independent Stage 1 instances (different seeds) and plots them onto the objective function's contour map.

## Dependencies

- [globalsearch-rs](https://github.com/GermanHeim/globalsearch-rs)
- [plotly.rs](https://github.com/plotly/plotly.rs)
- [argmin_testfunctions](https://github.com/argmin-rs/argmin/tree/main/crates/argmin-testfunctions)
- [ndarray](https://github.com/rust-ndarray/ndarray)
- [serde](https://github.com/serde-rs/serde)
- [serde_json](https://github.com/serde-rs/json)
- [clap](https://github.com/clap-rs/clap)
- [rand](https://github.com/rust-random/rand)

## Project Structure

```text
globalsearch-benches/
├── src/
   ├── main.rs                     # Performance runner and plotting logic
   ├── bin/
   │   ├── compare.rs              # A/B comparison orchestrator
   │   └── visualize_stage_one.rs  # 2D landscape visualizer for Stage 1
   └── functions/                  # Benchmark problem implementations
```

## License

Distributed under the MIT License. See [`LICENSE.txt`](https://github.com/GermanHeim/globalsearch-benches/blob/main/LICENSE.txt) for more information.
