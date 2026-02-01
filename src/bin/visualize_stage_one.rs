use argmin_testfunctions::{ackley, cross_in_tray, levy, rastrigin, rosenbrock};
use globalsearch::problem::Problem;
use globalsearch::scatter_search::ScatterSearch;
use globalsearch::types::{EvaluationError, OQNLPParams};
use ndarray::{Array1, Array2};
use plotly::common::{Marker, Mode, Title};
use plotly::{Contour, Layout, Plot, Scatter};
use std::error::Error;

#[derive(Clone)]
struct VisualProblem {
    name: String,
    obj_fn: fn(&[f64]) -> f64,
    bounds: [[f64; 2]; 2],
}

impl Problem for VisualProblem {
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        Ok((self.obj_fn)(x.as_slice().unwrap()))
    }
    fn variable_bounds(&self) -> Array2<f64> {
        let mut b = Array2::zeros((2, 2));
        b[[0, 0]] = self.bounds[0][0];
        b[[0, 1]] = self.bounds[0][1];
        b[[1, 0]] = self.bounds[1][0];
        b[[1, 1]] = self.bounds[1][1];
        b
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let problems = vec![
        VisualProblem {
            name: "Rosenbrock".to_string(),
            obj_fn: |x| rosenbrock(x),
            bounds: [[-2.0, 2.0], [-1.0, 3.0]],
        },
        VisualProblem {
            name: "Rastrigin".to_string(),
            obj_fn: |x| rastrigin(x),
            bounds: [[-5.12 + 1.0, 5.12 + 1.0], [-5.12 + 1.0, 5.12 + 1.0]],
        },
        VisualProblem {
            name: "Ackley".to_string(),
            obj_fn: |x| ackley(x),
            bounds: [[-5.0 + 1.0, 5.0 + 1.0], [-5.0 + 1.0, 5.0 + 1.0]],
        },
        VisualProblem {
            name: "Griewank".to_string(),
            obj_fn: |x| {
                let sum = x.iter().map(|&xi| xi.powi(2)).sum::<f64>() / 4000.0;
                let prod = x
                    .iter()
                    .enumerate()
                    .map(|(i, &xi)| (xi / ((i + 1) as f64).sqrt()).cos())
                    .product::<f64>();
                sum - prod + 1.0
            },
            bounds: [[-600.0 + 1.0, 600.0 + 1.0], [-600.0 + 1.0, 600.0 + 1.0]],
        },
        VisualProblem {
            name: "Levy".to_string(),
            obj_fn: |x| levy(x),
            bounds: [[-10.0, 10.0], [-10.0, 10.0]],
        },
        VisualProblem {
            name: "SixHumpCamel".to_string(),
            obj_fn: |x| {
                let x1 = x[0];
                let x2 = x[1];
                (4.0 - 2.1 * x1.powi(2) + x1.powi(4) / 3.0) * x1.powi(2)
                    + x1 * x2
                    + (-4.0 + 4.0 * x2.powi(2)) * x2.powi(2)
            },
            bounds: [[-3.0, 3.0], [-2.0, 2.0]],
        },
        VisualProblem {
            name: "CrossInTray".to_string(),
            obj_fn: |x| cross_in_tray(&[x[0], x[1]]),
            bounds: [[-10.0, 10.0], [-10.0, 10.0]],
        },
    ];

    for prob in problems {
        println!("Visualizing Stage 1 population for: {}", prob.name);

        let res = 80;
        let b = prob.bounds;
        let x_space: Vec<f64> = (0..res)
            .map(|i| b[0][0] + (b[0][1] - b[0][0]) * (i as f64 / (res - 1) as f64))
            .collect();
        let y_space: Vec<f64> = (0..res)
            .map(|i| b[1][0] + (b[1][1] - b[1][0]) * (i as f64 / (res - 1) as f64))
            .collect();

        let mut z = Vec::new();
        for y in &y_space {
            let mut row = Vec::new();
            for x in &x_space {
                row.push((prob.obj_fn)(&[*x, *y]));
            }
            z.push(row);
        }

        let mut plot = Plot::new();
        let num_runs = 6;
        let rows = 2;
        let cols = 3;

        for run in 0..num_runs {
            let params = OQNLPParams {
                seed: run as u64 * 82731,
                population_size: 20,
                ..OQNLPParams::default()
            };

            let ss = ScatterSearch::new(prob.clone(), params)?;
            let (ref_set, _) = ss.run()?;

            let px: Vec<f64> = ref_set.iter().map(|(p, _)| p[0]).collect();
            let py: Vec<f64> = ref_set.iter().map(|(p, _)| p[1]).collect();

            let trace_index = run + 1;
            let x_axis =
                if trace_index == 1 { "x".to_string() } else { format!("x{}", trace_index) };
            let y_axis =
                if trace_index == 1 { "y".to_string() } else { format!("y{}", trace_index) };

            let contour = Contour::new(x_space.clone(), y_space.clone(), z.clone())
                .show_scale(false)
                .show_legend(false)
                .x_axis(&x_axis)
                .y_axis(&y_axis);
            plot.add_trace(contour);

            let scatter = Scatter::new(px, py)
                .name(format!("Run {}", run + 1))
                .mode(Mode::Markers)
                .marker(Marker::new().size(5).color(plotly::common::color::NamedColor::Red))
                .x_axis(&x_axis)
                .y_axis(&y_axis);
            plot.add_trace(scatter);
        }

        let layout = Layout::new()
            .title(Title::with_text(format!(
                "{} - Stage 1 Population (6 Stochastic Runs)",
                prob.name
            )))
            .grid(
                plotly::layout::LayoutGrid::new()
                    .rows(rows)
                    .columns(cols)
                    .pattern(plotly::layout::GridPattern::Independent),
            )
            .width(1200)
            .height(800);

        plot.set_layout(layout);

        let _ = std::fs::create_dir_all("plots");

        let filename = format!("plots/{}_population.html", prob.name.to_lowercase());
        plot.write_html(&filename);
        println!("  Saved plot to {}", filename);
    }

    Ok(())
}
