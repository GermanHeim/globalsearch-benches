use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

struct DirectoryGuard {
    root: PathBuf,
    swapped: bool,
}

impl DirectoryGuard {
    fn new(root: PathBuf) -> Self {
        Self { root, swapped: false }
    }

    fn swap(&mut self) -> std::io::Result<()> {
        let src = self.root.join("src");
        let src_new = self.root.join("src-new");
        let src_temp = self.root.join("src-original-temp");

        if !src_new.exists() {
            return Err(std::io::Error::new(std::io::ErrorKind::NotFound, "src-new not found"));
        }

        if src_temp.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "src-original-temp already exists. Previous run might have failed.",
            ));
        }

        println!("Renaming {:?} -> {:?}", src, src_temp);
        fs::rename(&src, &src_temp)?;

        println!("Renaming {:?} -> {:?}", src_new, src);
        fs::rename(&src_new, &src)?;

        self.swapped = true;
        Ok(())
    }

    fn restore(&mut self) -> std::io::Result<()> {
        if !self.swapped {
            return Ok(());
        }

        let src = self.root.join("src");
        let src_new = self.root.join("src-new");
        let src_temp = self.root.join("src-original-temp");

        println!("Restoring...");
        if src.exists() {
            println!("Renaming {:?} -> {:?}", src, src_new);
            fs::rename(&src, &src_new)?;
        }
        if src_temp.exists() {
            println!("Renaming {:?} -> {:?}", src_temp, src);
            fs::rename(&src_temp, &src)?;
        }
        self.swapped = false;
        Ok(())
    }
}

impl Drop for DirectoryGuard {
    fn drop(&mut self) {
        if self.swapped {
            println!("Auto-restoring directory structure...");
            if let Err(e) = self.restore() {
                eprintln!("Error restoring directories: {}", e);
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let current_dir = env::current_dir()?;
    let root_dir = current_dir.parent().ok_or("Cannot find parent dir")?.to_path_buf();

    let src_new = root_dir.join("src-new");
    if !src_new.exists() {
        println!("'src-new' folder not found at {:?}. Running standard benchmarks only.", src_new);
        run_bench(&["--release"])?;
        return Ok(());
    }

    println!("Found 'src-new'. Starting Comparison Benchmark Suite.");

    if Path::new("baseline_results.json").exists() {
        fs::remove_file("baseline_results.json")?;
    }

    println!("\n- Phase 1: Baseline (Original Source)");
    run_bench(&["--release", "--", "--save-json", "baseline_results.json"])?;

    {
        let mut guard = DirectoryGuard::new(root_dir.clone());
        println!("\n- Swapping src WITH src-new");
        guard.swap()?;

        // Clean dependent package to force recompilation
        // We need to run cargo clean -p globalsearch.
        // This command must be run within the globalsearch-benches directory (where we are).
        println!("Cleaning globalsearch package to ensure rebuild...");
        let clean_status = Command::new("cargo").args(["clean", "-p", "globalsearch"]).status()?;

        if !clean_status.success() {
            eprintln!(
                "Warning: 'cargo clean -p globalsearch' failed. Rebuild might not pick up changes."
            );
        }

        println!("\n- Phase 2: Comparison (New Source)");
        run_bench(&["--release", "--", "--load-baseline", "baseline_results.json"])?;

        println!("\n- Restoring directory structure");
        // guard dropped here automatically restores
    }

    println!("Comparison complete.");
    Ok(())
}

fn run_bench(args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    // We want to run the MAIN binary of this package.
    // Since we are currently running 'compare' binary which is inside the same package,
    // we can invoke cargo run --bin globalsearch-benches
    let mut cmd = Command::new("cargo");
    cmd.arg("run").arg("--bin").arg("globalsearch-benches");
    cmd.args(args);

    let status = cmd.status()?;

    if !status.success() {
        return Err("Benchmark command failed".into());
    }
    Ok(())
}
