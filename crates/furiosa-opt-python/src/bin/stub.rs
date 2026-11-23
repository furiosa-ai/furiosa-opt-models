//! Write the engine's type stubs into its Python package.

use pyo3_stub_gen::pyproject::PyProject;
use std::fs;
use std::{io, path::Path, process::Command};

fn main() -> pyo3_stub_gen::Result<()> {
    let info = furiosa_opt_python::stub_info()?;
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    // The wheel's pyproject sits in the engine's directory and builds this package.
    let project = PyProject::parse_toml(
        manifest
            .with_file_name("furiosa-opt-engine")
            .join("pyproject.toml"),
    )?;
    let (package, _) = project
        .module_name()
        .rsplit_once('.')
        .ok_or_else(|| io::Error::other("the extension needs a package"))?;
    let directory = project
        .python_source()
        .ok_or_else(|| io::Error::other("maturin needs python-source"))?
        .join(package);
    let patch = |text: String, anchor: &str, replacement: &str| {
        assert_eq!(text.matches(anchor).count(), 1, "stub anchor: {anchor}");
        text.replacen(anchor, replacement, 1)
    };
    let mut paths = Vec::new();
    for (name, module) in &info.modules {
        // Attribute docstrings indent their blank lines too, which ruff refuses.
        let mut text: String = module
            .format_with_config(false)
            .lines()
            .flat_map(|line| [line.trim_end(), "\n"])
            .collect();
        // Native classes that permit subclasses are disjoint bases for type checkers.
        for class in module.class.values().filter(|class| class.subclass) {
            if !text.lines().any(|line| line == "import typing_extensions") {
                text = patch(
                    text,
                    "\n__all__ = [",
                    "\nimport typing_extensions\n\n__all__ = [",
                );
            }
            let suffix = if class.bases.is_empty() { ":" } else { "(" };
            let anchor = format!("\nclass {}{suffix}", class.name);
            let replacement = format!("\n@typing_extensions.disjoint_base{anchor}");
            text = patch(text, &anchor, &replacement);
        }
        let file = match name.strip_prefix("engine.") {
            None => {
                text = patch(
                    text,
                    "def __getitem__(self, rows: slice)",
                    "def __getitem__(self, rows: slice, /)",
                );
                "__init__.pyi".to_string()
            }
            Some(submodule) => {
                text = text.replace("import engine\n", &format!("import {package} as engine\n"));
                text = text.replace("from engine import ", &format!("from {package} import "));
                format!("{submodule}.pyi")
            }
        };
        if name == "engine.decoder" {
            // The stub writes enums last, but a request's default `keep` names one.
            let start = text
                .find("\n@typing.final\nclass Keep(")
                .ok_or_else(|| io::Error::other("missing Keep enum"))?;
            let enums = text.split_off(start);
            text = patch(
                text,
                "\n@typing.final\nclass Request:",
                &format!("{enums}\n@typing.final\nclass Request:"),
            );
        }
        let path = directory.join(file);
        fs::write(&path, text)?;
        paths.push(path);
    }
    let root = manifest.join("../..");
    for args in [&["format"][..], &["check", "--fix"][..], &["format"][..]] {
        let status = Command::new("uv")
            .current_dir(&root)
            .args(["run", "--only-group", "dev", "ruff"])
            .args(args)
            .args(&paths)
            .status()?;
        if !status.success() {
            return Err(io::Error::other(format!("ruff {} failed", args[0])).into());
        }
    }
    Ok(())
}
