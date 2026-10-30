//! The type stubs of the `engine` module, for the type checkers of frameworks.
//!
//! `src/bin/stub.rs` writes the `.pyi` files that tell a framework's type checker every class and
//! method of the module, such as the `__init__.pyi` of the wheel's `furiosa_opt_engine` package.
//! Each binding registers its stub information as an `inventory` item. Only the crate that submits
//! the items can gather them, so this crate provides `stub_info`.

use pyo3_stub_gen::{Result, StubGenConfig, StubInfo};

/// The stub information every binding of the `engine` module registers.
pub fn stub_info() -> Result<StubInfo> {
    StubInfo::from_project_root(
        "engine".into(),
        env!("CARGO_MANIFEST_DIR").into(),
        false,
        StubGenConfig::default(),
    )
}

#[cfg(test)]
mod tests {
    use pyo3::prelude::*;
    use std::collections::BTreeSet;

    /// Every class the registered Python module exports has stub information, and the stub
    /// information describes no other class, without opening a device.
    #[test]
    fn describes_registered_classes() -> pyo3_stub_gen::Result<()> {
        let info = super::stub_info()?;
        Python::initialize();
        Python::attach(|py| -> pyo3_stub_gen::Result<()> {
            let registered = PyModule::new(py, "engine")?;
            super::super::register(&registered)?;
            for (name, module) in &info.modules {
                let described = module
                    .class
                    .values()
                    .map(|class| class.name)
                    .chain(module.enum_.values().map(|enumeration| enumeration.name))
                    .map(str::to_string)
                    .chain(module.submodules.iter().cloned())
                    .collect::<BTreeSet<_>>();
                let registered = match name.strip_prefix("engine.") {
                    Some(name) => registered.getattr(name)?,
                    None => registered.clone().into_any(),
                };
                assert_eq!(
                    described,
                    registered
                        .getattr("__all__")?
                        .extract::<Vec<String>>()?
                        .into_iter()
                        .collect::<BTreeSet<_>>(),
                    "registered classes in {name}"
                );
            }
            Ok(())
        })
    }
}
