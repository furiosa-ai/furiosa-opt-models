//! Tied fields: the subtrees they read, and what the leaves there leave for them.

use std::any::{Any, TypeId};
use std::collections::BTreeMap;

use super::Context;
use furiosa_opt_std::prelude::{HbmTensor, M, MaterializableScalar};

use crate::load::{Checkpoint, LoadError};

/// The subtrees tied fields read, and what their leaves left for them, by checkpoint name and
/// leaf type.
#[derive(Default)]
pub(super) struct Ties {
    targets: Vec<String>,
    shared: BTreeMap<(String, TypeId), Shared>,
}

/// What the leaf of a tie target left for the tied fields read after it.
enum Shared {
    /// A second owner of the leaf's storage, after a build.
    Owner(Box<dyn Any>),
    /// An in-place write, which reached the tied leaf of the leaf's type: that leaf is its storage.
    Written,
}

impl Context<'_> {
    /// Runs `visit` with `targets` (relative to the current name) leaving their leaves for the
    /// tied fields `visit` reads after them; what no tied field took is dropped after it.
    pub fn ties<R>(&mut self, targets: &[&str], visit: impl FnOnce(&mut Self) -> R) -> R {
        let base = self.name();
        let count = self.ties.targets.len();
        self.ties.targets.extend(
            targets
                .iter()
                .map(|target| Self::join(&[base.clone(), (*target).into()])),
        );
        let result = visit(self);
        let added = self.ties.targets.split_off(count);
        self.ties
            .shared
            .retain(|(name, _), _| !added.iter().any(|target| Self::under(name, target)));
        result
    }

    /// Refuses a config whose `key` says otherwise than `tied`, the tying the struct's type fixes.
    pub fn tying(&mut self, source: &Checkpoint, key: &str, tied: bool) -> Result<(), LoadError> {
        let (name, value) = self.config_value(source, key)?;
        let invalid = |reason: String| LoadError::InvalidData { name, reason };
        match value.as_bool() {
            Some(actual) if actual == tied => Ok(()),
            Some(actual) => Err(invalid(format!("{actual}, the model type ties: {tied}"))),
            None => Err(invalid(format!("{value} is not a boolean"))),
        }
    }

    /// The leaf of type `L` at the current position, as a leaf type's checkpoint build makes it. In
    /// a tied field it is a second owner of the target's storage, which `split` makes, when the
    /// target left one; otherwise `convert` builds it, and a leaf of a tie target leaves a second
    /// owner for the tied fields read after it.
    pub fn leaf<L: 'static>(
        &mut self,
        source: &Checkpoint,
        convert: impl FnOnce(&mut Self) -> Result<L, LoadError>,
        split: impl Fn(&Self, L) -> Result<(L, L), LoadError>,
    ) -> Result<L, LoadError> {
        self.compare(source)?;
        self.share(convert, split)
    }

    /// The leaf of type `L` at the current position, as [`Context::leaf`] makes it, for a leaf
    /// whose value is built on host: the caller compared the stored copy of a tied field first.
    pub fn share<L: 'static>(
        &mut self,
        convert: impl FnOnce(&mut Self) -> Result<L, LoadError>,
        split: impl Fn(&Self, L) -> Result<(L, L), LoadError>,
    ) -> Result<L, LoadError> {
        let key = (self.name(), TypeId::of::<L>());
        if self.cursor.tied.is_some() {
            return match self.ties.shared.remove(&key) {
                Some(Shared::Owner(owner)) => {
                    let owner = *owner.downcast::<L>().expect("kept under its type");
                    let (leaf, rest) = split(self, owner)?;
                    self.ties.shared.insert(key, Shared::Owner(Box::new(rest)));
                    Ok(leaf)
                }
                _ => convert(self),
            };
        }
        let leaf = convert(self)?;
        if !self.targets_here() {
            return Ok(leaf);
        }
        let (leaf, owner) = split(self, leaf)?;
        self.ties.shared.insert(key, Shared::Owner(Box::new(owner)));
        Ok(leaf)
    }

    /// The leaf of type `L` at the current position written in place by `write`, as a leaf type's
    /// update makes it, except in a tied field whose target's write in this traversal reached it,
    /// the storage the two share.
    pub fn overwrite<L: 'static>(
        &mut self,
        source: &Checkpoint,
        write: impl FnOnce(&mut Self) -> Result<(), LoadError>,
    ) -> Result<(), LoadError> {
        self.compare(source)?;
        let key = (self.name(), TypeId::of::<L>());
        if self.cursor.tied.is_some() {
            return match self.ties.shared.get(&key) {
                Some(Shared::Written) => Ok(()),
                _ => write(self),
            };
        }
        write(self)?;
        if self.targets_here() {
            self.ties.shared.insert(key, Shared::Written);
        }
        Ok(())
    }

    /// In a tied field, records the target's name and the stored copy under the field's own
    /// name, refusing a copy that differs from the target; a config leaf has neither.
    pub fn compare(&mut self, source: &Checkpoint) -> Result<(), LoadError> {
        let name = self.name();
        let Some((target, own)) = self
            .cursor
            .tied
            .as_ref()
            .filter(|_| self.cursor.config.is_none())
        else {
            return Ok(());
        };
        let rest = name
            .strip_prefix(target.as_str())
            .expect("a tied field reads below its target");
        let copy = Self::join(&[own.clone(), rest.trim_start_matches('.').into()]);
        self.read(&name);
        let Some(stored) = source.get(&copy) else {
            return Ok(());
        };
        self.read(&copy);
        let (name, value) = self.tensor(source)?;
        match (stored.dtype(), stored.shape(), stored.data())
            == (value.dtype(), value.shape(), value.data())
        {
            true => Ok(()),
            false => Err(LoadError::InvalidData {
                name: copy,
                reason: format!("tied copy differs from {name}"),
            }),
        }
    }

    /// Two owners of the storage of `leaf`, an HBM leaf at the current position, for the tied
    /// fields of its type.
    pub fn owners<D: MaterializableScalar, C: M + 'static, S: M + 'static>(
        &self,
        leaf: HbmTensor<D, C, S>,
    ) -> Result<(HbmTensor<D, C, S>, HbmTensor<D, C, S>), LoadError> {
        let buffer = leaf.into_buffer();
        buffer
            .alias(..)
            .and_then(HbmTensor::try_from)
            .and_then(|owner| Ok((HbmTensor::try_from(buffer)?, owner)))
            .map_err(|source| LoadError::Device {
                path: self.path(),
                source,
            })
    }

    /// Whether the current leaf lies in a tie target that a tied field reads later.
    fn targets_here(&self) -> bool {
        let name = self.name();
        self.cursor.tied.is_none()
            && self
                .ties
                .targets
                .iter()
                .any(|target| Self::under(&name, target))
    }
}
