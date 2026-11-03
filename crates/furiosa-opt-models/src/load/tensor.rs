//! A checkpoint tensor made into a leaf's host value: shape checked, floats converted, laid out
//! as the leaf's chips hold it.
//!
//! - A checkpoint stores a leaf's values over the axes its element mapping names, in the order it
//!   first names them, each whole; a stored dimension may hold a run of them, as `q_proj`'s rows
//!   hold the leaf's query heads and their features.
//! - A chip mapping naming one of them splits it, a share a chip (`m![W / 1024]` with
//!   `m![W % 1024, In]`); one naming none holds every value on every chip, read as one copy.

use furiosa_opt_std::prelude::{
    AxisTerm, HostTensor, Index, M, Mapping, MappingExt, MaterializableScalar, SequencerMode,
    sequence,
};
use furiosa_opt_std::runtime::HostBuffer;
use std::borrow::Cow;

use safetensors::Dtype;
use safetensors::tensor::TensorView;

use super::{Checkpoint, Context, LoadError};
use crate::tensor::memory::dtype;

/// `host` laid out as `To` a contiguous run at a time, or none when its layout is `To`'s or no walk
/// of whole-byte runs reads each cell once (a broadcast, inner padding): the upload lays those out.
pub(crate) fn laid_out<D: MaterializableScalar, E: M, To: M>(
    host: &HostTensor<D, E>,
) -> Option<HostTensor<D, To>> {
    let (from, to) = (E::to_value(), To::to_value());
    if from.normalize() == to.normalize() {
        return None;
    }
    let (live, block, padded) = unpadded(to);
    if live.size() != from.size() {
        return None;
    }
    let walk = sequence(&[&from], &[&live], SequencerMode::Read)
        .ok()?
        .remove(0);
    let bytes = |cells: usize| {
        (cells * D::BITS)
            .is_multiple_of(8)
            .then_some(cells * D::BITS / 8)
    };
    // Innermost first: the run, then each outer step and its stride in the source; a run spanning
    // several blocks of `To`'s innermost factor goes a block at a time.
    let mut wheels = walk
        .0
        .iter()
        .map(|(_, entry)| (entry.mapping.size(), entry.memory_stride));
    let mut steps = match wheels.next()? {
        (cells, 1) if block.is_multiple_of(cells) => vec![(1, cells)],
        (cells, 1) if cells.is_multiple_of(block) => vec![(cells / block, block), (1, block)],
        _ => return None,
    };
    let run = steps.pop()?.1;
    steps.extend(wheels);
    let steps: Vec<(usize, usize)> = (steps.into_iter())
        .map(|(size, stride)| Some((size, bytes(stride).filter(|&stride| stride > 0)?)))
        .collect::<Option<_>>()?;
    let (run, block, padded) = (bytes(run)?, bytes(block)?, bytes(padded)?);
    let source: Vec<u8> = host.clone().into_buffer().into();
    let mut image = vec![0; bytes(To::SIZE)?];
    for at in 0..bytes(live.size())? / run {
        let (offset, _) = (steps.iter()).fold((0, at), |(offset, rest), &(size, stride)| {
            (offset + rest % size * stride, rest / size)
        });
        let cell = at * run / block * padded + at * run % block;
        image[cell..cell + run].copy_from_slice(&source[offset..offset + run]);
    }
    HostTensor::try_from(HostBuffer::from_vec(image)).ok()
}

/// `mapping` without the padding of its innermost factor, with that factor's live and padded
/// cells: equal when it is not padded.
fn unpadded(mapping: Mapping) -> (Mapping, usize, usize) {
    match mapping {
        Mapping::Pair { left, right } => {
            let (right, live, padded) = unpadded((*right).clone());
            ((*left).clone().pair(right), live, padded)
        }
        Mapping::Padding { inner, padding, .. } => {
            let inner = (*inner).clone();
            let live = inner.size();
            (inner, live, padding)
        }
        other => {
            let size = other.size();
            (other, size, size)
        }
    }
}

impl Checkpoint {
    /// The bytes of the float tensor `view` of `name` as BF16: its own, or its FP32 values rounded
    /// as a leaf converts them.
    pub(crate) fn bf16<'v>(
        view: &'v TensorView<'_>,
        name: &str,
    ) -> Result<Cow<'v, [u8]>, LoadError> {
        Self::admit_bf16(view.dtype(), name)?;
        Ok(match view.dtype() {
            Dtype::F32 => Cow::Owned(
                (view.data().as_chunks::<4>().0.iter())
                    .flat_map(|&bytes| {
                        half::bf16::from_f32(f32::from_le_bytes(bytes)).to_le_bytes()
                    })
                    .collect(),
            ),
            _ => Cow::Borrowed(view.data()),
        })
    }

    /// That the float tensor `name`, stored as `dtype`, converts to BF16 as [`Self::bf16`] does:
    /// BF16 itself, or FP32.
    pub(crate) fn admit_bf16(dtype: Dtype, name: &str) -> Result<(), LoadError> {
        match dtype {
            Dtype::BF16 | Dtype::F32 => Ok(()),
            dtype => Err(LoadError::InvalidData {
                name: name.into(),
                reason: format!("stored as {dtype:?}, the leaf holds BF16"),
            }),
        }
    }

    /// The host value, laid out as `E`, of the leaf of element mapping `S` at the context's
    /// position: the tensor at its name, or the value of its config key in every element.
    pub(crate) fn host<D: MaterializableScalar, S: M, E: M>(
        &self,
        context: &mut Context<'_>,
    ) -> Result<HostTensor<D, E>, LoadError> {
        if let Some(key) = context.config_key().map(str::to_owned) {
            let (name, value) = context.config_number(self, &key)?;
            let shape: Vec<usize> = Self::stored(&S::to_value())
                .iter()
                .map(|axis| axis.size)
                .collect();
            let data = (value as f32).to_le_bytes().repeat(shape.iter().product());
            let view = TensorView::new(Dtype::F32, shape, &data).expect("one value per element");
            return Self::value::<D, S, E>(&view, &name);
        }
        let (name, view) = context.tensor(self)?;
        Self::value::<D, S, E>(&view, &name)
    }

    /// The leaf at the context's position, laid out as `E`, checked as `host` checks it without
    /// reading values: for other crates' tests of a published checkpoint, declared or stored.
    #[cfg(any(test, feature = "test-support"))]
    pub(super) fn check<D: MaterializableScalar, S: M, E: M>(
        &self,
        context: &mut Context<'_>,
    ) -> Result<(), LoadError> {
        if context.config_key().is_some() {
            return self.host::<D, S, E>(context).map(drop);
        }
        let (name, (dtype, shape)) = context.header(self)?;
        Self::admit::<D, S, E>(dtype, shape, &name).map(drop)
    }

    /// The dtype in which a leaf of `D`, element mapping `S` and layout `E` holds tensor `name`:
    /// each dimension of `shape` is a run of the stored axes, each whole in `E`, and `from` is `D`
    /// or a float that converts.
    fn admit<D: MaterializableScalar, S: M, E: M>(
        from: Dtype,
        shape: &[usize],
        name: &str,
    ) -> Result<Dtype, LoadError> {
        let invalid = |reason: String| LoadError::InvalidData {
            name: name.into(),
            reason,
        };
        let stored = Self::stored(&S::to_value());
        let leaf: Vec<usize> = stored.iter().map(|axis| axis.size).collect();
        let one = |shape: &[usize]| shape.len() <= 1 && shape.iter().product::<usize>() == 1;
        let runs = || {
            let mut sizes = leaf.iter();
            shape.iter().all(|&dimension| {
                let mut run = 1;
                while run < dimension {
                    match sizes.next() {
                        Some(size) => run *= size,
                        None => return false,
                    }
                }
                run == dimension
            }) && sizes.next().is_none()
        };
        if !runs() && !(one(shape) && one(&leaf)) {
            return Err(invalid(format!("shape {shape:?}, the leaf holds {leaf:?}")));
        }
        let held = E::to_value().axes();
        let whole = |axis: &AxisTerm| {
            let mut pieces: Vec<_> = held.iter().filter(|h| h.symbol == axis.symbol).collect();
            pieces.sort_by_key(|piece| piece.stride);
            let tiled = pieces.iter().try_fold(1, |covered, piece| {
                (piece.stride == covered).then_some(covered * piece.modulo)
            });
            tiled == Some(axis.size)
        };
        if let Some(axis) = stored.iter().find(|axis| !whole(axis)) {
            return Err(LoadError::Unsupported {
                name: name.into(),
                reason: format!("the leaf holds part of the axis {}", axis.symbol),
            });
        }
        let to = dtype::<D>().map_err(|error| LoadError::Unsupported {
            name: name.into(),
            reason: error.to_string(),
        })?;
        let float = |dtype| matches!(dtype, Dtype::F32 | Dtype::BF16 | Dtype::F16);
        match from == to || (float(from) && float(to)) {
            true => Ok(to),
            false => Err(invalid(format!(
                "stored as {from:?}, the leaf holds {to:?}"
            ))),
        }
    }

    /// The host value of a leaf of `D` and element mapping `S` holding the checkpoint tensor
    /// `view`, laid out as `E` (`S` for one copy, `m![{ C }, { S }]` for each chip's share), once
    /// `admit` admits it: float values converted to `D`, zero in the padding.
    pub(super) fn value<D: MaterializableScalar, S: M, E: M>(
        view: &TensorView<'_>,
        name: &str,
    ) -> Result<HostTensor<D, E>, LoadError> {
        let to = Self::admit::<D, S, E>(view.dtype(), view.shape(), name)?;
        let from = view.dtype();
        let data = view.data();
        let data = match from == to {
            true => data.to_vec(),
            false => {
                let values =
                    Self::floats(view).expect("`admit` takes another dtype only from a float one");
                match to {
                    Dtype::F32 => values.into_iter().flat_map(f32::to_le_bytes).collect(),
                    Dtype::BF16 => values
                        .into_iter()
                        .flat_map(|v| half::bf16::from_f32(v).to_le_bytes())
                        .collect(),
                    Dtype::F16 => values
                        .into_iter()
                        .flat_map(|v| half::f16::from_f32(v).to_le_bytes())
                        .collect(),
                    _ => unreachable!("`admit` takes another dtype only into a float one"),
                }
            }
        };
        let width = to.size();
        let (stored, layout) = (Self::stored(&S::to_value()), E::to_value());
        let ordered = Mapping::from_terms(stored.iter().map(|axis| axis.to_term()));
        let image = match layout.normalize() == ordered.normalize() {
            true => data,
            false => {
                let mut image = vec![0; layout.size() * width];
                let cells = layout.iter(&stored, &Index::new(), true).enumerate();
                for (cell, at) in cells.filter_map(|(cell, at)| Some((cell, at?))) {
                    image[cell * width..][..width].copy_from_slice(&data[at * width..][..width]);
                }
                image
            }
        };
        HostTensor::try_from(HostBuffer::from_vec(image)).map_err(|error| LoadError::InvalidData {
            name: name.into(),
            reason: error.to_string(),
        })
    }

    /// The values of `view` as FP32, when it stores FP32, BF16 or F16 ones.
    pub(crate) fn floats(view: &TensorView<'_>) -> Option<Vec<f32>> {
        let data = view.data();
        match view.dtype() {
            Dtype::F32 => Some(
                (data.as_chunks::<4>().0.iter())
                    .map(|&b| f32::from_le_bytes(b))
                    .collect(),
            ),
            Dtype::BF16 => Some(
                (data.as_chunks::<2>().0.iter())
                    .map(|&b| half::bf16::from_le_bytes(b).to_f32())
                    .collect(),
            ),
            Dtype::F16 => Some(
                (data.as_chunks::<2>().0.iter())
                    .map(|&b| half::f16::from_le_bytes(b).to_f32())
                    .collect(),
            ),
            _ => None,
        }
    }

    /// Whether the chips of `C` split an axis a leaf of element mapping `S` stores, a share a chip;
    /// chips naming none of its axes each hold every value.
    pub(super) fn splits<C: M, S: M>() -> bool {
        let chips = C::to_value().idents();
        Self::stored(&S::to_value())
            .iter()
            .any(|axis| chips.contains(&axis.symbol))
    }

    /// The axes a checkpoint stores the values of a leaf laid out as `element` over, outermost
    /// first: those it names, in the order it first names them, each whole.
    pub(super) fn stored(element: &Mapping) -> Vec<AxisTerm> {
        let mut axes: Vec<AxisTerm> = Vec::new();
        let mut pending = vec![element];
        while let Some(mapping) = pending.pop() {
            match mapping {
                Mapping::Symbol { symbol, size }
                    if axes.iter().all(|axis| axis.symbol != *symbol) =>
                {
                    axes.push(AxisTerm {
                        symbol: *symbol,
                        size: *size,
                        stride: 1,
                        modulo: *size,
                    });
                }
                Mapping::Stride { inner, .. }
                | Mapping::Modulo { inner, .. }
                | Mapping::Resize { inner, .. }
                | Mapping::Padding { inner, .. } => pending.push(&**inner),
                Mapping::Pair { left, right } => pending.extend([&**right, &**left]),
                Mapping::Symbol { .. } | Mapping::Broadcast { .. } => {}
            }
        }
        axes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use furiosa_opt_std::prelude::*;

    axes![O = 2, I = 4];

    /// A chip mapping splitting an axis gives each chip its share of the whole stored tensor; a
    /// leaf holding part of an axis on every chip is refused.
    #[test]
    fn chips_hold_their_shares_of_the_stored_tensor() {
        let data: Vec<u8> = (0..8).flat_map(|at| (at as f32).to_le_bytes()).collect();
        let tensor = TensorView::new(Dtype::F32, vec![2, 4], &data).unwrap();
        let split = Checkpoint::value::<f32, m![O, I % 2], m![I / 2, O, I % 2]>(&tensor, "w");
        assert_eq!(
            split.unwrap().into_vec(),
            [0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0]
        );
        let part = Checkpoint::value::<f32, m![O, I % 2], m![O, I % 2]>(&tensor, "w").err();
        assert!(
            matches!(&part, Some(LoadError::Unsupported { name, reason })
                if name == "w" && reason == "the leaf holds part of the axis I"),
            "{part:?}"
        );
    }

    /// A layout whose cells a walk of whole-byte runs reads gets each cell where it places it, for
    /// FP32, E2M1 codes two a byte and a padded innermost factor; runs splitting a byte and inner
    /// padding are declined.
    #[test]
    fn lays_out_whole_runs_or_declines() {
        type Grouped = m![O / 2, I / 2, O % 2, I % 2];
        let at = |o: usize, i: usize| ((o / 2 * 2 + i / 2) * 2 + o % 2) * 2 + i % 2;
        let values: Vec<f32> = (0..8).map(|value| value as f32).collect();
        let grouped = laid_out::<f32, m![O, I], Grouped>(&HostTensor::from_vec(values.clone()));
        let mut expected = vec![0.0; 8];
        for (o, i) in (0..2).flat_map(|o| (0..4).map(move |i| (o, i))) {
            expected[at(o, i)] = values[o * 4 + i];
        }
        assert_eq!(grouped.unwrap().into_vec(), expected);
        let codes: Vec<f4e2m1> = (0..8).map(|code| f4e2m1::from_bits(code as u8)).collect();
        let grouped = laid_out::<f4e2m1, m![O, I], Grouped>(&HostTensor::from_vec(codes.clone()));
        let found: Vec<u8> = grouped
            .unwrap()
            .into_vec()
            .iter()
            .map(|code| code.to_bits())
            .collect();
        let mut expected = vec![0; 8];
        for (o, i) in (0..2).flat_map(|o| (0..4).map(move |i| (o, i))) {
            expected[at(o, i)] = codes[o * 4 + i].to_bits();
        }
        assert_eq!(found, expected);
        let split = laid_out::<f4e2m1, m![O, I], m![I, O]>(&HostTensor::from_vec(codes));
        assert!(split.is_none(), "a run of one code splits a byte");
        let padded = laid_out::<f32, m![O, I], m![O, I # 8]>(&HostTensor::from_vec(values.clone()));
        let rows = values
            .chunks(4)
            .flat_map(|row| row.iter().copied().chain([0.0; 4]));
        assert_eq!(padded.unwrap().into_vec(), rows.collect::<Vec<_>>());
        let inner = laid_out::<f32, m![O, I], m![O # 4, I]>(&HostTensor::from_vec(values));
        assert!(
            inner.is_none(),
            "padding outside the innermost factor is the upload's"
        );
    }

    /// A stored dimension holding a run of the leaf's axes reads as the axes; one that splits an
    /// axis, or leaves one out, is refused.
    #[test]
    fn a_dimension_holds_a_run_of_the_leaf_axes() {
        axes![H = 2, F = 2];
        let data: Vec<u8> = (0..8).flat_map(|at| (at as f32).to_le_bytes()).collect();
        let flat = TensorView::new(Dtype::F32, vec![4, 2], &data).unwrap();
        let heads = Checkpoint::value::<f32, m![H, F, O], m![H, F, O]>(&flat, "w").unwrap();
        assert_eq!(heads.into_vec(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let split = Checkpoint::value::<f32, m![O, I], m![O, I]>(&flat, "w").err();
        assert!(
            matches!(split, Some(LoadError::InvalidData { .. })),
            "{split:?}"
        );
        let short = TensorView::new(Dtype::F32, vec![8], &data).unwrap();
        let short = Checkpoint::value::<f32, m![H, F, O, I], m![H, F, O, I]>(&short, "w").err();
        assert!(
            matches!(short, Some(LoadError::InvalidData { .. })),
            "{short:?}"
        );
    }
}
