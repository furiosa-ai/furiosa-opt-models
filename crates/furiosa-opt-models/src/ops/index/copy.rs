//! Device bytes copied into device bytes, such as token IDs or the rows of BF16 values, in a few
//! launches of fixed-length pieces rather than one copy a word.

use crate::Module;
use crate::specialize::index::copy::LENGTHS;
use crate::tensor::{memory::alias, parallel::Chip};
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::launch;
use std::io;

/// Pieces of each length one launch copies.
const PIECES: usize = 64;

/// Pieces of each length but the single word a [`place`] of few pieces copies: a launch guards
/// every slot it holds, and a batch's decode rows are up to [`PIECES`] single words.
const FEW: usize = 8;

axes![Words1024 = 1024, Words64 = 64, Words16 = 16, Words4 = 4];

// `axes!` takes literal sizes, so the piece axes repeat the lengths.
const _: () = assert!(
    Words1024::SIZE == LENGTHS[0]
        && Words64::SIZE == LENGTHS[1]
        && Words16::SIZE == LENGTHS[2]
        && Words4::SIZE == LENGTHS[3]
        && LENGTHS[4] == 1
);

/// Consecutive words, as many as `L` maps.
type Piece<L> = HbmTensor<i32, Chip, L>;

/// One length's pieces.
type Pieces<L> = Slice<Piece<L>>;

/// The pieces one launch copies from, by length, longest first: piece `i` of a length in the
/// source is copied into piece `i` of that length in the destination.
type Sources = (
    Pieces<m![Words1024]>,
    Pieces<m![Words64]>,
    Pieces<m![Words16]>,
    Pieces<m![Words4]>,
    Pieces<m![1]>,
);

/// The destinations of one launch's [`Sources`]: the single words' are [`PIECES`] views, those
/// past the launch's single words naming spare words that no copy writes.
type Targets = (
    Pieces<m![Words1024]>,
    Pieces<m![Words64]>,
    Pieces<m![Words16]>,
    Pieces<m![Words4]>,
    [Piece<m![1]>; PIECES],
);

/// Copies `from[i]` into `to[i]` for each of `from`, at most `N` of them.
#[derive(DeviceSend)]
struct Copies<const N: usize>;

impl<'a, L: M, const N: usize, Env> Module<(&'a Pieces<L>, &'a mut Pieces<L>), Env, &mut ()>
    for Copies<N>
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        _: Env,
        _: &mut (),
        (from, to): (&'a Pieces<L>, &'a mut Pieces<L>),
    ) {
        // A read of the length waits until every copy issued before it ends, so it is read once.
        let len = from.len();
        #[unroll]
        for piece in 0..N {
            if piece < len {
                from[piece]
                    .view()
                    .to_hbm_view(&mut context.tdma, to[piece].view_mut());
            }
        }
    }
}

/// The single words, each into a destination of its own.
impl<'a, Env> Module<(&'a Pieces<m![1]>, &'a mut [Piece<m![1]>; PIECES]), Env, &mut ()>
    for Copies<PIECES>
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        _: Env,
        _: &mut (),
        (from, to): (&'a Pieces<m![1]>, &'a mut [Piece<m![1]>; PIECES]),
    ) {
        let len = from.len();
        #[unroll]
        for piece in 0..PIECES {
            if piece < len {
                from[piece]
                    .view()
                    .to_hbm_view(&mut context.tdma, to[piece].view_mut());
            }
        }
    }
}

/// Copies HBM slices into HBM slices in pair order, submitting without waiting; a later read of a
/// destination waits for it. A length mismatch in any pair, or a length that is not whole words,
/// refuses the batch before anything moves.
// TODO: once std offers a host-submitted HBM-to-HBM copy, this body is that submit; drop `place`.
pub fn copy<'a>(
    device: &mut Device,
    pairs: impl IntoIterator<Item = (&'a HbmBuffer, &'a mut HbmBuffer)>,
) -> io::Result<()> {
    /// Launch `launch`'s share of one length's `(source, destination)` pieces.
    fn views<L: M>(
        pieces: &[(HbmBuffer, HbmBuffer)],
        launch: usize,
    ) -> io::Result<(Pieces<L>, Pieces<L>)> {
        let share = || pieces.iter().skip(launch * PIECES).take(PIECES);
        Ok((
            share()
                .map(|(from, _)| alias(from, ..))
                .collect::<io::Result<_>>()?,
            share()
                .map(|(_, to)| alias(to, ..))
                .collect::<io::Result<_>>()?,
        ))
    }
    let word = size_of::<i32>();
    let invalid = |message| io::Error::new(io::ErrorKind::InvalidInput, message);
    let mut classes: [Vec<(HbmBuffer, HbmBuffer)>; LENGTHS.len()] = Default::default();
    for (from, to) in pairs {
        let len = from.size();
        if len != to.size() {
            return Err(invalid("a copy's source and destination differ in length"));
        }
        if !len.is_multiple_of(word) {
            return Err(invalid("a copy's length is not whole words"));
        }
        // Pieces of each length, longest first, each the longest the rest of the pair fills.
        let mut at = 0;
        for (class, length) in LENGTHS.into_iter().enumerate() {
            let bytes = length * word;
            for _ in 0..(len - at) / bytes {
                let piece =
                    |buffer: &HbmBuffer| buffer.alias(at..at + bytes).map_err(io::Error::other);
                classes[class].push((piece(from)?, piece(to)?));
                at += bytes;
            }
        }
    }
    let count = classes
        .iter()
        .map(Vec::len)
        .max()
        .unwrap_or(0)
        .div_ceil(PIECES);
    let spare = HbmBuffer::alloc(device, PIECES * word).map_err(io::Error::other)?;
    // Single words copy into arguments of their own to issue back to back; `spare` pads the rest.
    let words = |pieces: &[(HbmBuffer, HbmBuffer)],
                 launch: usize|
     -> io::Result<(Pieces<m![1]>, [Piece<m![1]>; PIECES])> {
        let share = || pieces.iter().skip(launch * PIECES).take(PIECES);
        let free = (share().len()..PIECES).map(|at| alias(&spare, at * word..(at + 1) * word));
        let to: Vec<Piece<m![1]>> = share()
            .map(|(_, to)| alias(to, ..))
            .chain(free)
            .collect::<io::Result<_>>()?;
        Ok((
            share()
                .map(|(from, _)| alias(from, ..))
                .collect::<io::Result<_>>()?,
            to.try_into()
                .unwrap_or_else(|_| unreachable!("a launch binds every single-word destination")),
        ))
    };
    let launches = (0..count)
        .map(|launch| {
            let ((a, f), (b, g), (c, h), (d, i), (e, j)) = (
                views(&classes[0], launch)?,
                views(&classes[1], launch)?,
                views(&classes[2], launch)?,
                views(&classes[3], launch)?,
                words(&classes[4], launch)?,
            );
            Ok(((a, b, c, d, e), (f, g, h, i, j)))
        })
        .collect::<io::Result<Vec<(Sources, Targets)>>>()?;
    for (from, mut to) in launches {
        let few = [from.0.len(), from.1.len(), from.2.len(), from.3.len()]
            .into_iter()
            .all(|len| len <= FEW);
        match few {
            true => launch(place, (&mut *device, &Copies::<FEW>, &from, &mut to)).submit(),
            false => launch(place, (&mut *device, &Copies::<PIECES>, &from, &mut to)).submit(),
        }
        .accepted()
        .map(drop)
        .map_err(io::Error::other)?;
    }
    Ok(())
}

/// Copies every piece of `from` into its piece of `to`, at most `N` of each length but the single
/// word.
#[device(chip = 1)]
fn place<const N: usize>(
    context: &mut Device,
    copies: &Copies<N>,
    from: &Sources,
    to: &mut Targets,
) {
    copies.forward(&mut *context, (), &mut (), (&from.0, &mut to.0));
    copies.forward(&mut *context, (), &mut (), (&from.1, &mut to.1));
    copies.forward(&mut *context, (), &mut (), (&from.2, &mut to.2));
    copies.forward(&mut *context, (), &mut (), (&from.3, &mut to.3));
    Copies::<PIECES>.forward(context, (), &mut (), (&from.4, &mut to.4));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::DEVICE;

    /// `len` words of `buffer` from word `at`, another owner of them.
    fn words(buffer: &HbmBuffer, at: usize, len: usize) -> HbmBuffer {
        let word = size_of::<i32>();
        buffer.alias(at * word..(at + len) * word).unwrap()
    }

    /// Copied runs of words land at their rows, rows outside them keep their values: a run of
    /// pieces of every length, more than `FEW` of one, and a short run.
    #[tokio::test(flavor = "current_thread")]
    async fn places_runs_at_their_rows() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        axes![Source = 3072, Target = 4096];
        let tokens: Vec<i32> = (0..3072).map(|at| 9_000 + 7 * at).collect();
        let source: HbmTensor<i32, Chip, m![Source]> =
            HostTensor::<i32, m![Source]>::from_vec(tokens.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let source = source.into_buffer();
        let target: HbmTensor<i32, Chip, m![Target]> =
            HostTensor::<i32, m![Target]>::from_vec(vec![-1; 4096])
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let target = target.into_buffer();
        // Words 3.. at row 10, 2 * 1024 + 9 * 64 + 3 * 16 + 3 * 4 + 3 of them; words 3000.. at
        // row 3500.
        let sources = [words(&source, 3, 2687), words(&source, 3000, 5)];
        let mut targets = [words(&target, 10, 2687), words(&target, 3500, 5)];
        copy(&mut device, sources.iter().zip(&mut targets)).unwrap();
        let target: HbmTensor<i32, Chip, m![Target]> = alias(&target, ..).unwrap();
        let rows = target
            .to_host::<m![Target]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let expected: Vec<i32> = (0..4096)
            .map(|row| match row {
                10..2697 => tokens[row - 7],
                3500..3505 => tokens[row - 500],
                _ => -1,
            })
            .collect();
        assert_eq!(rows, expected);
    }

    /// Single words read from buffers of their own, more than one launch carries, land at their
    /// rows through the copy a batch's decode rows take; rows past them keep their values.
    #[tokio::test(flavor = "current_thread")]
    async fn places_words_of_their_own_buffers() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        axes![Target = 1024];
        let count = PIECES + 6;
        let mut sources = Vec::new();
        for row in 0..count {
            let word: HbmTensor<i32, Chip, m![1]> =
                HostTensor::<i32, m![1]>::from_vec(vec![5_000 + row as i32])
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap();
            sources.push(word.into_buffer());
        }
        let target: HbmTensor<i32, Chip, m![Target]> =
            HostTensor::<i32, m![Target]>::from_vec(vec![-1; 1024])
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let target = target.into_buffer();
        let mut targets: Vec<HbmBuffer> = (0..count).map(|row| words(&target, row, 1)).collect();
        copy(&mut device, sources.iter().zip(&mut targets)).unwrap();
        let target: HbmTensor<i32, Chip, m![Target]> = alias(&target, ..).unwrap();
        let rows = target
            .to_host::<m![Target]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let expected: Vec<i32> = (0..1024)
            .map(|row| match row < count {
                true => 5_000 + row as i32,
                false => -1,
            })
            .collect();
        assert_eq!(rows, expected);
    }
}
