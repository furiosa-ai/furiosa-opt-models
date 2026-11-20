//! Prepared host weights and their reusable, bounded page-locked transfer images.
//!
//! The loader has already converted values and laid them out for the target tensor. A source
//! retains one replicated image or every chip's part in chip order; pinning changes neither.

use furiosa_opt_models::load::LoadError;
use furiosa_opt_std::prelude::*;
use std::{io, ops::Range, sync::Arc};

/// Maximum allocation size of a pinned piece, matching the repro's transfer images.
const PIECE: usize = 1 << 30;

/// A prepared tensor's bytes, retained on the host across placements and evictions.
#[derive(Clone)]
pub(super) struct Source {
    bytes: Arc<HostBuffer>,
    len: usize,
    shards: Option<usize>,
}

impl Source {
    /// Retain a host result already laid out as one target chip's elements or all chips' parts.
    /// Equal byte lengths alone do not establish compatible layouts.
    pub(super) fn from_host<D: MaterializableScalar, C: M, S: M, E: M>(
        host: HostTensor<D, E>,
        name: &str,
    ) -> Result<Self, LoadError> {
        let unsupported = |reason: &str| LoadError::Unsupported {
            name: name.into(),
            reason: reason.into(),
        };
        if C::SIZE == 0 {
            return Err(unsupported("a resident tensor requires at least one chip"));
        }
        let layout = E::to_value().normalize();
        let shards = if layout == S::to_value().normalize() {
            None
        } else if layout == <m![{ C }, { S }]>::to_value().normalize() {
            Some(C::SIZE)
        } else {
            return Err(unsupported(
                "resident host weights must be prepared in element order or chip-major order",
            ));
        };
        let total = HbmTensor::<D, C, S>::size();
        let len = total / C::SIZE;
        let mut bytes = host.into_buffer();
        if len == 0 || !total.is_multiple_of(C::SIZE) || bytes.len() != len * shards.unwrap_or(1) {
            return Err(unsupported(
                "resident host weights must hold a nonempty whole byte image per chip",
            ));
        }
        // Converted host trees may spell a replicated weight as identical chip-major images.
        // Retaining all of those copies multiplies a large pipeline's host memory by its chip
        // count. Compare actual bytes, so real tensor-parallel shards retain their own images.
        // Keep explicitly pinned input storage as supplied, including its placement policy.
        let shards = if shards.is_some_and(|count| count > 1)
            && !bytes.is_pinned()
            && bytes
                .chunks_exact(len)
                .skip(1)
                .all(|part| part == &bytes[..len])
        {
            bytes = HostBuffer::from_vec(bytes[..len].to_vec());
            None
        } else {
            shards
        };
        Ok(Self {
            bytes: Arc::new(bytes),
            len,
            shards,
        })
    }

    /// Bytes of one chip's copy.
    pub(super) fn len(&self) -> usize {
        self.len
    }

    /// Page-lock every chip's prepared bytes once, using the calling thread's NUMA policy.
    /// Retaining the source is independent of retaining this image.
    pub(super) fn pin(&self) -> io::Result<Image> {
        let mut pieces = Vec::new();
        for shard in 0..self.shards.unwrap_or(1) {
            for at in (0..self.len).step_by(PIECE) {
                let len = PIECE.min(self.len - at);
                let mut bytes = HostBuffer::zero(len);
                if !bytes.is_pinned() {
                    return Err(io::Error::new(
                        io::ErrorKind::OutOfMemory,
                        format!("the system cannot page-lock another {len} bytes"),
                    ));
                }
                let begin = shard * self.len + at;
                bytes.copy_from_slice(&self.bytes[begin..begin + len]);
                pieces.push(Pinned {
                    bytes,
                    chip: self.shards.map(|_| shard),
                    at,
                });
            }
        }
        Ok(Image {
            pieces,
            len: self.len,
            shards: self.shards,
        })
    }

    /// The pageable source in each destination chip's byte order.
    pub(super) fn pieces(&self, chips: usize) -> io::Result<Vec<Piece<'_>>> {
        check_chips(self.shards, chips)?;
        Ok(match self.shards {
            Some(_) => vec![Piece {
                host: HostBufferSlice::from(&*self.bytes),
                chips: 0..chips,
                bytes: 0..self.len,
            }],
            None => (0..chips)
                .map(|chip| Piece {
                    host: HostBufferSlice::from(&*self.bytes),
                    chips: chip..chip + 1,
                    bytes: 0..self.len,
                })
                .collect(),
        })
    }
}

/// Page-locked pieces of a prepared source, reusable across requests.
pub(super) struct Image {
    pieces: Vec<Pinned>,
    len: usize,
    shards: Option<usize>,
}

struct Pinned {
    bytes: HostBuffer,
    chip: Option<usize>,
    at: usize,
}

/// Host bytes and the byte interval they fill on their destination chips.
pub(super) struct Piece<'a> {
    pub(super) host: HostBufferSlice<'a>,
    pub(super) chips: Range<usize>,
    pub(super) bytes: Range<usize>,
}

impl Image {
    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn pieces(&self, chips: usize) -> io::Result<Vec<Piece<'_>>> {
        check_chips(self.shards, chips)?;
        Ok(self
            .pieces
            .iter()
            .flat_map(|piece| {
                let ranks = piece.chip.map_or(0..chips, |chip| chip..chip + 1);
                ranks.map(move |chip| Piece {
                    host: HostBufferSlice::from(&piece.bytes),
                    chips: chip..chip + 1,
                    bytes: piece.at..piece.at + piece.bytes.len(),
                })
            })
            .collect())
    }
}

fn check_chips(shards: Option<usize>, chips: usize) -> io::Result<()> {
    if chips == 0 || shards.is_some_and(|expected| expected != chips) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "prepared weight shards do not match the destination chip count",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use furiosa_opt_models::tensor::parallel::Shard;

    axes![Out = 4, In = 4];
    type Chips = m![Out / 2];
    type Shape = m![Out % 2, In];

    #[test]
    fn retains_prepared_parts_and_refuses_equal_sized_permutations() {
        let values: Vec<f32> = (0..16).map(|at| at as f32).collect();
        let source = Source::from_host::<f32, Chips, Shape, m![Out, In]>(
            HostTensor::from_vec(values.clone()),
            "weight",
        )
        .unwrap();
        assert_eq!(source.len(), 8 * 4);
        assert_eq!(source.shards, Some(2));
        assert_eq!(
            &**source.bytes,
            values
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>()
        );
        assert!(source.pieces(1).is_err());
        let pieces = source.pieces(2).unwrap();
        assert_eq!(pieces.len(), 1);
        assert_eq!(pieces[0].chips, 0..2);
        assert_eq!(pieces[0].bytes, 0..32);

        let error = Source::from_host::<f32, Chips, Shape, m![In, Out]>(
            HostTensor::from_vec(values),
            "weight",
        )
        .err()
        .unwrap();
        assert!(matches!(error, LoadError::Unsupported { name, .. } if name == "weight"));
    }

    #[test]
    fn retains_one_source_for_replicated_weights() {
        let single = Source::from_host::<f32, m![{ Symbol<Shard<4>> }], m![In], m![In]>(
            HostTensor::from_vec([1., 2., 3., 4.]),
            "norm",
        )
        .unwrap();
        let expanded = Source::from_host::<
            f32,
            m![{ Symbol<Shard<4>> }],
            m![In],
            m![{ Symbol<Shard<4>> }, In],
        >(
            HostTensor::from_vec([1., 2., 3., 4.].repeat(4)),
            "expanded_norm",
        )
        .unwrap();
        for source in [single, expanded] {
            let shared = source.clone();
            assert!(Arc::ptr_eq(&source.bytes, &shared.bytes));
            assert_eq!(source.len(), 16);
            assert_eq!(
                source.bytes.len(),
                16,
                "identical chip images retain one host copy"
            );
            assert!(source.shards.is_none());
            let pieces = source.pieces(4).unwrap();
            assert_eq!(pieces.len(), 4);
            for (chip, piece) in pieces.iter().enumerate() {
                assert_eq!(piece.chips, chip..chip + 1);
                assert_eq!(piece.bytes, 0..16);
            }
            assert!(source.pieces(0).is_err());
        }
    }

    /// The image owns pinned copies in chip order, independent of the pageable source.
    #[test]
    #[ignore = "requires page-locked host memory from /dev/dma_heap/system"]
    fn pins_prepared_bytes_in_chip_order() {
        let source = Source::from_host::<f32, Chips, Shape, m![Out, In]>(
            HostTensor::from_vec((0..16).map(|at| at as f32)),
            "weight",
        )
        .unwrap();
        let image = source.pin().unwrap();
        drop(source);
        assert_eq!(image.len(), 32);
        assert_eq!(image.pieces.len(), 2);
        for (chip, piece) in image.pieces.iter().enumerate() {
            assert!(piece.bytes.is_pinned());
            assert_eq!(piece.chip, Some(chip));
            assert_eq!(piece.at, 0);
            let expected: Vec<u8> = (chip * 8..chip * 8 + 8)
                .flat_map(|at| (at as f32).to_le_bytes())
                .collect();
            assert_eq!(&piece.bytes[..], expected);
        }
        assert_eq!(image.pieces(2).unwrap().len(), 2);
        assert!(image.pieces(4).is_err());
    }

    /// A submitted image may span several runtime queue slots. Dropping its caller-owned
    /// pieces must not cancel the later slots or release their pinned memory prematurely.
    #[test]
    #[ignore = "requires four explicitly selected RNGD chips and page-locked host memory"]
    fn uploads_every_piece_when_an_image_exceeds_one_runtime_slot() {
        let mut device = Device::new(Topology { chips: 4, pes: 8 }).unwrap();
        let parts = 70;
        let len = parts * 8;
        let expected: Vec<u8> = (0..4 * parts * 2)
            .flat_map(|at| ((at * 17 % 251) as u32).to_le_bytes())
            .collect();
        let mut pieces = Vec::new();
        for chip in 0..4 {
            for part in 0..parts {
                let mut bytes = HostBuffer::zero(8);
                assert!(bytes.is_pinned());
                let begin = chip * len + part * 8;
                bytes.copy_from_slice(&expected[begin..begin + 8]);
                pieces.push(Pinned {
                    bytes,
                    chip: Some(chip),
                    at: part * 8,
                });
            }
        }
        let image = Image {
            pieces,
            len,
            shards: Some(4),
        };
        assert_eq!(image.pieces(4).unwrap().len(), 280);
        let (buffer, completion) = super::super::upload::issue(&mut device, &image).unwrap();
        drop(image);
        let pending = completion.is_some();
        if let Some(completion) = completion {
            completion.wait().unwrap();
        }
        let bytes = futures_executor::block_on(
            buffer
                .slice(.., ..)
                .unwrap()
                .to_host(&mut device.pdma)
                .submit(),
        )
        .unwrap();
        assert_eq!(&bytes[..], expected);
        println!("280-piece pinned upload: async completion returned: {pending}");
    }
}
