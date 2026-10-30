//! The tilings a linear's contraction runs with: the listed tiling of a shape that has one, and
//! for any other the widest legal one; the contraction derives its geometry from them.

/// How a projection's input rows reach the slices that hold its contraction parts, and how the
/// parts' partial sums meet.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Routing {
    /// Every slice holds every row.
    Replicated,
    /// Rows stream to the slices transposed, a packet a slice.
    Transposed,
    /// Rows pass around the slices in a ring.
    Ring,
    /// Each part's slice holds that part of the rows.
    Chunk,
}

use Routing::*;

/// A slice's data memory is sixteen pages of 32 KiB.
const PAGE: usize = 32 * 1024;

/// Data memory a resident BF16 slab and its E4M3 copy (three bytes a coefficient) may take: a
/// policy, the widest measured tiling's eleven pages, leaving five to staged rows and results.
const RESIDENT: usize = 11 * PAGE;

/// Row lanes one contraction pass stages beside the slab, at most a vector packet's eight.
const LANES: usize = 8;

/// Inputs a derived part holds a multiple of where widths allow, so its E2M1 block scales, an E4M3
/// byte per 16 inputs, fill whole eight-byte data memory writes for every weight form.
const SCALED: usize = 16 * 8;

/// Rows of an HBM envelope one staging is written beside a resident weight slab: with the widest
/// slab of [`tiling`] its staged copy still fits a slice's data memory.
pub(crate) const BAND: usize = 64;

/// Data memory a slice gives a row tile's weight slab, in bytes.
const SLAB: usize = 301_056;

/// Bytes of a BF16 weight or row element, which a projection over rows stages.
const BF16: usize = 2;

/// Data memory a derived row tile's weight slab takes at most: a quarter of a slice's, so the
/// slab and its staged rows hold less than half of it and a second live layer fits beside them.
const DERIVED_SLAB: usize = 4 * PAGE;

/// A shape `input` by `output`, the tiling listed for it, and how long a BF16 projection of `rows`
/// rows took on RNGD with that tiling and with the derived one.
#[expect(
    dead_code,
    reason = "no rule reads the timing yet; it is a cost model's evidence"
)]
struct Measured {
    input: usize,
    output: usize,
    /// Input parts, each of `input / parts` inputs.
    parts: usize,
    /// Output blocks of a pass and cluster, each of `output / (2 · passes · blocks)` outputs.
    blocks: usize,
    /// Weight slabs loaded one after another.
    passes: usize,
    routing: Routing,
    rows: usize,
    /// Microseconds with the listed tiling; none where it fails to compile at these rows.
    listed: Option<u32>,
    /// Microseconds with the derived tiling; none where the listed one was not timed.
    derived: Option<u32>,
}

/// The listed tilings, each kept where it beats the derived one by over 1% or was not timed.
///
/// TODO: replace them by a cost-model rule (row replication bytes, slab loads, routing)
/// calibrated on their times, and delete the entries it matches within 1%.
const MEASURED: [Measured; 43] = [
    // Replicated
    Measured {
        input: 512,
        output: 10752,
        parts: 1,
        blocks: 64,
        passes: 1,
        routing: Replicated,
        rows: 8,
        listed: Some(101),
        derived: Some(105),
    },
    Measured {
        input: 1024,
        output: 6144,
        parts: 4,
        blocks: 32,
        passes: 1,
        routing: Replicated,
        rows: 8,
        listed: Some(79),
        derived: Some(161),
    },
    Measured {
        input: 1152,
        output: 1152,
        parts: 4,
        blocks: 16,
        passes: 1,
        routing: Replicated,
        rows: 8,
        listed: Some(64),
        derived: Some(140),
    },
    Measured {
        input: 1152,
        output: 3456,
        parts: 4,
        blocks: 16,
        passes: 1,
        routing: Replicated,
        rows: 8,
        listed: Some(96),
        derived: Some(154),
    },
    Measured {
        input: 1152,
        output: 4608,
        parts: 4,
        blocks: 16,
        passes: 1,
        routing: Replicated,
        rows: 8,
        listed: Some(78),
        derived: Some(164),
    },
    Measured {
        input: 2048,
        output: 12288,
        parts: 8,
        blocks: 32,
        passes: 1,
        routing: Replicated,
        rows: 8,
        listed: Some(158),
        derived: Some(285),
    },
    Measured {
        input: 2560,
        output: 19456,
        parts: 8,
        blocks: 32,
        passes: 1,
        routing: Replicated,
        rows: 8,
        listed: Some(279),
        derived: Some(319),
    },
    Measured {
        input: 2688,
        output: 10752,
        parts: 4,
        blocks: 64,
        passes: 1,
        routing: Replicated,
        rows: 8,
        listed: Some(277),
        derived: Some(351),
    },
    Measured {
        input: 2688,
        output: 24192,
        parts: 4,
        blocks: 16,
        passes: 7,
        routing: Replicated,
        rows: 8,
        listed: Some(437),
        derived: Some(615),
    },
    Measured {
        input: 3072,
        output: 1024,
        parts: 4,
        blocks: 16,
        passes: 1,
        routing: Replicated,
        rows: 8,
        listed: Some(49),
        derived: Some(160),
    },
    Measured {
        input: 4096,
        output: 24576,
        parts: 32,
        blocks: 8,
        passes: 4,
        routing: Replicated,
        rows: 8,
        listed: Some(410),
        derived: Some(438),
    },
    Measured {
        input: 4096,
        output: 32768,
        parts: 32,
        blocks: 4,
        passes: 8,
        routing: Replicated,
        rows: 8,
        listed: Some(585),
        derived: Some(627),
    },
    Measured {
        input: 4608,
        output: 1152,
        parts: 8,
        blocks: 16,
        passes: 1,
        routing: Replicated,
        rows: 8,
        listed: Some(100),
        derived: Some(157),
    },
    Measured {
        input: 5120,
        output: 1024,
        parts: 16,
        blocks: 16,
        passes: 1,
        routing: Replicated,
        rows: 8,
        listed: Some(77),
        derived: Some(146),
    },
    Measured {
        input: 5120,
        output: 4096,
        parts: 16,
        blocks: 16,
        passes: 1,
        routing: Replicated,
        rows: 8,
        listed: Some(173),
        derived: Some(207),
    },
    Measured {
        input: 5120,
        output: 10240,
        parts: 16,
        blocks: 16,
        passes: 1,
        routing: Replicated,
        rows: 8,
        listed: Some(279),
        derived: Some(299),
    },
    Measured {
        input: 5120,
        output: 51200,
        parts: 8,
        blocks: 32,
        passes: 10,
        routing: Replicated,
        rows: 8,
        listed: None,
        derived: None,
    },
    Measured {
        input: 5376,
        output: 64,
        parts: 8,
        blocks: 8,
        passes: 1,
        routing: Replicated,
        rows: 8,
        listed: Some(38),
        derived: Some(42),
    },
    Measured {
        input: 5376,
        output: 192,
        parts: 8,
        blocks: 8,
        passes: 1,
        routing: Replicated,
        rows: 8,
        listed: Some(45),
        derived: Some(106),
    },
    Measured {
        input: 5376,
        output: 2048,
        parts: 8,
        blocks: 32,
        passes: 1,
        routing: Replicated,
        rows: 8,
        listed: Some(158),
        derived: Some(277),
    },
    Measured {
        input: 5376,
        output: 4096,
        parts: 8,
        blocks: 32,
        passes: 1,
        routing: Replicated,
        rows: 8,
        listed: Some(228),
        derived: Some(309),
    },
    Measured {
        input: 5376,
        output: 8192,
        parts: 8,
        blocks: 32,
        passes: 2,
        routing: Replicated,
        rows: 8,
        listed: Some(318),
        derived: Some(365),
    },
    Measured {
        input: 6144,
        output: 2048,
        parts: 16,
        blocks: 16,
        passes: 1,
        routing: Replicated,
        rows: 8,
        listed: Some(98),
        derived: Some(188),
    },
    Measured {
        input: 8192,
        output: 5120,
        parts: 16,
        blocks: 16,
        passes: 1,
        routing: Replicated,
        rows: 8,
        listed: Some(207),
        derived: Some(319),
    },
    Measured {
        input: 8192,
        output: 5376,
        parts: 64,
        blocks: 4,
        passes: 3,
        routing: Replicated,
        rows: 8,
        listed: Some(199),
        derived: Some(274),
    },
    Measured {
        input: 9728,
        output: 2560,
        parts: 16,
        blocks: 16,
        passes: 1,
        routing: Replicated,
        rows: 8,
        listed: Some(229),
        derived: Some(330),
    },
    Measured {
        input: 10240,
        output: 2560,
        parts: 16,
        blocks: 16,
        passes: 1,
        routing: Replicated,
        rows: 8,
        listed: Some(167),
        derived: Some(217),
    },
    Measured {
        input: 10752,
        output: 5376,
        parts: 16,
        blocks: 16,
        passes: 1,
        routing: Replicated,
        rows: 8,
        listed: Some(385),
        derived: Some(397),
    },
    Measured {
        input: 12288,
        output: 4096,
        parts: 32,
        blocks: 8,
        passes: 1,
        routing: Replicated,
        rows: 8,
        listed: Some(279),
        derived: Some(357),
    },
    Measured {
        input: 16384,
        output: 4096,
        parts: 32,
        blocks: 8,
        passes: 2,
        routing: Replicated,
        rows: 8,
        listed: Some(393),
        derived: Some(654),
    },
    Measured {
        input: 16384,
        output: 5120,
        parts: 32,
        blocks: 8,
        passes: 2,
        routing: Replicated,
        rows: 8,
        listed: Some(376),
        derived: Some(497),
    },
    Measured {
        input: 16384,
        output: 5376,
        parts: 64,
        blocks: 4,
        passes: 3,
        routing: Replicated,
        rows: 8,
        listed: Some(349),
        derived: Some(414),
    },
    Measured {
        input: 25600,
        output: 5120,
        parts: 8,
        blocks: 16,
        passes: 5,
        routing: Replicated,
        rows: 8,
        listed: Some(598),
        derived: Some(772),
    },
    // Ring
    Measured {
        input: 1024,
        output: 4096,
        parts: 2,
        blocks: 64,
        passes: 1,
        routing: Ring,
        rows: 8,
        listed: Some(72),
        derived: Some(145),
    },
    Measured {
        input: 2048,
        output: 4096,
        parts: 8,
        blocks: 32,
        passes: 1,
        routing: Ring,
        rows: 8,
        listed: Some(88),
        derived: Some(241),
    },
    Measured {
        input: 2560,
        output: 6144,
        parts: 4,
        blocks: 48,
        passes: 1,
        routing: Ring,
        rows: 8,
        listed: Some(104),
        derived: Some(201),
    },
    // Chunk
    Measured {
        input: 2048,
        output: 1024,
        parts: 16,
        blocks: 16,
        passes: 1,
        routing: Chunk,
        rows: 8,
        listed: Some(32),
        derived: Some(116),
    },
    Measured {
        input: 2048,
        output: 2048,
        parts: 16,
        blocks: 16,
        passes: 1,
        routing: Chunk,
        rows: 8,
        listed: Some(52),
        derived: Some(204),
    },
    Measured {
        input: 4096,
        output: 1024,
        parts: 32,
        blocks: 4,
        passes: 1,
        routing: Chunk,
        rows: 1,
        listed: Some(31),
        derived: Some(1290),
    },
    Measured {
        input: 4096,
        output: 2560,
        parts: 32,
        blocks: 8,
        passes: 1,
        routing: Chunk,
        rows: 8,
        listed: Some(84),
        derived: Some(167),
    },
    Measured {
        input: 4096,
        output: 4096,
        parts: 32,
        blocks: 4,
        passes: 1,
        routing: Chunk,
        rows: 8,
        listed: Some(136),
        derived: Some(265),
    },
    Measured {
        input: 4096,
        output: 5120,
        parts: 32,
        blocks: 8,
        passes: 1,
        routing: Chunk,
        rows: 8,
        listed: Some(117),
        derived: Some(271),
    },
    // Transposed
    Measured {
        input: 4096,
        output: 6144,
        parts: 4,
        blocks: 48,
        passes: 1,
        routing: Transposed,
        rows: 8,
        listed: Some(150),
        derived: Some(278),
    },
];

/// A projection over rows in HBM and its row tile. That kernel holds a block of rows in the
/// register file, so its tile is K groups and kept outputs rather than a [`Measured`] tiling.
struct RowTile {
    input: usize,
    output: usize,
    /// Rows a cluster holds of a block, as [`half`] gives them.
    half: usize,
    /// K groups: the slices whose partials one inter-slice reduction sums.
    parts: usize,
    /// Outputs one slice holds after the reduction, for each of its rows.
    kept: usize,
    /// Weight passes.
    passes: usize,
}

/// The listed row tiles of served shapes; a shape they lack takes [`derived`]'s.
///
/// TODO: replace them by a cost-model rule calibrated on measurements; none records these values.
const ROW_TILES: [RowTile; 22] = [
    RowTile {
        input: 128,
        output: 2048,
        half: 128,
        parts: 16,
        kept: 128,
        passes: 1,
    },
    RowTile {
        input: 1792,
        output: 5376,
        half: 128,
        parts: 64,
        kept: 336,
        passes: 1,
    },
    RowTile {
        input: 2048,
        output: 2048,
        half: 128,
        parts: 16,
        kept: 128,
        passes: 1,
    },
    RowTile {
        input: 2048,
        output: 3072,
        half: 128,
        parts: 16,
        kept: 192,
        passes: 1,
    },
    RowTile {
        input: 2048,
        output: 6144,
        half: 128,
        parts: 16,
        kept: 384,
        passes: 1,
    },
    RowTile {
        input: 2048,
        output: 16384,
        half: 128,
        parts: 16,
        kept: 1024,
        passes: 1,
    },
    RowTile {
        input: 3584,
        output: 5376,
        half: 128,
        parts: 64,
        kept: 336,
        passes: 1,
    },
    RowTile {
        input: 5120,
        output: 5376,
        half: 128,
        parts: 64,
        kept: 336,
        passes: 1,
    },
    RowTile {
        input: 5376,
        output: 1792,
        half: 128,
        parts: 64,
        kept: 112,
        passes: 1,
    },
    RowTile {
        input: 5376,
        output: 5376,
        half: 128,
        parts: 64,
        kept: 336,
        passes: 1,
    },
    RowTile {
        input: 5376,
        output: 7168,
        half: 128,
        parts: 64,
        kept: 448,
        passes: 1,
    },
    RowTile {
        input: 5376,
        output: 21504,
        half: 128,
        parts: 64,
        kept: 448,
        passes: 3,
    },
    RowTile {
        input: 5376,
        output: 28672,
        half: 128,
        parts: 64,
        kept: 448,
        passes: 4,
    },
    RowTile {
        input: 7168,
        output: 5376,
        half: 128,
        parts: 128,
        kept: 336,
        passes: 1,
    },
    RowTile {
        input: 8192,
        output: 2048,
        half: 128,
        parts: 64,
        kept: 128,
        passes: 1,
    },
    RowTile {
        input: 14336,
        output: 5376,
        half: 128,
        parts: 128,
        kept: 168,
        passes: 2,
    },
    RowTile {
        input: 5120,
        output: 5376,
        half: 64,
        parts: 64,
        kept: 168,
        passes: 1,
    },
    RowTile {
        input: 5376,
        output: 7168,
        half: 64,
        parts: 64,
        kept: 224,
        passes: 1,
    },
    RowTile {
        input: 5376,
        output: 21504,
        half: 64,
        parts: 64,
        kept: 224,
        passes: 3,
    },
    RowTile {
        input: 5376,
        output: 28672,
        half: 64,
        parts: 64,
        kept: 224,
        passes: 4,
    },
    RowTile {
        input: 7168,
        output: 5376,
        half: 64,
        parts: 128,
        kept: 168,
        passes: 1,
    },
    RowTile {
        input: 14336,
        output: 5376,
        half: 64,
        parts: 128,
        kept: 56,
        passes: 3,
    },
];

/// The `(chunk, block, passes, routing)` of `(input, output)`: a listed shape's entry, else the
/// widest legal replicated tiling of fewest passes, preferring parts of whole E2M1 scale blocks.
pub(crate) const fn tiling(input: usize, output: usize) -> (usize, usize, usize, Routing) {
    let mut at = 0;
    while at < MEASURED.len() {
        let entry = &MEASURED[at];
        if entry.input == input && entry.output == output {
            return (
                input / entry.parts,
                output / (2 * entry.passes * entry.blocks),
                entry.passes,
                entry.routing,
            );
        }
        at += 1;
    }
    let mut unit = SCALED;
    loop {
        let mut passes = 1;
        while passes <= output / 8 {
            if let Some(choice) = widest(input, output, passes, unit) {
                return choice;
            }
            passes += 1;
        }
        assert!(
            unit != 1,
            "no contraction tiling fits these projection widths"
        );
        unit = 1;
    }
}

/// The legal replicated tiling of `passes` slabs, parts a multiple of `unit` inputs, with the most
/// active slices, then cheap row staging, then fewest parts; none if no tiling is legal.
const fn widest(
    input: usize,
    output: usize,
    passes: usize,
    unit: usize,
) -> Option<(usize, usize, usize, Routing)> {
    if !output.is_multiple_of(2 * passes) {
        return None;
    }
    let half = output / (2 * passes);
    let mut best = None;
    let mut rank = (0, false);
    let mut parts = 1;
    while parts <= super::SLICES {
        let mut blocks = 1;
        while parts * blocks <= super::SLICES {
            // A slab index shares its digit with the top, so several passes leave it one.
            let mut top = 1;
            while top <= if passes == 1 { super::SLICES - 1 } else { 1 } {
                if input.is_multiple_of(parts) && half.is_multiple_of(blocks * top) {
                    let (chunk, block) = (input / parts, half / (blocks * top));
                    let active = parts * blocks * top;
                    let cheap = cheap(chunk, block);
                    if (active > rank.0 || (active == rank.0 && cheap && !rank.1))
                        && chunk.is_multiple_of(unit)
                        && legal(input, output, (chunk, block, passes))
                    {
                        rank = (active, cheap);
                        best = Some((chunk, block, passes, Replicated));
                    }
                }
                top += 2;
            }
            blocks *= 2;
        }
        parts *= 2;
    }
    best
}

/// Whether a part's eight-lane fill ping-pongs in half the tensor registers and a band of staged
/// rows fits the data memory beside the slab.
const fn cheap(chunk: usize, block: usize) -> bool {
    2 * LANES * chunk * 2 <= super::TRF && 2 * BAND * chunk + 3 * block * chunk <= 16 * PAGE
}

/// Whether `(chunk, block, passes)` partitions `input` by `output` into the slice layout and fits a
/// slice's tensor registers and data memory.
pub(crate) const fn legal(
    input: usize,
    output: usize,
    (chunk, block, passes): (usize, usize, usize),
) -> bool {
    if chunk == 0
        || block == 0
        || !chunk.is_multiple_of(32)
        || !input.is_multiple_of(chunk)
        || !block.is_multiple_of(4)
        || !output.is_multiple_of(2 * passes * block)
    {
        return false;
    }
    let top = odd(output / (2 * passes * block));
    let slices = output / (2 * passes * block * top) * (input / chunk) * top.next_power_of_two();
    (top == 1 || passes == 1)
        && super::SLICES.is_multiple_of(slices)
        && LANES * chunk * 2 <= super::TRF
        && 3 * block * chunk <= RESIDENT
}

/// The odd factor of `value`: the blocks of a pass and cluster the top output digit carves.
pub(crate) const fn odd(value: usize) -> usize {
    value >> value.trailing_zeros()
}

/// Rows of `rows`, each `inputs` BF16 inputs, one contraction pass carries: all while at most 32
/// fit the tensor registers beside the weight, else one lane group of `lanes`.
pub(crate) const fn batch(rows: usize, inputs: usize, lanes: usize) -> usize {
    if rows <= 32 && rows * inputs * 2 <= super::TRF {
        rows
    } else {
        lanes
    }
}

/// Output rows of a block of `block` rows one restoration scales: a power of two dividing the
/// block, at most 64, so their FP32 scales take at most 2 KiB of the VRF.
pub(crate) const fn restored(block: usize) -> usize {
    let mut rows = 64;
    while !block.is_multiple_of(rows) {
        rows /= 2;
    }
    rows
}

/// Input parts a block-scaled row splits into across slices: the fewest, at least two, whose
/// eight rows' BF16 parts fill at most a slice's tensor registers.
pub(crate) const fn parts(input: usize) -> usize {
    let mut parts = 2;
    while 8 * 2 * (input / parts) > super::TRF {
        parts *= 2;
    }
    parts
}

/// Output rows a slice restores in one pass of a block-scaled projection: the most, up to twelve,
/// dividing its `owned` rows whose block scales over a `part`-input part fill at most the VRF.
pub(crate) const fn height(owned: usize, part: usize) -> usize {
    let mut rows = 12;
    while !owned.is_multiple_of(rows) || rows * part / 16 * 4 > super::VRF {
        rows -= 1;
    }
    rows
}

/// Rows a cluster holds of a row tile's block: 128 when the rows come in blocks of 256, else 64.
pub(crate) const fn half(rows: usize) -> usize {
    if rows.is_multiple_of(256) {
        128
    } else {
        assert!(rows.is_multiple_of(128), "rows come in blocks of 128");
        64
    }
}

/// The row tile `(parts, kept, passes)` of `(input, output)` at `half` rows a cluster: a listed
/// shape's entry, else [`derived`]'s; a tile that does not [`hold`] fails to compile.
pub(crate) const fn row_tile(input: usize, output: usize, half: usize) -> (usize, usize, usize) {
    let tile = 'listed: {
        let mut at = 0;
        while at < ROW_TILES.len() {
            let entry = &ROW_TILES[at];
            if entry.input == input && entry.output == output && entry.half == half {
                break 'listed (entry.parts, entry.kept, entry.passes);
            }
            at += 1;
        }
        derived(input, output, half)
    };
    assert!(
        hold(input, output, half, tile),
        "the row tile does not fit the projection"
    );
    tile
}

/// Whether the row tile `(parts, kept, passes)` splits `input` by `output` over the slices at
/// `half` rows a cluster, a K group's rows of a block in a TRF bank, a slice's outputs in its slab.
const fn hold(
    input: usize,
    output: usize,
    half: usize,
    (parts, kept, passes): (usize, usize, usize),
) -> bool {
    let chunk = input / parts;
    let outputs = parts / (half / 8) * kept;
    input.is_multiple_of(parts)
        && chunk.is_multiple_of(4)
        && half * chunk * BF16 <= super::TRF / 2
        && kept.is_multiple_of(8)
        && outputs * chunk * BF16 <= SLAB
        && super::SLICES / parts * outputs * passes == output
}

/// The row tile of the fewest weight passes, then the fewest K groups, whose slab fits
/// [`DERIVED_SLAB`]: a K group's rows of a block fill one of the two TRF banks, and each slice's
/// outputs are whole groups of eight kept outputs over the K groups the reduction spreads them on.
const fn derived(input: usize, output: usize, half: usize) -> (usize, usize, usize) {
    let mut passes = 1;
    while passes <= output {
        let mut parts = half / 8;
        while parts <= super::SLICES {
            let chunk = input / parts;
            let groups = parts / (half / 8);
            if input.is_multiple_of(parts)
                && chunk.is_multiple_of(4)
                && half * chunk * BF16 <= super::TRF / 2
                && output.is_multiple_of(super::SLICES / parts * passes)
            {
                let outputs = output / (super::SLICES / parts * passes);
                if outputs.is_multiple_of(8 * groups) && outputs * chunk * BF16 <= DERIVED_SLAB {
                    return (parts, outputs / groups, passes);
                }
            }
            parts *= 2;
        }
        passes += 1;
    }
    panic!("no row tile fits this projection")
}
