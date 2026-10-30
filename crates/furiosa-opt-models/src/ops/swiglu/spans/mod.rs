//! The span MLP's stages over a span's normalized tiles, run by the weight forms, block-scaled
//! E4M3 ([`fp8`]) or BF16 linears ([`linear`]).

use furiosa_opt_std::prelude::*;

use crate::ops::linear::Groups;
use crate::tensor::parallel::Chip;
use product::Halves;

pub mod fp8;
mod gate;
mod linear;
mod product;

axes![Term = 2];
/// A chip's two clusters, each projecting half of the intermediate features.
type Cluster<Intermediate> = m![Intermediate / (Intermediate / 2)];
/// A slice's output features: its cluster's half over the cluster's 32 output blocks.
type Out<Intermediate> = m![Intermediate % (Intermediate / 64)];
/// A cluster's output blocks.
type Blocks<Intermediate> = m![Intermediate % (Intermediate / 2) / (Intermediate / 64)];
type K<Hidden> = m![Hidden % 128];
type Partition<Hidden, Intermediate> = m![{ Blocks<Intermediate> }, Hidden % 1024 / 128];
type Slice<Intermediate, Rows> = m![{ Blocks<Intermediate> }, Rows / (Rows / 8)];
type Projection<Intermediate, Rows> = DmTensor<
    bf16,
    Chip,
    Cluster<Intermediate>,
    Slice<Intermediate, Rows>,
    m![Rows % (Rows / 8), { Out<Intermediate> }],
>;

#[cfg(test)]
mod tests {
    use super::fp8::{self, QuantizedMlp};
    use super::*;
    use crate::Module;
    use crate::modules::generic::schedule::{Fill, Spill};
    use crate::ops::attention::geometry::Hidden;
    use crate::ops::linear::Linear;
    use crate::ops::normalization::tests::random;
    use crate::ops::normalization::{
        Blocked,
        span::{Owned, Span},
    };
    use crate::ops::swiglu::{Down, GateUp, Intermediate, Mlp};
    use crate::tensor::form::Weight;
    use crate::tensor::live::{Live, tests::count};
    use crate::tensor::scalar::Value;
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;

    type Rows<H> = HbmTensor<bf16, Chip, Blocked<H, Span>>;
    type Tile<H> = Live<Owned<H, Span>, Span>;
    type Linears<H, I> = Mlp<H, I, Linear<H, GateUp<I>, Chip>, Linear<I, H, Chip>>;

    const SEEDS: (u64, u64, u64) = (
        0x9e37_79b9_7f4a_7c15,
        0x0123_4567_89ab_cdef,
        0xd1b5_4a32_d192_ed03,
    );

    /// The span's rows through the MLP's stages one after another.
    #[device(chip = 1)]
    fn sequential<H: AxisName, I: AxisName, F: DeviceSend, D: DeviceSend>(
        device: &mut Device,
        mlp: &Mlp<H, I, F, D>,
        rows: &Value,
        input: &Rows<H>,
    ) -> Rows<H>
    where
        Down<I>: AxisName,
        Mlp<H, I, F, D>: for<'x, 's> Module<&'x Tile<H>, (), &'s mut (), Output = Tile<H>>,
    {
        let live = Live::<(), Span>::read(rows);
        let tile = live.with(input.fill(&mut *device, live));
        let output = mlp.forward(&mut *device, (), &mut (), &tile);
        let mut stored = Rows::<H>::new();
        stored.spill(&mut *device, output.value(), live);
        stored
    }

    /// Logical weights as the MLP computes with them: gate and up `[I, H]`, down `[H, I]`, each
    /// coefficient its stored value times its channel scale.
    struct Weights {
        hidden: usize,
        width: usize,
        gate: Vec<f32>,
        up: Vec<f32>,
        down: Vec<f32>,
    }

    impl Weights {
        /// Pseudo-random E4M3-representable coefficients of `hidden` inputs and `width`
        /// intermediate features for seeds `(gate, up, down)`.
        fn new(
            (hidden, width): (usize, usize),
            (gate, up, down): (u64, u64, u64),
        ) -> (Self, [Vec<f8e4m3>; 3]) {
            let coefficients = |count, seed| -> Vec<f8e4m3> {
                random(count, seed)
                    .iter()
                    .map(|v| f8e4m3::from_f32(v.to_f32()))
                    .collect()
            };
            let stored = [
                coefficients(width * hidden, gate),
                coefficients(width * hidden, up),
                coefficients(hidden * width, down),
            ];
            let scaled = |values: &[f8e4m3], scale: f32| {
                values
                    .iter()
                    .map(|v| v.to_f32() * scale)
                    .collect::<Vec<f32>>()
            };
            let weights = Self {
                hidden,
                width,
                gate: scaled(&stored[0], Self::GATE),
                up: scaled(&stored[1], Self::GATE),
                down: scaled(&stored[2], Self::DOWN),
            };
            (weights, stored)
        }

        const GATE: f32 = 1.0 / 32.0;
        const DOWN: f32 = 1.0 / 64.0;

        /// `down(bf16(bf16(silu(gate x)) * up x))` of one row, gate and up rounded to BF16.
        fn reference(&self, row: &[f32]) -> Vec<f64> {
            let rounded = |x: f64| f64::from(bf16::from_f32(x as f32).to_f32());
            let dot = |a: &[f32], b: &[f32]| -> f64 {
                a.iter()
                    .zip(b)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum()
            };
            let (hidden, width) = (self.hidden, self.width);
            let product: Vec<f32> = (0..width)
                .map(|j| {
                    let gate = rounded(dot(row, &self.gate[j * hidden..][..hidden]));
                    let up = rounded(dot(row, &self.up[j * hidden..][..hidden]));
                    rounded(rounded(gate / (1.0 + (-gate).exp())) * up) as f32
                })
                .collect();
            (0..hidden)
                .map(|o| dot(&product, &self.down[o * width..][..width]))
                .collect()
        }
    }

    /// The E4M3 MLP of `stored` coefficients, every channel scale the same.
    async fn quantized<H: AxisName, I: AxisName>(
        device: &mut Device,
        stored: &[Vec<f8e4m3>; 3],
    ) -> QuantizedMlp<H, I>
    where
        Down<I>: AxisName,
    {
        let packed = [stored[0].clone(), stored[1].clone()].concat();
        let gate_up = HostTensor::<f8e4m3, m![Term, I, H]>::from_vec(packed)
            .to_hbm::<Chip, fp8::Blocked<H, I>>(&mut device.pdma)
            .await
            .unwrap();
        let channels = HostTensor::<f32, m![Term, I]>::from_vec(vec![Weights::GATE; 2 * I::SIZE])
            .to_hbm::<Chip, m![Term, I]>(&mut device.pdma)
            .await
            .unwrap();
        let down = HostTensor::<f8e4m3, m![H, I]>::from_vec(stored[2].clone())
            .to_hbm::<Chip, fp8::Stored<H, I>>(&mut device.pdma)
            .await
            .unwrap();
        let scales = HostTensor::<f32, m![H]>::from_vec(vec![Weights::DOWN; H::SIZE])
            .to_hbm::<Chip, m![H]>(&mut device.pdma)
            .await
            .unwrap();
        Mlp::new((gate_up, channels), (down, scales))
    }

    /// The BF16 MLP of the same logical weights.
    async fn linears<H: AxisName, I: AxisName>(
        device: &mut Device,
        weights: &Weights,
    ) -> Linears<H, I>
    where
        Down<I>: AxisName,
    {
        let bf16s = |values: &[f32]| {
            values
                .iter()
                .map(|v| bf16::from_f32(*v))
                .collect::<Vec<_>>()
        };
        let packed = [bf16s(&weights.gate), bf16s(&weights.up)].concat();
        let gate_up = HostTensor::<bf16, m![{ Symbol<GateUp<I>> }, H]>::from_vec(packed)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let down = HostTensor::<bf16, m![H, I]>::from_vec(bf16s(&weights.down))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        Mlp::new(
            Linear::new(Weight::new(gate_up)),
            Linear::new(Weight::new(down)),
        )
    }

    /// One MLP over a span at each live row count: the live rows match the reference within
    /// `tolerance` of each row's RMS, an all-zero row stays zero at the floor of its row scales,
    /// and changing the gate, up or down weights alone changes the rows.
    async fn check<H: AxisName, I: AxisName, F: DeviceSend, D: DeviceSend>(
        device: &mut Device,
        mlp: &Mlp<H, I, F, D>,
        weights: &Weights,
        changed: [&Mlp<H, I, F, D>; 3],
        tolerance: f64,
    ) where
        Down<I>: AxisName,
        Mlp<H, I, F, D>: for<'x, 's> Module<&'x Tile<H>, (), &'s mut (), Output = Tile<H>>,
    {
        let hidden = H::SIZE;
        let mut values = random(Span::SIZE * hidden, 0x2545_f491_4f6c_dd1d);
        let zero = 100;
        values[zero * hidden..][..hidden].fill(bf16::from_f32(0.0));
        let rows: Vec<f32> = values.iter().map(|v| v.to_f32()).collect();
        let input = HostTensor::<bf16, m![Span, H]>::from_vec(values)
            .to_hbm::<Chip, Blocked<H, Span>>(&mut device.pdma)
            .await
            .unwrap();
        let run = async |device: &mut Device, mlp: &Mlp<H, I, F, D>, live: usize| {
            let count = count(live).to_hbm(&mut device.pdma).await.unwrap();
            launch(sequential, (&mut *device, mlp, &count, &input))
                .await
                .unwrap()
                .to_host::<m![Span, H]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec()
                .iter()
                .map(|v| f64::from(v.to_f32()))
                .collect::<Vec<f64>>()
        };
        let error = |got: &[f64], expected: &[f64]| {
            let (difference, norm) = got.iter().zip(expected).fold((0.0, 0.0), |(d, n), (g, e)| {
                (d + (g - e) * (g - e), n + e * e)
            });
            (difference / norm).sqrt()
        };
        for live in [1, 200, 300, 512] {
            let output = run(&mut *device, mlp, live).await;
            let zeros = &output[zero * hidden..][..hidden];
            assert!(
                live <= zero || zeros.iter().all(|&v| v == 0.0),
                "{live} live rows, zero row"
            );
            for row in [0, live / 2, live - 1]
                .into_iter()
                .filter(|&row| row != zero)
            {
                let expected = weights.reference(&rows[row * hidden..][..hidden]);
                let got = &output[row * hidden..][..hidden];
                let error = error(got, &expected);
                assert!(
                    error <= tolerance,
                    "{live} live rows, row {row}: relative error {error}"
                );
            }
        }
        let base = run(&mut *device, mlp, 300).await;
        for (part, changed) in ["gate", "up", "down"].into_iter().zip(changed) {
            let other = run(&mut *device, changed, 300).await;
            let at = 299 * hidden..300 * hidden;
            assert!(
                error(&other[at.clone()], &base[at]) > 0.1,
                "changed {part} weights"
            );
        }
    }

    /// The logical weights of `H` inputs and `I` features, and the three that each change one of
    /// the gate, up or down weights.
    fn weights<H: AxisName, I: AxisName>() -> [(Weights, [Vec<f8e4m3>; 3]); 4]
    where
        Down<I>: AxisName,
    {
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)].map(|(g, u, d)| {
            Weights::new((H::SIZE, I::SIZE), (SEEDS.0 + g, SEEDS.1 + u, SEEDS.2 + d))
        })
    }

    /// The BF16 MLP of `H` inputs and `I` features within BF16 rounding of the reference.
    async fn linears_match<H: AxisName, I: AxisName>(device: &mut Device)
    where
        Down<I>: AxisName,
    {
        let [base, changes @ ..] = weights::<H, I>();
        let mlp = linears::<H, I>(&mut *device, &base.0).await;
        let mut changed = Vec::new();
        for (weights, _) in &changes {
            changed.push(linears::<H, I>(&mut *device, weights).await);
        }
        check(
            &mut *device,
            &mlp,
            &base.0,
            [&changed[0], &changed[1], &changed[2]],
            0.02,
        )
        .await;
    }

    /// Both weight forms of the same MLP: the E4M3 one within its activation quantization of the
    /// reference, the BF16 one within BF16 rounding.
    #[tokio::test(flavor = "current_thread")]
    async fn matches_the_reference_in_both_forms() {
        type H = Hidden<4096>;
        type I = Intermediate<12288>;
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let [base, changes @ ..] = weights::<H, I>();
        let mlp = quantized::<H, I>(&mut device, &base.1).await;
        let mut changed = Vec::new();
        for (_, stored) in &changes {
            changed.push(quantized::<H, I>(&mut device, stored).await);
        }
        check(
            &mut device,
            &mlp,
            &base.0,
            [&changed[0], &changed[1], &changed[2]],
            0.06,
        )
        .await;
        linears_match::<H, I>(&mut device).await;
    }

    /// The BF16 MLP at each listed hidden width and intermediate width, padded feature slices
    /// included, within BF16 rounding of the reference.
    #[tokio::test(flavor = "current_thread")]
    async fn bf16_matches_the_reference_at_every_width() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        linears_match::<Hidden<1024>, Intermediate<3072>>(&mut device).await;
        linears_match::<Hidden<2048>, Intermediate<6144>>(&mut device).await;
        linears_match::<Hidden<2560>, Intermediate<9728>>(&mut device).await;
        linears_match::<Hidden<5120>, Intermediate<14336>>(&mut device).await;
        linears_match::<Hidden<6656>, Intermediate<19968>>(&mut device).await;
    }
}
