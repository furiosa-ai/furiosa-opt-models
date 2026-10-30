//! The gated MLP with packed gate/up weights: `down(swiglu(gate_up(x)))`, its SwiGLU rounding
//! the SiLU of the gate to BF16 before the up half multiplies it.
//!
//! - Rows in HBM run the projections over rows and their SwiGLU in turn.
//! - A span's rows on chip run three stages its weight forms ([`spans`]) implement: the gate/up
//!   projections, the SwiGLU, and the down projection, which takes its product in two steps
//!   (staged, then projected). A schedule may land a tile's gate/up weights ahead of it.

use furiosa_opt_std::prelude::*;

use crate::load::{Checkpoint, Children, Context, Load, LoadError};
use crate::modules::generic::schedule::Begin;
use crate::ops::normalization::span::Owned;
use crate::ops::pointwise::SwiGlu;
use crate::tensor::form::Projections;
use crate::tensor::live::Live;
use crate::{Module, Op};
use core::marker::PhantomData;

mod rows;
pub mod spans;

/// Intermediate feature axis shared by dense feed-forward implementations.
#[derive(Clone, Debug)]
pub struct Intermediate<const N: usize, const PADDED: usize = N>;

impl<const N: usize, const PADDED: usize> AxisName for Intermediate<N, PADDED> {
    const NAME: Ident = Ident::new("Intermediate");
    const SIZE: usize = N;
}

/// The two weight matrices packed as gate followed by up.
#[derive(Debug, Clone)]
pub struct GateUp<I>(PhantomData<I>);

impl<I: AxisName> AxisName for GateUp<I> {
    const NAME: Ident = Ident::new("GateUp");
    const SIZE: usize = I::SIZE * 2;
}

/// Storage width of the down projection, including its kernel's zero padding.
#[derive(Debug, Clone)]
pub struct Down<I>(PhantomData<I>);

impl<const N: usize, const PADDED: usize> AxisName for Down<Intermediate<N, PADDED>> {
    const NAME: Ident = Intermediate::<N, PADDED>::NAME;
    const SIZE: usize = {
        assert!(PADDED >= N);
        PADDED
    };
}

/// A packed gate/up projection and a down projection, in a weight form each: BF16 linears or
/// block-scaled E4M3 ([`spans::fp8`]).
#[derive(DeviceSend)]
pub struct Mlp<Hidden: AxisName, Intermediate: AxisName, GateUpProjection, DownProjection>
where
    Down<Intermediate>: AxisName,
{
    gate_up: GateUpProjection,
    down: DownProjection,
    features: PhantomData<(Hidden, Intermediate)>,
}

impl<Hidden: AxisName, Intermediate: AxisName, GateUpProjection, DownProjection>
    Mlp<Hidden, Intermediate, GateUpProjection, DownProjection>
where
    Down<Intermediate>: AxisName,
{
    pub const fn new(gate_up: GateUpProjection, down: DownProjection) -> Self {
        Self {
            gate_up,
            down,
            features: PhantomData,
        }
    }

    /// The gate/up and down projections [`Self::new`] took.
    pub fn into_parts(self) -> (GateUpProjection, DownProjection) {
        (self.gate_up, self.down)
    }

    /// The gate/up and down projections, borrowed.
    pub fn parts(&self) -> (&GateUpProjection, &DownProjection) {
        (&self.gate_up, &self.down)
    }
}

/// The checkpoint's gate and up projections combine into the packed weight form, then the down
/// projection loads into its own.
impl<'a, H: AxisName, I: AxisName, F, D> Load<&'a Checkpoint> for Mlp<H, I, F, D>
where
    Down<I>: AxisName,
    F: Load<Projections<'a, 2>>,
    D: Load<Projections<'a, 1>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let mut children = Children::default();
            let names = ["gate_proj", "up_proj"];
            let gate_up = cx.field("gate_up", "", |cx| F::load(cx, Projections { from, names }));
            let gate_up = children.ask(gate_up)?;
            let names = ["down_proj"];
            let down = cx.field("down", "", |cx| D::load(cx, Projections { from, names }));
            let down = children.ask(down)?;
            let (gate_up, down) = children.node(gate_up.zip(down))?;
            Ok(Self::new(gate_up, down))
        })
    }
}

/// A borrowed MLP borrows both weight forms.
impl<'a, H: AxisName, I: AxisName, F: Load<&'a G>, D: Load<&'a E>, G, E> Load<&'a Mlp<H, I, G, E>>
    for Mlp<H, I, F, D>
where
    Down<I>: AxisName,
{
    fn load(context: &mut Context<'_>, from: &'a Mlp<H, I, G, E>) -> Result<Self, LoadError> {
        let gate_up = context.field("gate_up", "", |cx| F::load(cx, &from.gate_up))?;
        let down = context.field("down", "", |cx| D::load(cx, &from.down))?;
        Ok(Self::new(gate_up, down))
    }
}

// Gate then up in the packed projection.
axes![Term = 2];

/// Rows in HBM: the gate/up projection, its SwiGLU and the down projection, each over rows.
impl<'x, H: AxisName, I: AxisName, R: AxisName, X: M, F, D, Y, Z, O, Env: Copy>
    Module<&'x HbmTensor<bf16, X, m![R, H]>, Env, &mut ()> for Mlp<H, I, F, D>
where
    Down<I>: AxisName,
    F: for<'s> Module<&'x HbmTensor<bf16, X, m![R, H]>, Env, &'s mut (), Output = Y>,
    SwiGlu: for<'a, 's> Module<&'a Y, Env, &'s mut (), Output = Z>,
    D: for<'a, 's> Module<&'a Z, Env, &'s mut (), Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        input: &'x HbmTensor<bf16, X, m![R, H]>,
    ) -> O {
        let projected = self.gate_up.forward(&mut *device, env, &mut (), input);
        let product = Module::forward(&SwiGlu, &mut *device, env, &mut (), &projected);
        self.down.forward(device, env, &mut (), &product)
    }
}

/// A span's rows on chip: the three stages, one after another.
impl<'x, H, I, Rows, Capacity, F, D, A, G, P, C, Env>
    Module<&'x Live<Owned<H, Rows>, Capacity>, Env, &mut ()> for Mlp<H, I, F, D>
where
    H: AxisName,
    I: AxisName,
    Rows: AxisName,
    Capacity: AxisName,
    Down<I>: AxisName,
    F: for<'s> Module<Begin<()>, (), &'s mut (), Output = A>,
    F: for<'s> Module<(A, Live<&'x Owned<H, Rows>, Capacity>), (), &'s mut (), Output = G>,
    SwiGlu: for<'s> Module<Live<G, Capacity>, (), &'s mut (), Output = P>,
    D: for<'s> Module<Begin<Live<P, Capacity>>, (), &'s mut (), Output = C>,
    D: for<'s> Module<Live<C, Capacity>, (), &'s mut (), Output = Owned<H, Rows>>,
{
    type Output = Live<Owned<H, Rows>, Capacity>;

    fn forward(
        &self,
        device: &mut Device,
        _: Env,
        _: &mut (),
        normalized: &'x Live<Owned<H, Rows>, Capacity>,
    ) -> Self::Output {
        let tile = normalized.with(());
        let weights = self.gate_up.forward(&mut *device, (), &mut (), Begin(()));
        let projected = self.gate_up.forward(
            &mut *device,
            (),
            &mut (),
            (weights, tile.with(normalized.value())),
        );
        let product = Module::forward(&SwiGlu, &mut *device, (), &mut (), tile.with(projected));
        let staged = self
            .down
            .forward(&mut *device, (), &mut (), Begin(tile.with(product)));
        tile.with(self.down.forward(device, (), &mut (), tile.with(staged)))
    }
}

/// Rows in HBM, a tile of rows at a time; as for rows on chip, the SiLU of the gate rounds to
/// BF16 before it multiplies the up half.
impl<'x, I: AxisName, R: AxisName, Chips: M>
    Op<&'x HbmTensor<bf16, Chips, m![R, { Symbol<GateUp<I>> }]>> for SwiGlu
{
    type Output = HbmTensor<bf16, Chips, m![R, I]>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x HbmTensor<bf16, Chips, m![R, { Symbol<GateUp<I>> }]>,
    ) -> HbmTensor<bf16, Chips, m![R, I]> {
        let mut output = HbmTensor::new();
        Op::forward(&SwiGlu, device, (input, &mut output));
        output
    }
}

#[cfg(test)]
mod tests {
    use crate::Module;
    use furiosa_opt_std::runtime::launch;

    use super::*;
    use crate::ops::attention::geometry::Hidden;
    use crate::ops::linear::Linear;
    use crate::ops::normalization;
    use crate::tensor::axis::Rows;
    use crate::tensor::form::Weight;
    use crate::tensor::parallel::Chip;
    use crate::tests::DEVICE;

    struct Fixture;

    type RowsMlp<H, I> = Mlp<H, I, Linear<H, GateUp<I>>, Linear<I, H>>;

    impl Fixture {
        fn silu(x: f64) -> f64 {
            x / (1.0 + (-x).exp())
        }

        fn rounded(x: f64) -> f64 {
            f64::from(bf16::from_f32(x as f32).to_f32())
        }
    }

    /// `R` gate/up rows in HBM, every `stride`-th row checked against the reference with the SiLU
    /// rounded to BF16 first.
    impl Fixture {
        async fn gated_rows_match<R: AxisName>(stride: usize) {
            let _lease = DEVICE.lock().await;
            type I = Intermediate<14336>;
            let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
            let width = I::SIZE;
            let packed: Vec<bf16> =
                normalization::tests::random(R::SIZE * 2 * width, 0x2545_f491_4f6c_dd1d)
                    .iter()
                    .map(|v| bf16::from_f32(v.to_f32() * 6.0))
                    .collect();
            let input = HostTensor::<bf16, m![R, { Symbol<GateUp<I>> }]>::from_vec(packed.clone())
                .to_hbm::<Chip, m![R, { Symbol<GateUp<I>> }]>(&mut device.pdma)
                .await
                .unwrap();
            let output = launch(gated_rows, (&mut device, &input))
                .await
                .unwrap()
                .to_host::<m![R, I]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for (index, got) in output.iter().enumerate() {
                let (row, column) = (index / width, index % width);
                if row % stride != 0 {
                    continue;
                }
                let gate = f64::from(packed[row * 2 * width + column].to_f32());
                let up = f64::from(packed[row * 2 * width + width + column].to_f32());
                let expected = Self::rounded(Self::silu(gate)) * up;
                let got = f64::from(got.to_f32());
                assert!(
                    (got - expected).abs() <= expected.abs() / 128.0 + 1e-3,
                    "row {row} column {column}: {got} vs {expected}"
                );
            }
        }
    }
    /// The MLP over `R` rows in HBM: `checked` rows match the reference with BF16 gate, up, SiLU
    /// and product, within 2 % of each row's RMS.
    impl Fixture {
        async fn rows_mlp_matches<R: AxisName>(checked: &[usize]) {
            let _lease = DEVICE.lock().await;
            type H = Hidden<5376>;
            type I = Intermediate<14336>;
            let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
            let random = normalization::tests::random;
            let (hidden, width) = (H::SIZE, I::SIZE);
            let inputs = random(R::SIZE * hidden, 0x2545_f491_4f6c_dd1d);
            let up_weight = random(2 * width * hidden, 0x9e37_79b9_7f4a_7c15);
            let down_weight = random(hidden * width, 0x0123_4567_89ab_cdef);
            let widened = |values: &[bf16]| values.iter().map(|v| v.to_f32()).collect::<Vec<f32>>();
            let (x, w1, w2) = (widened(&inputs), widened(&up_weight), widened(&down_weight));
            let up: Linear<H, GateUp<I>> = Linear::new(Weight::new(
                HostTensor::<bf16, m![{ Symbol<GateUp<I>> }, H]>::from_vec(up_weight)
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
            ));
            let down: Linear<I, H> = Linear::new(Weight::new(
                HostTensor::<bf16, m![H, I]>::from_vec(down_weight)
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
            ));
            let mlp: RowsMlp<H, I> = Mlp::new(up, down);
            let input = HostTensor::<bf16, m![R, H]>::from_vec(inputs)
                .to_hbm::<Chip, m![R, H]>(&mut device.pdma)
                .await
                .unwrap();
            let output = launch(mlp_rows, (&mut device, &mlp, &input))
                .await
                .unwrap()
                .to_host::<m![R, H]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            let dot = |a: &[f32], b: &[f32]| -> f64 {
                a.iter()
                    .zip(b)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum()
            };
            for &row in checked {
                let x = &x[row * hidden..][..hidden];
                let product: Vec<f32> = (0..width)
                    .map(|j| {
                        let gate = Self::rounded(dot(x, &w1[j * hidden..][..hidden]));
                        let up = Self::rounded(dot(x, &w1[(width + j) * hidden..][..hidden]));
                        Self::rounded(Self::rounded(Self::silu(gate)) * up) as f32
                    })
                    .collect();
                let expected: Vec<f64> = (0..hidden)
                    .map(|o| dot(&product, &w2[o * width..][..width]))
                    .collect();
                let rms = (expected.iter().map(|e| e * e).sum::<f64>() / hidden as f64).sqrt();
                for (column, expected) in expected.iter().enumerate() {
                    let got = f64::from(output[row * hidden + column].to_f32());
                    assert!(
                        (got - expected).abs() <= 0.02 * rms,
                        "row {row} column {column}: {got} vs {expected} (row rms {rms})"
                    );
                }
            }
        }
    }

    /// The MLP over `R` rows in HBM with the gated MLP oracle's sparse weights: every row matches
    /// the oracle within 3 % and one BF16 ulp of every summed term.
    impl Fixture {
        async fn oracle_matches<H: AxisName, I: AxisName, R: AxisName>()
        where
            Down<I>: AxisName,
        {
            use crate::modules::feedforward::glu::tests::{inputs, oracle, weights};
            let _lease = DEVICE.lock().await;
            let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
            let [gate, up, down] = weights(H::SIZE, I::SIZE);
            let up =
                HostTensor::<bf16, m![{ Symbol<GateUp<I>> }, H]>::from_vec([gate, up].concat())
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap();
            let down = HostTensor::<bf16, m![H, I]>::from_vec(down)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let mlp: RowsMlp<H, I> =
                Mlp::new(Linear::new(Weight::new(up)), Linear::new(Weight::new(down)));
            let inputs = inputs(R::SIZE, H::SIZE);
            let expected = oracle(&inputs, H::SIZE, I::SIZE);
            let input = HostTensor::<bf16, m![R, H]>::from_vec(inputs)
                .to_hbm::<Chip, m![R, H]>(&mut device.pdma)
                .await
                .unwrap();
            let output = launch(mlp_rows, (&mut device, &mlp, &input))
                .await
                .unwrap()
                .to_host::<m![R, H]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for (index, (got, &(expected, ulps))) in output.iter().zip(&expected).enumerate() {
                let got = got.to_f32();
                assert!(
                    (got - expected).abs() <= 0.03 * expected.abs().max(1.0) + ulps,
                    "row {} column {}: {got} vs {expected}",
                    index / H::SIZE,
                    index % H::SIZE
                );
            }
        }
    }

    #[device(chip = 1)]
    fn gated_rows<I: AxisName, R: AxisName>(
        device: &mut Device,
        input: &HbmTensor<bf16, Chip, m![R, { Symbol<GateUp<I>> }]>,
    ) -> HbmTensor<bf16, Chip, m![R, I]> {
        Op::forward(&SwiGlu, device, input)
    }
    #[device(chip = 1)]
    fn mlp_rows<H: AxisName, I: AxisName, R: AxisName>(
        device: &mut Device,
        mlp: &RowsMlp<H, I>,
        input: &HbmTensor<bf16, Chip, m![R, H]>,
    ) -> HbmTensor<bf16, Chip, m![R, H]>
    where
        Down<I>: AxisName,
    {
        mlp.forward(device, (), &mut (), input)
    }
    /// Rows in HBM take `bf16(bf16(silu(gate)) * up)` of their two halves, the SiLU rounded first.
    #[tokio::test(flavor = "current_thread")]
    async fn rows_in_hbm_round_the_silu_first() {
        Fixture::gated_rows_match::<Rows<128>>(1).await;
    }
    /// Four rows at the ends of the first two tiles through the whole MLP.
    #[tokio::test(flavor = "current_thread")]
    async fn rows_mlp_matches_the_reference() {
        Fixture::rows_mlp_matches::<Rows<128>>(&[0, 63, 64, 127]).await;
    }

    /// Each decoder width's MLP over rows matches the gated MLP's oracle.
    #[tokio::test(flavor = "current_thread")]
    async fn rows_mlp_matches_the_oracle_at_every_width() {
        Fixture::oracle_matches::<Hidden<1024>, Intermediate<3072>, Rows<128>>().await;
        Fixture::oracle_matches::<Hidden<2048>, Intermediate<6144>, Rows<128>>().await;
        Fixture::oracle_matches::<Hidden<2560>, Intermediate<9728>, Rows<128>>().await;
        Fixture::oracle_matches::<Hidden<4096>, Intermediate<12288>, Rows<128>>().await;
        Fixture::oracle_matches::<Hidden<4096>, Intermediate<14336>, Rows<128>>().await;
        Fixture::oracle_matches::<Hidden<5120>, Intermediate<25600>, Rows<128>>().await;
    }
}
