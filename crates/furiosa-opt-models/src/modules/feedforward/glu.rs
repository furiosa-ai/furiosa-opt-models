//! The gated MLP: `down(activation(gate(x)) * up(x))`, three linears around an activation and a
//! product.

use crate::load::Named;
use crate::modules::residual::Sum;
use crate::{Fork, Then};

/// Computes `down(activation(gate(x)) * up(x))`; the linears are its checkpoint's `gate_proj`,
/// `up_proj` and `down_proj`.
pub type Glu<Gate, Activation, Up, Multiply, Down> = Sum<
    Fork<(
        Then<Named<"gate_proj", Gate>, Activation>,
        Named<"up_proj", Up>,
    )>,
    Then<Multiply, Named<"down_proj", Down>>,
>;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::Module;
    use crate::ops::attention::geometry::Hidden;
    use crate::ops::linear::{Affine, Groups};
    use crate::ops::normalization::Spread;
    use crate::ops::pointwise::{activation::Silu, multiply::Multiply};
    use crate::ops::swiglu::Intermediate;
    use crate::tensor::axis::{Lanes, Rows, Sixteen};
    use crate::tensor::live::{Live, tests::count};
    use crate::tensor::{parallel::Chip, scalar::Value};
    use crate::tests::DEVICE;
    use furiosa_opt_std::prelude::*;
    use furiosa_opt_std::runtime::launch;

    /// BF16 gate, up and down projections around SiLU and the product.
    type Bf16<H, I> = Glu<
        Affine<HbmTensor<bf16, Chip, m![I, H]>, ()>,
        Silu,
        Affine<HbmTensor<bf16, Chip, m![I, H]>, ()>,
        Multiply,
        Affine<HbmTensor<bf16, Chip, m![H, I]>, ()>,
    >;

    /// The state slots of a gated MLP: one unit a child.
    type Units = ((((), ()), ()), ((), ()));

    /// Each down row reads one channel in each of this many equal ranges of the intermediate width.
    const PARTS: usize = 8;

    fn coefficient(term: usize, row: usize, column: usize) -> bf16 {
        bf16::from_f32(((row * 31 + column * 13 + term * 47) % 37) as f32 / 16.0 - 1.125)
    }

    /// The hidden columns every gate and up row reads.
    fn columns(hidden: usize) -> [usize; 6] {
        [
            0,
            129,
            hidden / 4 - 1,
            hidden / 2,
            3 * hidden / 4 + 1,
            hidden - 1,
        ]
    }

    fn channel(intermediate: usize, row: usize, part: usize) -> usize {
        let range = intermediate / PARTS;
        part * range + (row * 17 + part * 131) % range
    }

    pub(crate) fn inputs(rows: usize, hidden: usize) -> Vec<bf16> {
        (0..rows)
            .flat_map(|row| {
                (0..hidden).map(move |column| {
                    bf16::from_f32(
                        ((row * 19 + column * 29 + row * column % 17) % 113) as f32 / 64.0 - 0.875,
                    )
                })
            })
            .collect()
    }

    /// `down(silu(gate x) * up x)` of every row: the gate, up, SiLU and product each round to
    /// BF16 and the down projection sums in FP32, within one BF16 ulp of every summed term.
    pub(crate) fn oracle(inputs: &[bf16], hidden: usize, intermediate: usize) -> Vec<(f32, f32)> {
        let rounded = |value: f32| bf16::from_f32(value).to_f32();
        inputs
            .chunks_exact(hidden)
            .flat_map(|row| {
                let activated = move |channel: usize| {
                    let term = |term: usize| {
                        columns(hidden)
                            .iter()
                            .enumerate()
                            .map(|(index, &column)| {
                                row[column].to_f32() * coefficient(term, channel, index).to_f32()
                            })
                            .sum::<f32>()
                    };
                    let gate = rounded(term(0));
                    rounded(rounded(gate / (1.0 + (-gate).exp())) * rounded(term(1)))
                };
                (0..hidden).map(move |output| {
                    (0..PARTS)
                        .map(|part| {
                            let term = activated(channel(intermediate, output, part))
                                * coefficient(2, output, part).to_f32();
                            (term, term.abs() / 128.0)
                        })
                        .fold((0.0, 0.0), |(sum, ulps), (term, ulp)| {
                            (sum + term, ulps + ulp)
                        })
                })
            })
            .collect()
    }

    /// The gate, up and down coefficients `oracle` reads, every other coefficient zero.
    pub(crate) fn weights(hidden: usize, intermediate: usize) -> [Vec<bf16>; 3] {
        let [mut gate, mut up] = [0, 1].map(|_| vec![bf16::from_f32(0.0); intermediate * hidden]);
        for channel in 0..intermediate {
            for (index, column) in columns(hidden).into_iter().enumerate() {
                gate[channel * hidden + column] = coefficient(0, channel, index);
                up[channel * hidden + column] = coefficient(1, channel, index);
            }
        }
        let mut down = vec![bf16::from_f32(0.0); hidden * intermediate];
        for row in 0..hidden {
            for part in 0..PARTS {
                down[row * intermediate + channel(intermediate, row, part)] =
                    coefficient(2, row, part);
            }
        }
        [gate, up, down]
    }

    async fn projection<Out: AxisName, In: AxisName>(
        device: &mut Device,
        weight: Vec<bf16>,
    ) -> Affine<HbmTensor<bf16, Chip, m![Out, In]>, ()> {
        Affine::new(
            HostTensor::<bf16, m![Out, In]>::from_vec(weight)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            (),
        )
    }

    #[device(chip = 1)]
    fn projected<H: AxisName, I: AxisName, R: AxisName>(
        device: &mut Device,
        mlp: &Bf16<H, I>,
        rows: &Value,
        input: &HbmTensor<bf16, Chip, m![R, H]>,
    ) -> HbmTensor<bf16, Chip, m![R, H]>
    where
        Bf16<H, I>: for<'d, 'x> Module<
                &'x Live<Spread<H, R>, R>,
                (),
                &'d mut Units,
                Output = Live<Spread<H, R>, R>,
            >,
    {
        let value = input.to_dm(&mut device.tdma);
        let input: Live<Spread<H, R>, R> = Live::<(), R>::read(rows).with(value);
        mlp.forward(&mut *device, (), &mut Units::default(), &input)
            .value()
            .to_hbm(&mut device.tdma)
    }

    /// Every live row gives the oracle's row, at every row live, at a partial last group and at
    /// all but one group.
    async fn check<H: AxisName, I: AxisName, R: AxisName>()
    where
        Bf16<H, I>: for<'d, 'x> Module<
                &'x Live<Spread<H, R>, R>,
                (),
                &'d mut Units,
                Output = Live<Spread<H, R>, R>,
            >,
    {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let [gate, up, down] = weights(H::SIZE, I::SIZE);
        let mlp: Bf16<H, I> = Sum(
            Fork((
                Then(Named(projection(&mut device, gate).await), Silu),
                Named(projection(&mut device, up).await),
            )),
            Then(Multiply, Named(projection(&mut device, down).await)),
        );
        let inputs = inputs(R::SIZE, H::SIZE);
        let expected = oracle(&inputs, H::SIZE, I::SIZE);
        let input = HostTensor::<bf16, m![R, H]>::from_vec(inputs)
            .to_hbm::<Chip, m![R, H]>(&mut device.pdma)
            .await
            .unwrap();
        let lanes = Lanes::<R>::SIZE;
        let groups = <Groups<R> as M>::SIZE;
        for live in [R::SIZE, R::SIZE - lanes + 1, (groups - 1) * lanes, 1] {
            let rows = count(live).to_hbm(&mut device.pdma).await.unwrap();
            let output = launch(projected, (&mut device, &mlp, &rows, &input))
                .await
                .unwrap()
                .to_host::<m![R, H]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for (index, (got, &(expected, ulps))) in output.iter().zip(&expected).enumerate() {
                let (row, column) = (index / H::SIZE, index % H::SIZE);
                let got = got.to_f32();
                assert!(
                    row >= live || (got - expected).abs() <= 0.03 * expected.abs().max(1.0) + ulps,
                    "{live} live rows, row {row} column {column}: {got} vs {expected}"
                );
            }
        }
    }

    /// Each decoder width's MLP matches the oracle on sixteen rows.
    #[tokio::test(flavor = "current_thread")]
    async fn matches_the_oracle_at_every_width() {
        check::<Hidden<1024>, Intermediate<3072>, Sixteen>().await;
        check::<Hidden<2048>, Intermediate<6144>, Sixteen>().await;
        check::<Hidden<2560>, Intermediate<9728>, Sixteen>().await;
        check::<Hidden<4096>, Intermediate<12288>, Sixteen>().await;
        check::<Hidden<4096>, Intermediate<14336>, Sixteen>().await;
        check::<Hidden<5120>, Intermediate<25600>, Sixteen>().await;
    }

    /// Above sixteen rows every row group reaches the down projection and the groups stay apart.
    #[tokio::test(flavor = "current_thread")]
    async fn keeps_row_groups_apart_above_sixteen_rows() {
        check::<Hidden<4096>, Intermediate<12288>, Rows<64>>().await;
    }
}
