use super::*;
use crate::ops::layout::Rows as Layout;

type Intermediate = super::Intermediate<25600>;

// Gate/up rows spill to HBM and each weight streams in slabs: no fused tiling of this width keeps
// them on chip yet.
impl<'d, 'x, Hidden: AxisName, Rows: AxisName>
    Module<(&'d mut Device, (&'x Spread<Hidden, Rows>, Live<Rows>))> for Mlp<Hidden, Intermediate>
{
    type Output = Spread<Hidden, Rows>;

    /// Only the live row groups are projected; the later groups land zeros.
    fn forward(
        &self,
        (device, (hidden, live)): (&'d mut Device, (&'x Spread<Hidden, Rows>, Live<Rows>)),
    ) -> Self::Output {
        // The pair lands in HBM, where SwiGLU reads its halves: its 256 features a slice match no
        // output block of the projection, so a copy into feature slices would gather across them.
        let mut pair: Packed<Intermediate, Rows> = HbmTensor::new();
        self.gate_up
            .affine
            .forward((
                &mut *device,
                (live, hidden.view(), |at| {
                    Layout::<Rows>::group(hidden.view(), at)
                }),
            ))
            .view()
            .to_hbm_view(&mut device.tdma, pair.view_mut());
        let product = SwiGlu.forward((&mut *device, (pair, live)));
        let mut output: Spread<Hidden, Rows> = DmTensor::new();
        self.down
            .affine
            .forward((
                &mut *device,
                (live, product.view(), |at| {
                    Layout::<Rows>::group(product.view(), at)
                }),
            ))
            .view()
            .to_dm_view(&mut device.tdma, output.view_mut());
        type Chunk<Hidden> = Symbol<crate::ops::normalization::Chunk<Hidden>>;
        type Size<Rows> = Symbol<Group<Rows>>;
        let groups = <m![Rows / ({ Size<Rows> })] as M>::SIZE;
        for later in 0..groups - live.count() {
            output
                .view_mut()
                .tile::<m![Rows / ({ Size<Rows> })], 1, m![
                    Rows / ({ Size<Rows> }) = 1 #{!} (Rows / ({ Size<Rows> })),
                    Rows % ({ Size<Rows> }),
                    Hidden % ({ Chunk<Hidden> })
                ]>(live.count() + later)
                .memset(const { bf16::from_f32(0.0) }, &mut device.sub);
        }
        output
    }
}
