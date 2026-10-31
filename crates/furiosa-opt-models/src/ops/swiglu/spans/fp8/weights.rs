use super::*;

impl<Hidden: AxisName, Intermediate: AxisName, W, S> Op<Begin<usize>>
    for fp8::Rows<m![Hidden], m![Term, Intermediate], W, S>
where
    W: Borrow<HbmTensor<f8e4m3, Chip, Blocked<Hidden, Intermediate>>>,
{
    type Output = GateUp<Hidden, Intermediate>;

    fn forward(&self, context: &mut Device, Begin(panel): Begin<usize>) -> Self::Output {
        self.0
            .borrow()
            .view()
            .tile::<m![Hidden / 1024], 1, m![Term, Hidden / 1024 = 1 # (Hidden / 1024),
                Intermediate / (Intermediate / 64), Hidden % 1024 / 128, { Out<Intermediate> },
                Hidden % 128]>(panel)
            .to_dm(&mut context.tdma)
    }
}

impl<Hidden: AxisName, Intermediate: AxisName, W, S> Op<Begin<()>>
    for fp8::Rows<m![Hidden], m![Term, Intermediate], W, S>
where
    W: Borrow<HbmTensor<f8e4m3, Chip, Blocked<Hidden, Intermediate>>>,
{
    type Output = GateUp<Hidden, Intermediate>;

    fn forward(&self, context: &mut Device, _: Begin<()>) -> Self::Output {
        self.forward(context, Begin(0))
    }
}

impl<Hidden: AxisName, Intermediate: AxisName, W, S> Op<Begin<usize>>
    for fp8::Rows<m![Intermediate], m![Hidden], W, S>
where
    W: Borrow<HbmTensor<f8e4m3, Chip, Stored<Hidden, Intermediate>>>,
{
    type Output = Down<Hidden, Intermediate>;

    fn forward(&self, context: &mut Device, Begin(panel): Begin<usize>) -> Self::Output {
        let mut down = Down::new();
        self.0
            .borrow()
            .view()
            .tile::<m![{ Out<Intermediate> } / 64], 1, StoredPanel<Hidden, Intermediate>>(panel)
            .to_dm_view(&mut context.tdma, down.view_mut());
        down
    }
}

pub type GateUpWeights<H, I, X = Hbm> = fp8::Rows<
    m![H],
    m![Term, I],
    <X as Memory>::Tensor<f8e4m3, Chip, Blocked<H, I>>,
    <X as Memory>::Tensor<f32, Chip, m![Term, I]>,
>;
pub type DownWeights<H, I, X = Hbm> = fp8::Rows<
    m![I],
    m![H],
    <X as Memory>::Tensor<f8e4m3, Chip, Stored<H, I>>,
    <X as Memory>::Tensor<f32, Chip, m![H]>,
>;
pub type QuantizedMlp<H, I, X = Hbm> = Mlp<H, I, GateUpWeights<H, I, X>, DownWeights<H, I, X>>;

impl<'a, H: AxisName, I: AxisName, W, S> Load<Projections<'a, 2>>
    for fp8::Rows<m![H], m![Term, I], W, S>
where
    W: Load<HostTensor<f8e4m3, m![{ Chip }, { m![Term, I, H] }]>>,
    S: Load<HostTensor<f32, m![{ Chip }, { m![Term, I] }]>>,
{
    fn load(context: &mut Context<'_>, projections: Projections<'a, 2>) -> Result<Self, LoadError> {
        let [(gate, gates), (up, ups)] = projections.read(context, |from, cx| {
            cx.field("weight", "weight", |cx| {
                fp8::folded::<m![H], m![I]>(from, cx)
            })
        })?;
        let codes = HostTensor::<f8e4m3, m![{ Chip }, { m![Term, I, H] }]>::from_vec(
            [gate, up].concat().repeat(Chip::SIZE),
        );
        let scales = HostTensor::<f32, m![{ Chip }, { m![Term, I] }]>::from_vec(
            [gates, ups].concat().repeat(Chip::SIZE),
        );
        Ok(Self::new(
            context.field("0", "", |cx| Load::load(cx, codes))?,
            context.field("1", "", |cx| Load::load(cx, scales))?,
        ))
    }
}

impl<'a, H: AxisName, I: AxisName, W, S> Load<Projections<'a, 1>> for fp8::Rows<m![I], m![H], W, S>
where
    W: Load<HostTensor<f8e4m3, m![{ Chip }, { m![H, I] }]>>,
    S: Load<HostTensor<f32, m![{ Chip }, { m![H] }]>>,
{
    fn load(context: &mut Context<'_>, projections: Projections<'a, 1>) -> Result<Self, LoadError> {
        let [(codes, scales)] = projections.read(context, |from, cx| {
            cx.field("weight", "weight", |cx| {
                fp8::folded::<m![I], m![H]>(from, cx)
            })
        })?;
        let codes =
            HostTensor::<f8e4m3, m![{ Chip }, { m![H, I] }]>::from_vec(codes.repeat(Chip::SIZE));
        let scales =
            HostTensor::<f32, m![{ Chip }, { m![H] }]>::from_vec(scales.repeat(Chip::SIZE));
        Ok(Self::new(
            context.field("0", "", |cx| Load::load(cx, codes))?,
            context.field("1", "", |cx| Load::load(cx, scales))?,
        ))
    }
}
