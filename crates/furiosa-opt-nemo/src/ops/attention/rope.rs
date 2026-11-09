use crate::{
    DeviceModule,
    ops::{cast::Convert, layout::Packet},
};
use furiosa_opt_std::prelude::*;

use super::full::heads::tiles;

pub mod group;
pub mod resident;

use crate::ops::layout::axes::Chip;
use crate::ops::vector::Zip;

use crate::ops::layout::BroadcastClusters as Cluster;

// The rotation pairs each position with the one a half-head away: `Half` is the half a value is
// read from, `OutHalf` the half it is rotated into, `Row` its position within either.
axes![Half = 2, OutHalf = 2];

#[derive(Clone, Debug)]
pub struct Row<Head>(core::marker::PhantomData<Head>);

impl<Head: AxisName> AxisName for Row<Head> {
    const NAME: Ident = Ident::new("Row");
    const SIZE: usize = Head::SIZE / 2;
}

type Table<Slice, Count, Row, Chips = Chip> =
    DmTensor<bf16, Chips, Cluster, Slice, m![{ Count }, OutHalf, Row]>;

type CoefficientRows<Rows, Rotated, Chips = Chip> = HbmTensor<bf16, Chips, m![Rows, Rotated]>;

#[cfg(test)]
mod tests {
    use furiosa_opt_std::prelude::*;
    use furiosa_opt_std::runtime::launch;

    use super::super::tests::Head;
    use super::Chip;

    use crate::ops::layout::axes::Single;

    axes![Groups = 2];

    #[device(chip = 1)]
    fn rotate<D: AxisName, R: AxisName>(
        device: &mut Device,
        rotary: &super::Rotary<D, R>,
        input: &HbmTensor<bf16, Chip, m![Groups, D]>,
        rotation: &super::Rotation<R>,
    ) -> HbmTensor<bf16, Chip, m![Groups, D]> {
        let mut copied = HbmTensor::new();
        input
            .view()
            .to_hbm_view(&mut device.tdma, copied.view_mut());
        rotary.rotate(device, copied, rotation)
    }

    async fn check_products<D: AxisName, R: AxisName>(device: &mut Device) {
        let rotary = super::Rotary::<D, R>::new();
        let input = (0..Groups::SIZE * D::SIZE)
            .map(|at| bf16::from_f32((at * 7 % 137) as f32 / 64.0 - 1.0))
            .collect::<Vec<_>>();
        let low = (0..R::SIZE)
            .map(|i| bf16::from_f32(if i < R::SIZE / 2 { 0.8 } else { 0.6 }))
            .collect::<Vec<_>>();
        let high = (0..R::SIZE)
            .map(|i| bf16::from_f32(if i < R::SIZE / 2 { -0.6 } else { 0.8 }))
            .collect::<Vec<_>>();
        let values = HostTensor::<bf16, m![Groups, D]>::from_vec(input.clone())
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let rotation = super::Rotation {
            low: HostTensor::<bf16, m![Single, R]>::from_vec(low.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            high: HostTensor::<bf16, m![Single, R]>::from_vec(high.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
        };
        let actual = launch(rotate, (&mut *device, &rotary, &values, &rotation))
            .await
            .unwrap()
            .to_host::<m![Groups, D]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (at, actual) in actual.into_iter().enumerate() {
            let channel = at % D::SIZE;
            let pair = channel % (D::SIZE / 2);
            let expected = if pair < R::SIZE / 2 {
                let start = at / D::SIZE * D::SIZE;
                let coefficient = pair + channel / (D::SIZE / 2) * (R::SIZE / 2);
                let a = bf16::from_f32(input[start + pair].to_f32() * low[coefficient].to_f32());
                let b = bf16::from_f32(
                    input[start + D::SIZE / 2 + pair].to_f32() * high[coefficient].to_f32(),
                );
                bf16::from_f32(a.to_f32() + b.to_f32())
            } else {
                input[at]
            };
            assert_eq!(actual, expected, "channel {at}");
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn rounds_products_before_adding() {
        axes![Full = 256, Partial = 512];
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(rotate.topology()).unwrap();
        check_products::<Full, Full>(&mut device).await;
        check_products::<Partial, Head>(&mut device).await;
    }
}

#[derive(Clone, Debug)]
struct PairWidth<Rotated>(core::marker::PhantomData<Rotated>);

impl<Rotated: AxisName> AxisName for PairWidth<Rotated> {
    const NAME: Ident = Ident::new("PairWidth");
    const SIZE: usize = Rotated::SIZE / 2;
}

/// Whole heads within each rotary DM slice.
type HeadTile<Heads, Head, D = bf16, Chips = Chip> =
    DmTensor<D, Chips, Cluster, m![{Heads} # 256], m![Head]>;

/// Rotate the first Rotated/2 pairs across the two Head/2 halves by each row's rotation; `D` is
/// the element type of the rotated key output.
#[derive(DeviceSend)]
pub struct Rotary<Head: AxisName, Rotated: AxisName = Head, D: Scalar = bf16, Chips: M = Chip>(
    core::marker::PhantomData<(Head, Rotated, D, Chips)>,
);

impl<Head: AxisName, Rotated: AxisName, D: Scalar, Chips: M> Rotary<Head, Rotated, D, Chips> {
    pub const fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}

impl<'q, 'k, Head: AxisName, Width: AxisName, R: AxisName, Chips: M>
    crate::Module<(
        &mut Device,
        (
            (
                &'q HbmTensor<bf16, Chips, m![R, Width]>,
                &'k HbmTensor<bf16, Chips, m![R, Width]>,
            ),
            &Rotation<Head, Chips, R>,
        ),
    )> for Rotary<Head, Head, bf16, Chips>
{
    type Output = (
        HbmTensor<bf16, Chips, m![R, Width]>,
        HbmTensor<bf16, Chips, m![R, Width]>,
    );

    fn forward(
        &self,
        (device, ((query, key), rotation)): (
            &mut Device,
            (
                (
                    &'q HbmTensor<bf16, Chips, m![R, Width]>,
                    &'k HbmTensor<bf16, Chips, m![R, Width]>,
                ),
                &Rotation<Head, Chips, R>,
            ),
        ),
    ) -> Self::Output {
        (
            Heads::<Width, R, Head, Chips, _>(core::marker::PhantomData)
                .forward((&mut *device, (query, rotation))),
            Heads::<Width, R, Head, Chips, _>(core::marker::PhantomData)
                .forward((device, (key, rotation))),
        )
    }
}

impl<'q, 'k, Head: AxisName, Width: AxisName, R: AxisName, Chips: M>
    crate::Module<(
        &mut Device,
        (
            (
                &'q tiles::Major<R, Width, Head, Chips>,
                &'k tiles::Major<R, Width, Head, Chips>,
            ),
            &Rotation<Head, Chips, R>,
        ),
    )> for Rotary<Head, Head, bf16, Chips>
{
    type Output = (
        tiles::Major<R, Width, Head, Chips>,
        tiles::Major<R, Width, Head, Chips>,
    );

    fn forward(
        &self,
        (device, ((query, key), rotation)): (
            &mut Device,
            (
                (
                    &'q tiles::Major<R, Width, Head, Chips>,
                    &'k tiles::Major<R, Width, Head, Chips>,
                ),
                &Rotation<Head, Chips, R>,
            ),
        ),
    ) -> Self::Output {
        (
            Heads::<Width, R, Head, Chips, _>(core::marker::PhantomData)
                .forward((&mut *device, (query, rotation))),
            Heads::<Width, R, Head, Chips, _>(core::marker::PhantomData)
                .forward((device, (key, rotation))),
        )
    }
}

struct Heads<Width, R, Head, Chips, Q>(core::marker::PhantomData<(Width, R, Head, Chips, Q)>);

impl<Head: AxisName, Chips: M, Width: AxisName, R: AxisName, Q: M, L>
    crate::Module<(&mut Device, (&L, &Rotation<Head, Chips, R>))>
    for Heads<Width, R, Head, Chips, Q>
where
    L: tiles::Layout<R, Width, Head, Chips, Order = Q>,
{
    type Output = L;

    fn forward(
        &self,
        (device, (input, rotation)): (&mut Device, (&L, &Rotation<Head, Chips, R>)),
    ) -> L {
        let mut output = L::new();
        tiles::Strips::<R, Width, Head, _, Chips, Q>(
            core::marker::PhantomData,
            Rotate { rotation },
        )
        .forward((device, (input, &mut output)));
        output
    }
}

impl<Head: AxisName, Chips: M, Width: AxisName, R: AxisName, Q: M, L>
    crate::Module<(&mut Device, ((&L, &L), &Rotation<Head, Chips, R>))>
    for Heads<Width, R, Head, Chips, Q>
where
    L: tiles::Layout<R, Width, Head, Chips, Order = Q>,
{
    type Output = (L, L);

    fn forward(
        &self,
        (device, ((query, key), rotation)): (&mut Device, ((&L, &L), &Rotation<Head, Chips, R>)),
    ) -> (L, L) {
        let (mut rotated_query, mut rotated_key) = (L::new(), L::new());
        tiles::Strips::<R, Width, Head, _, Chips, Q>(
            core::marker::PhantomData,
            Rotate { rotation },
        )
        .forward((
            device,
            ((query, key), (&mut rotated_query, &mut rotated_key)),
        ));
        (rotated_query, rotated_key)
    }
}

impl<Head: AxisName, Width: AxisName, R: AxisName, Chips: M>
    crate::Module<(
        &mut Device,
        (
            &(
                &HbmTensor<bf16, Chips, m![R, Width]>,
                &HbmTensor<bf16, Chips, m![R, Width]>,
            ),
            &Rotation<Head, Chips, R>,
        ),
    )> for Rotary<Head, Head, bf16, Chips>
{
    type Output = (
        HbmTensor<bf16, Chips, m![R, Width]>,
        HbmTensor<bf16, Chips, m![R, Width]>,
    );

    fn forward(
        &self,
        (device, (rows, rotation)): (
            &mut Device,
            (
                &(
                    &HbmTensor<bf16, Chips, m![R, Width]>,
                    &HbmTensor<bf16, Chips, m![R, Width]>,
                ),
                &Rotation<Head, Chips, R>,
            ),
        ),
    ) -> Self::Output {
        Heads::<Width, R, Head, Chips, _>(core::marker::PhantomData)
            .forward((device, (*rows, rotation)))
    }
}

impl<Head: AxisName, Width: AxisName, R: AxisName, Chips: M>
    crate::Module<(
        &mut Device,
        (
            &(
                &tiles::Major<R, Width, Head, Chips>,
                &tiles::Major<R, Width, Head, Chips>,
            ),
            &Rotation<Head, Chips, R>,
        ),
    )> for Rotary<Head, Head, bf16, Chips>
{
    type Output = (
        tiles::Major<R, Width, Head, Chips>,
        tiles::Major<R, Width, Head, Chips>,
    );

    fn forward(
        &self,
        (device, (rows, rotation)): (
            &mut Device,
            (
                &(
                    &tiles::Major<R, Width, Head, Chips>,
                    &tiles::Major<R, Width, Head, Chips>,
                ),
                &Rotation<Head, Chips, R>,
            ),
        ),
    ) -> Self::Output {
        Heads::<Width, R, Head, Chips, _>(core::marker::PhantomData)
            .forward((device, (*rows, rotation)))
    }
}

struct Rotate<'r, Head: AxisName, R: AxisName, Chips: M> {
    rotation: &'r Rotation<Head, Chips, R>,
}

type Lines<Width, Head> = m![{ tiles::T<Width, Head> } % ({ tiles::D<Width, Head> })];
type TileCoefficients<Width, Head, Chips> = VrfTensor<
    f32,
    Chips,
    tiles::Half<Width, Head>,
    tiles::Slices<Width, Head>,
    m![{ Lines<Width, Head> }, OutHalf, { Symbol<Row<Head>> }],
>;

struct FetchTiles<Width, Head, R, Chips>(core::marker::PhantomData<(Width, Head, R, Chips)>);

impl<Head: AxisName, R: AxisName, Chips: M, Width: AxisName>
    crate::Module<(&mut Device, (&Rotation<Head, Chips, R>, usize))>
    for FetchTiles<Width, Head, R, Chips>
{
    type Output = (
        TileCoefficients<Width, Head, Chips>,
        TileCoefficients<Width, Head, Chips>,
    );

    fn forward(
        &self,
        (device, (rotation, at)): (&mut Device, (&Rotation<Head, Chips, R>, usize)),
    ) -> (
        TileCoefficients<Width, Head, Chips>,
        TileCoefficients<Width, Head, Chips>,
    ) {
        let rows = |device: &mut Device, table: &CoefficientRows<R, Head, Chips>| {
            // SAFETY: the tiles split rows in order and each coefficient row is its two halves.
            let rows = unsafe {
                table.view().reshape::<Chips, m![{ tiles::S<R, Width, Head> }, { tiles::T<Width, Head> }, OutHalf, { Symbol<Row<Head>> }]>()
            };
            let landed: DmTensor<bf16, Chips, tiles::Half<Width, Head>, tiles::Slices<Width, Head>, m![{ Lines<Width, Head> }, OutHalf, { Symbol<Row<Head>> }]> = rows
                .tile::<m![{ tiles::S<R, Width, Head> }], 1, m![{ tiles::S<R, Width, Head> } = 1 # ({ tiles::S<R, Width, Head> }), { tiles::T<Width, Head> }, OutHalf, { Symbol<Row<Head>> }]>(at)
                .to_dm(&mut device.tdma);
            let operand: TileCoefficients<Width, Head, Chips> = device.sub.begin(landed.view())
                .fetch::<m![{ Lines<Width, Head> }, OutHalf, { Symbol<Row<Head>> } / 16], m![{ Symbol<Row<Head>> } % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![{ Lines<Width, Head> }, OutHalf, { Symbol<Row<Head>> } / 8], m![{ Symbol<Row<Head>> } % 8]>()
                .to_vrf();
            operand
        };
        (rows(device, &rotation.low), rows(device, &rotation.high))
    }
}

struct RotateTile<Head>(core::marker::PhantomData<Head>);

impl<Head: AxisName, Chips: M, Width: AxisName, Q: M>
    crate::Module<(
        &mut Device,
        (
            &tiles::Tile<bf16, Width, Head, Chips, Q>,
            &(
                TileCoefficients<Width, Head, Chips>,
                TileCoefficients<Width, Head, Chips>,
            ),
            &mut tiles::Tile<bf16, Width, Head, Chips, Q>,
        ),
    )> for RotateTile<Head>
{
    type Output = ();

    fn forward(
        &self,
        (device, (rows, (low, high), output)): (
            &mut Device,
            (
                &tiles::Tile<bf16, Width, Head, Chips, Q>,
                &(
                    TileCoefficients<Width, Head, Chips>,
                    TileCoefficients<Width, Head, Chips>,
                ),
                &mut tiles::Tile<bf16, Width, Head, Chips, Q>,
            ),
        ),
    ) {
        // SAFETY: a head is its two halves in order.
        let halves = unsafe {
            rows.view().reshape::<Chips, tiles::Half<Width, Head>, tiles::Slices<Width, Head>, m![{ Q }, Half, { Symbol<Row<Head>> }]>()
        };
        // SAFETY: the two output halves are the head in order.
        let rotated = unsafe {
            output.view_mut().reshape::<Chips, tiles::Half<Width, Head>, tiles::Slices<Width, Head>, m![{ Q }, OutHalf, { Symbol<Row<Head>> }]>()
        };
        device.main.begin_interleaved::<OutHalf, _, _, _, _, _>(halves.clone(), halves)
            .fetch::<m![{ Q }, OutHalf, Half, { Symbol<Row<Head>> } / 16], m![{ Symbol<Row<Head>> } % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Q }, OutHalf, Half, { Symbol<Row<Head>> } / 8], m![{ Symbol<Row<Head>> } % 8]>()
            .vector_init()
            .vector_intra_slice_unzip::<Half, m![{ Q }, OutHalf, { Symbol<Row<Head>> } / 8]>()
            .vector_narrow_split::<m![{ Q }, OutHalf, { Symbol<Row<Head>> } / 4], m![{ Symbol<Row<Head>> } % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), low, ())
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), (), high)
            .vector_fp_zip(FpBinaryOp::AddF)
            .vector_widen_concat::<m![{ Q }, OutHalf, { Symbol<Row<Head>> } / 8], m![{ Symbol<Row<Head>> } % 8]>()
            .vector_final()
            .cast::<bf16, m![{ Symbol<Row<Head>> } % 8 # 16]>()
            .commit_trim::<m![{ Symbol<Row<Head>> } % 8]>()
            .commit_view(rotated);
    }
}

impl<Head: AxisName, Width: AxisName, R: AxisName, Chips: M, Q: M>
    tiles::Pass<Width, Head, Chips, Q> for Rotate<'_, Head, R, Chips>
{
    fn apply(
        &self,
        device: &mut Device,
        rows: &tiles::Tile<bf16, Width, Head, Chips, Q>,
        at: usize,
        output: &mut tiles::Tile<bf16, Width, Head, Chips, Q>,
    ) {
        let coefficients = FetchTiles::<Width, Head, R, Chips>(core::marker::PhantomData)
            .forward((device, (self.rotation, at)));
        RotateTile::<Head>(core::marker::PhantomData)
            .forward((device, (rows, &coefficients, output)));
    }
}

impl<Head: AxisName, Width: AxisName, R: AxisName, Chips: M, Q: M>
    tiles::Both<Width, Head, Chips, Q> for Rotate<'_, Head, R, Chips>
{
    fn apply(
        &self,
        device: &mut Device,
        (first, second): (
            &tiles::Tile<bf16, Width, Head, Chips, Q>,
            &tiles::Tile<bf16, Width, Head, Chips, Q>,
        ),
        at: usize,
        (first_out, second_out): (
            &mut tiles::Tile<bf16, Width, Head, Chips, Q>,
            &mut tiles::Tile<bf16, Width, Head, Chips, Q>,
        ),
    ) {
        let coefficients = FetchTiles::<Width, Head, R, Chips>(core::marker::PhantomData)
            .forward((device, (self.rotation, at)));
        RotateTile::<Head>(core::marker::PhantomData)
            .forward((device, (first, &coefficients, first_out)));
        RotateTile::<Head>(core::marker::PhantomData)
            .forward((device, (second, &coefficients, second_out)));
    }
}

impl<Head: AxisName, Rotated: AxisName, D: Scalar, Chips: M> Default
    for Rotary<Head, Rotated, D, Chips>
{
    fn default() -> Self {
        Self::new()
    }
}

/// Each of `Rows` rows' rotation over the rotated dimensions: `[cos | sin]` for the first input
/// half and `[-sin | cos]` for the second.
#[derive(DeviceSend)]
pub struct Rotation<
    Rotated: AxisName,
    Chips: M = Chip,
    Rows: AxisName = crate::ops::layout::axes::Single,
> {
    pub low: CoefficientRows<Rows, Rotated, Chips>,
    pub high: CoefficientRows<Rows, Rotated, Chips>,
}

impl<Head: AxisName, Rotated: AxisName, Q: M, K: M, D: MaterializableScalar, Chip: M>
    crate::Module<(
        &mut Device,
        (
            (
                HbmTensor<bf16, Chip, m![{ Q }, Head]>,
                HbmTensor<bf16, Chip, m![{ K }, Head]>,
            ),
            &Rotation<Rotated, Chip>,
        ),
    )> for Rotary<Head, Rotated, D, Chip>
where
    f32: CastEngineCast<D>,
    Convert<D>: DeviceModule<HeadTile<K, Head, bf16, Chip>, Output = HeadTile<K, Head, D, Chip>>,
{
    type Output = (
        HbmTensor<bf16, Chip, m![{ Q }, Head]>,
        HbmTensor<D, Chip, m![{ K }, Head]>,
    );

    fn forward(
        &self,
        (device, ((query, key), rotation)): (
            &mut Device,
            (
                (
                    HbmTensor<bf16, Chip, m![{ Q }, Head]>,
                    HbmTensor<bf16, Chip, m![{ K }, Head]>,
                ),
                &Rotation<Rotated, Chip>,
            ),
        ),
    ) -> Self::Output {
        (
            self.rotate::<Q, bf16>(device, query, rotation),
            self.rotate::<K, D>(device, key, rotation),
        )
    }
}

impl<Head: AxisName, Rotated: AxisName, D: Scalar, Chip: M> Rotary<Head, Rotated, D, Chip> {
    fn rotate<Heads: M, Out: MaterializableScalar>(
        &self,
        device: &mut Device,
        input: HbmTensor<bf16, Chip, m![{ Heads }, Head]>,
        rotation: &Rotation<Rotated, Chip>,
    ) -> HbmTensor<Out, Chip, m![{ Heads }, Head]>
    where
        f32: CastEngineCast<Out>,
        Convert<Out>: DeviceModule<
                HeadTile<Heads, Head, bf16, Chip>,
                Output = HeadTile<Heads, Head, Out, Chip>,
            >,
    {
        const {
            assert!(
                Rotated::SIZE > 0
                    && Rotated::SIZE <= Head::SIZE
                    && Rotated::SIZE % 32 == 0
                    && Head::SIZE % Rotated::SIZE == 0
            );
        }
        type HeadSlices<Heads> = m![{Heads} # 256];
        type Row<R> = PairWidth<R>;
        let heads: DmTensor<bf16, Chip, Cluster, HeadSlices<Heads>, m![Head]> =
            input.to_dm(&mut device.tdma);
        let heads: DmTensor<bf16, Chip, Cluster, HeadSlices<Heads>, m![Half, {Symbol<Row<Head>>}]> =
            unsafe { heads.reshape() };
        let mut prefix: DmTensor<
            bf16,
            Chip,
            Cluster,
            HeadSlices<Heads>,
            m![Half, {Symbol<Row<Head>>} = (Rotated / 2)],
        > = DmTensor::new();
        heads.view().tile::<
            m![{Symbol<Row<Head>>} / (Rotated / 2)], 1,
            m![Half, {Symbol<Row<Head>>} / (Rotated / 2) = 1 # (Head / Rotated), {Symbol<Row<Head>>} % (Rotated / 2)]
        >(0).to_dm_view(&mut device.tdma, prefix.view_mut());
        let prefix: DmTensor<
            bf16,
            Chip,
            Cluster,
            HeadSlices<Heads>,
            m![1, 1, Half, {Symbol<Row<Rotated>>}],
        > = unsafe { prefix.reshape() };
        let mut output: DmTensor<
            Out,
            Chip,
            Cluster,
            HeadSlices<Heads>,
            m![1, 1, OutHalf, {Symbol<Row<Rotated>>}],
        > = DmTensor::new();
        // SAFETY: a one-row rotation has the same contiguous order as the rotation vector.
        let (low, high) = unsafe {
            (
                rotation.low.view().reshape::<Chip, m![Rotated]>(),
                rotation.high.view().reshape::<Chip, m![Rotated]>(),
            )
        };
        let low: DmTensor<bf16, Chip, Cluster, m![256], m![Rotated]> = low.to_dm(&mut device.tdma);
        let high: DmTensor<bf16, Chip, Cluster, m![256], m![Rotated]> =
            high.to_dm(&mut device.tdma);
        let low: Table<HeadSlices<Heads>, m![1], Row<Rotated>, Chip> = unsafe { low.reshape() };
        let high: Table<HeadSlices<Heads>, m![1], Row<Rotated>, Chip> = unsafe { high.reshape() };
        let mut factors: DmTensor<
            bf16,
            Chip,
            Cluster,
            HeadSlices<Heads>,
            m![1, 1, OutHalf, Half, {Symbol<Row<Rotated>>}],
        > = DmTensor::new();
        low.view().to_dm_view(
            &mut device.tdma,
            factors
                .view_mut()
                .tile::<m![Half], 1, m![1, 1, OutHalf, Half = 1 #{!} 2, {Symbol<Row<Rotated>>}]>(0),
        );
        high.view().to_dm_view(
            &mut device.tdma,
            factors
                .view_mut()
                .tile::<m![Half], 1, m![1, 1, OutHalf, Half = 1 #{!} 2, {Symbol<Row<Rotated>>}]>(1),
        );
        let factors: VrfTensor<f32, Chip, Cluster, HeadSlices<Heads>, m![1, 1, OutHalf, Half, {Symbol<Row<Rotated>>}]> = device.sub.begin(factors.view())
            .fetch::<m![OutHalf, Half, {Symbol<Row<Rotated>>} / 16], m![{Symbol<Row<Rotated>>} % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![OutHalf, Half, {Symbol<Row<Rotated>>} / 8], m![{Symbol<Row<Rotated>>} % 8]>()
            .to_vrf();
        let mut values: DmTensor<
            bf16,
            Chip,
            Cluster,
            HeadSlices<Heads>,
            m![1, 1, OutHalf, Half, {Symbol<Row<Rotated>>}],
        > = DmTensor::new();
        prefix
            .view()
            .to_dm_view(&mut device.tdma, values.view_mut());
        let products: DmTensor<bf16, Chip, Cluster, HeadSlices<Heads>, m![1, 1, OutHalf, Half, {Symbol<Row<Rotated>>}]> = device
            .main
            .begin(values.view())
            .fetch::<m![OutHalf, Half, {Symbol<Row<Rotated>>} / 16], m![{Symbol<Row<Rotated>>} % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![OutHalf, Half, {Symbol<Row<Rotated>>} / 8], m![{Symbol<Row<Rotated>>} % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![OutHalf, Half, {Symbol<Row<Rotated>>} / 4], m![{Symbol<Row<Rotated>>} % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &factors)
            .vector_widen_concat::<m![OutHalf, Half, {Symbol<Row<Rotated>>} / 8], m![{Symbol<Row<Rotated>>} % 8]>()
            .vector_final()
            .cast::<bf16, m![{Symbol<Row<Rotated>>} % 8 # 16]>()
            .commit_trim::<m![{Symbol<Row<Rotated>>} % 8]>()
            .commit();
        device.main.begin(products.view())
            .fetch::<m![OutHalf, Half, {Symbol<Row<Rotated>>} / 16], m![{Symbol<Row<Rotated>>} % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![OutHalf, Half, {Symbol<Row<Rotated>>} / 8], m![{Symbol<Row<Rotated>>} % 8]>()
            .vector_init()
            .vector_zip::<
                Half,
                m![OutHalf, {Symbol<Row<Rotated>>} / 8],
                m![OutHalf, {Symbol<Row<Rotated>>} / 4],
                m![{Symbol<Row<Rotated>>} % 4],
            >(FpBinaryOp::AddF)
            .vector_final()
            .cast::<Out, m![{Symbol<Row<Rotated>>} % 8 # ({Symbol<Packet<Out>>})]>()
            .commit_trim::<m![{Symbol<Row<Rotated>>} % 8]>()
            .commit_view(output.view_mut());
        let output: DmTensor<
            Out,
            Chip,
            Cluster,
            HeadSlices<Heads>,
            m![Half, {Symbol<Row<Head>>} = (Rotated / 2)],
        > = unsafe { output.reshape() };
        // SAFETY: join the existing consecutive halves without changing their scalar type.
        let heads: HeadTile<Heads, Head, bf16, Chip> = unsafe { heads.reshape() };
        let heads = crate::Module::forward(&Convert::<Out>::default(), (device, heads));
        // SAFETY: recover the same consecutive halves after element conversion.
        let mut heads: DmTensor<
            Out,
            Chip,
            Cluster,
            HeadSlices<Heads>,
            m![Half, {Symbol<Row<Head>>}],
        > = unsafe { heads.reshape() };
        output.view().to_dm_view(&mut device.tdma, heads.view_mut().tile::<
            m![{Symbol<Row<Head>>} / (Rotated / 2)], 1,
            m![Half, {Symbol<Row<Head>>} / (Rotated / 2) = 1 #{!} (Head / Rotated), {Symbol<Row<Head>>} % (Rotated / 2)]
        >(0));
        let heads: DmTensor<Out, Chip, Cluster, HeadSlices<Heads>, m![Head]> =
            unsafe { heads.reshape() };
        heads.to_hbm(&mut device.tdma)
    }
}
