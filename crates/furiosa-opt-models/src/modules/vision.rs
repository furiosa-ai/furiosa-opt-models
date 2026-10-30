//! Patch embeddings and learned-query pooling shared by vision encoders, and the positions,
//! padding and masks of an image cut into tiles; tile validity is caller metadata.
use crate::Module;
use crate::ops::index::sequence::Sum;
use crate::tensor::axis::Single;
use crate::tensor::image::{self, Grid, Image};
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

/// Projects patches, prepends a prefix, adds position values and normalizes the sequence.
#[derive(DeviceSend)]
pub struct Embedding<Project, Prefix, Position, Norm, Add> {
    /// Projection from input patches to hidden features.
    pub project: Project,
    /// Adds the learned prefix to the patch sequence.
    pub prefix: Prefix,
    /// Position values with the same type as the prefixed hidden sequence.
    pub position: Position,
    /// Normalization after adding position values.
    pub norm: Norm,
    /// Combines hidden and position values.
    pub add: Add,
}

/// A composition, not an op.
impl<I, Project, Prefix, Position, Norm, Add> !crate::Op<I>
    for Embedding<Project, Prefix, Position, Norm, Add>
{
}

/// The state is the norm's slot.
impl<I, X, Y, S, Project, Prefix, Position, Norm, Add, Env: Copy> Module<I, Env, S>
    for Embedding<Project, Prefix, Position, Norm, Add>
where
    Project: for<'s> Module<I, Env, &'s mut (), Output = X>,
    Prefix: for<'s> Module<X, Env, &'s mut (), Output = Position>,
    Add: for<'a, 'b, 's> Module<(&'a Position, &'b Position), Env, &'s mut (), Output = Position>,
    Norm: Module<Position, Env, S, Output = Y>,
{
    type Output = Y;

    fn forward(&self, device: &mut Device, env: Env, state: S, input: I) -> Y {
        let patches = self.project.forward(&mut *device, env, &mut (), input);
        let hidden = self.prefix.forward(&mut *device, env, &mut (), patches);
        let hidden = self
            .add
            .forward(&mut *device, env, &mut (), (&hidden, &self.position));
        self.norm.forward(device, env, state, hidden)
    }
}

/// Prepends a stored prefix to an input sequence along the concatenation axis.
#[derive(DeviceSend)]
pub struct Prepend<Prefix, Concat> {
    /// Learned values placed before the input sequence.
    pub prefix: Prefix,
    /// Concatenation with prefix as the first operand.
    pub concat: Concat,
}

/// A composition, not an op.
impl<X, Prefix, Concat> !crate::Op<X> for Prepend<Prefix, Concat> {}

impl<X, Prefix, Z, Concat, Env: Copy> Module<X, Env, &mut ()> for Prepend<Prefix, Concat>
where
    Concat: for<'a, 'b, 's> Module<(&'a Prefix, &'b X), Env, &'s mut (), Output = Z>,
{
    type Output = Z;

    fn forward(&self, device: &mut Device, env: Env, _: &mut (), input: X) -> Z {
        self.concat
            .forward(device, env, &mut (), (&self.prefix, &input))
    }
}

/// Pools hidden features with learned queries, then applies a residual transformation.
#[derive(DeviceSend)]
pub struct Pool<Probe, Query, Key, Value, Attend, Project, Residual> {
    /// Learned input to the query projection.
    pub probe: Probe,
    /// Projects the learned probe into attention queries.
    pub query: Query,
    /// Projects hidden features into attention keys.
    pub key: Key,
    /// Projects hidden features into attention values.
    pub value: Value,
    /// Attention from learned queries to the hidden features.
    pub attend: Attend,
    /// Output projection of the attended values.
    pub project: Project,
    /// Residual transformation of the pooled output.
    pub residual: Residual,
}

/// A composition, not an op.
impl<X, Probe, Query, Key, Value, Attend, Project, Residual> !crate::Op<&X>
    for Pool<Probe, Query, Key, Value, Attend, Project, Residual>
{
}

/// The state is the residual transformation's slot.
impl<X, Probe, Q, K, V, A, Y, O, S, Query, Key, Value, Attend, Project, Residual, Env: Copy>
    Module<&X, Env, S> for Pool<Probe, Query, Key, Value, Attend, Project, Residual>
where
    Query: for<'x, 's> Module<&'x Probe, Env, &'s mut (), Output = Q>,
    Key: for<'x, 's> Module<&'x X, Env, &'s mut (), Output = K>,
    Value: for<'x, 's> Module<&'x X, Env, &'s mut (), Output = V>,
    Attend: for<'a, 'b, 'c, 's> Module<(&'a Q, &'b K, &'c V), Env, &'s mut (), Output = A>,
    Project: for<'x, 's> Module<&'x A, Env, &'s mut (), Output = Y>,
    Residual: Module<Y, Env, S, Output = O>,
{
    type Output = O;

    fn forward(&self, device: &mut Device, env: Env, state: S, hidden: &X) -> O {
        let query = self.query.forward(&mut *device, env, &mut (), &self.probe);
        let key = self.key.forward(&mut *device, env, &mut (), hidden);
        let value = self.value.forward(&mut *device, env, &mut (), hidden);
        let attended = self
            .attend
            .forward(&mut *device, env, &mut (), (&query, &key, &value));
        let pooled = self.project.forward(&mut *device, env, &mut (), &attended);
        self.residual.forward(device, env, state, pooled)
    }
}

type Gate = HbmTensor<f32, m![1], m![1]>;
type Id = HbmTensor<i32, m![1], m![1]>;

/// Single-chip image features indexed by tile, row and feature.
pub type Features<D, T, R, W> = HbmTensor<D, m![1], m![T, R, W]>;

/// Flattened tile and row coordinates.
#[derive(Clone, Debug)]
pub struct Rows<T, R>(PhantomData<(T, R)>);

impl<T: AxisName, R: AxisName> AxisName for Rows<T, R> {
    const NAME: Ident = Ident::new("ImageRows");
    const SIZE: usize = T::SIZE * R::SIZE;
}

/// Final features followed by N captured layers, feature-major within the captures.
#[derive(Clone, Debug)]
pub struct Captured<W, const N: usize>(PhantomData<W>);

impl<W: AxisName, const N: usize> AxisName for Captured<W, N> {
    const NAME: Ident = Ident::new("CapturedFeatures");
    const SIZE: usize = W::SIZE * (N + 1);
}

impl<
    D: Scalar,
    T: AxisName,
    Channels: AxisName,
    H: AxisName,
    W: AxisName,
    Height: AxisName,
    Columns: AxisName,
    Width: AxisName,
    Bias,
    Env,
> Module<&HbmTensor<D, m![1], m![T, Channels, H, W]>, Env, &mut ()>
    for Patch<Channels, Width, Height, Columns, D, Bias>
{
    type Output = Features<D, T, Grid<H, W, Height, Columns>, Width>;

    fn forward(
        &self,
        _: &mut Device,
        _: Env,
        _: &mut (),
        _: &HbmTensor<D, m![1], m![T, Channels, H, W]>,
    ) -> Self::Output {
        todo!("NPU independent nonoverlapping patch projection for each image tile")
    }
}

/// x + tanh(gate) * embedding[aspect_id, tile, feature], broadcast over rows.
#[derive(DeviceSend)]
pub struct Positions<D: Scalar, A: AxisName, T: AxisName, W: AxisName> {
    /// Aspect-ratio table with flattened tile and feature coordinates.
    pub embedding: HbmTensor<D, m![1], m![A, {Symbol<Rows<T,W>>}]>,
    /// Scalar controlling the table contribution through tanh.
    pub gate: Gate,
}

impl<D: Scalar, A: AxisName, T: AxisName, W: AxisName, R: AxisName, Env>
    Module<(&Features<D, T, R, W>, &Id), Env, &mut ()> for Positions<D, A, T, W>
{
    type Output = Features<D, T, R, W>;

    fn forward(&self, _: &mut Device, _: Env, _: &mut (), _: (&Self::Output, &Id)) -> Self::Output {
        todo!()
    }
}

/// `x + (1 - tanh(gate)) * embedding + tanh(gate) * tiles[aspect_id]`.
#[derive(DeviceSend)]
pub struct GatedPosition<D: Scalar, A: AxisName, T: AxisName, R: AxisName, W: AxisName> {
    /// Shared row positions, broadcast over tiles.
    pub embedding: HbmTensor<D, m![1], m![R, W]>,
    /// Aspect-ratio-specific positions with flattened tile, row and feature coordinates.
    pub tiles: HbmTensor<D, m![1], m![A, {Symbol<Rows<T,Rows<R,W>>>}]>,
    /// Scalar blending the shared and aspect-ratio-specific positions through tanh.
    pub gate: Gate,
}

impl<D: Scalar, A: AxisName, T: AxisName, R: AxisName, W: AxisName, Env>
    Module<(&Features<D, T, R, W>, &Id), Env, &mut ()> for GatedPosition<D, A, T, R, W>
{
    type Output = Features<D, T, R, W>;

    fn forward(&self, _: &mut Device, _: Env, _: &mut (), _: (&Self::Output, &Id)) -> Self::Output {
        todo!()
    }
}

/// Prepend the learned class row independently to each tile.
#[derive(DeviceSend)]
pub struct Class<D: Scalar, W: AxisName> {
    /// Class features shared by all tiles.
    pub embedding: HbmTensor<D, m![1], m![W]>,
}

impl<D: Scalar, T: AxisName, R: AxisName, W: AxisName, Env>
    Module<&Features<D, T, R, W>, Env, &mut ()> for Class<D, W>
{
    type Output = Features<D, T, Sum<Single, R>, W>;

    fn forward(
        &self,
        _: &mut Device,
        _: Env,
        _: &mut (),
        _: &Features<D, T, R, W>,
    ) -> Self::Output {
        todo!()
    }
}

/// Zero-pad each tile's rows to Output, preserving the input rows.
#[derive(DeviceSend)]
pub struct Pad<Output>(pub PhantomData<Output>);

impl<D: Scalar, T: AxisName, R: AxisName, W: AxisName, Output: AxisName, Env>
    Module<&Features<D, T, R, W>, Env, &mut ()> for Pad<Output>
{
    type Output = Features<D, T, Output, W>;

    fn forward(
        &self,
        _: &mut Device,
        _: Env,
        _: &mut (),
        _: &Features<D, T, R, W>,
    ) -> Self::Output {
        const {
            assert!(Output::SIZE >= R::SIZE);
        }
        todo!()
    }
}

/// Remove padded rows; columns are final features then feature-major, layer-minor captures.
#[derive(DeviceSend)]
pub struct Merge<Tiles, Live, Padded>(pub PhantomData<(Tiles, Live, Padded)>);

impl<D: Scalar, Tiles: AxisName, Padded: AxisName, W: AxisName, Live: AxisName, const N: usize, Env>
    Module<
        (
            &HbmTensor<D, m![1], m![{Symbol<Rows<Tiles,Padded>>}, W]>,
            &[HbmTensor<D, m![1], m![{Symbol<Rows<Tiles,Padded>>}, W]>; N],
        ),
        Env,
        &mut (),
    > for Merge<Tiles, Live, Padded>
{
    type Output = HbmTensor<D, m![1], m![{Symbol<Rows<Tiles,Live>>}, {Symbol<Captured<W,N>>}]>;

    fn forward(
        &self,
        _: &mut Device,
        _: Env,
        _: &mut (),
        _: (
            &HbmTensor<D, m![1], m![{Symbol<Rows<Tiles,Padded>>}, W]>,
            &[HbmTensor<D, m![1], m![{Symbol<Rows<Tiles,Padded>>}, W]>; N],
        ),
    ) -> Self::Output {
        const {
            assert!(Live::SIZE <= Padded::SIZE);
        }
        todo!()
    }
}

/// Attention bias is zero for visible pairs and BF16 minimum when both endpoints are masked.
/// Padded rows and rows of invalid tiles are masked.
#[derive(DeviceSend)]
pub struct Mask<Tiles, Live, Padded>(pub PhantomData<(Tiles, Live, Padded)>);

impl<Tiles: AxisName, Live: AxisName, Padded: AxisName, Env>
    Module<&HbmTensor<i32, m![1], m![Tiles]>, Env, &mut ()> for Mask<Tiles, Live, Padded>
{
    type Output =
        HbmTensor<f32, m![1], m![{Symbol<Rows<Tiles,Padded>>},{Symbol<Rows<Tiles,Padded>>}]>;

    fn forward(
        &self,
        _: &mut Device,
        _: Env,
        _: &mut (),
        _: &HbmTensor<i32, m![1], m![Tiles]>,
    ) -> Self::Output {
        const {
            assert!(Live::SIZE <= Padded::SIZE);
        }
        todo!()
    }
}

/// Project nonoverlapping image patches into feature rows in row-major grid order.
#[derive(DeviceSend)]
pub struct Patch<
    Channels: AxisName,
    Width: AxisName,
    Height: AxisName,
    Columns: AxisName,
    D: Scalar = f32,
    Bias = HbmTensor<D, m![1], m![Width]>,
> {
    /// Projection coefficients ordered by output feature, channel, patch row and column.
    pub weight: HbmTensor<D, m![1], m![Width, Channels, Height, Columns]>,
    /// The additive bias per output feature, or `()` for none.
    pub bias: Bias,
}

impl<
    Channels: AxisName,
    Width: AxisName,
    Height: AxisName,
    Columns: AxisName,
    H: AxisName,
    W: AxisName,
    Bias,
    Env,
> Module<&Image<Channels, H, W>, Env, &mut ()>
    for Patch<Channels, Width, Height, Columns, f32, Bias>
{
    type Output = image::Rows<Width, Grid<H, W, Height, Columns>>;

    fn forward(
        &self,
        _: &mut Device,
        _: Env,
        _: &mut (),
        _: &Image<Channels, H, W>,
    ) -> Self::Output {
        todo!(
            "NPU nonoverlapping image patch projection with optional bias in row-major grid order"
        )
    }
}

/// Two-dimensional sine/cosine patch positions, with half the features per coordinate.
#[derive(DeviceSend)]
pub struct SinCos<Width, PatchHeight, PatchWidth>(
    pub PhantomData<(Width, PatchHeight, PatchWidth)>,
);

impl<Width, PatchHeight, PatchWidth> Default for SinCos<Width, PatchHeight, PatchWidth> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<
    C: AxisName,
    Width: AxisName,
    PatchHeight: AxisName,
    PatchWidth: AxisName,
    H: AxisName,
    W: AxisName,
    Env,
> Module<&Image<C, H, W>, Env, &mut ()> for SinCos<Width, PatchHeight, PatchWidth>
{
    type Output = image::Rows<Width, Grid<H, W, PatchHeight, PatchWidth>>;

    fn forward(&self, _: &mut Device, _: Env, _: &mut (), _: &Image<C, H, W>) -> Self::Output {
        const { assert!(Width::SIZE % 4 == 0) };
        todo!("NPU two-dimensional sine/cosine positions with half the channels per coordinate")
    }
}
