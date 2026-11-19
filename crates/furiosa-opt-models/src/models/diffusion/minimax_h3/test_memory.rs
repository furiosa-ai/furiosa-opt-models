use crate::load::{Checkpoint, Context, Load, LoadError};
use crate::tensor::memory::Memory;
use furiosa_opt_std::prelude::*;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::sync::Arc;

pub(super) struct Host;

pub(super) struct Leaf<D, C, S> {
    pub(super) bytes: Arc<[u8]>,
    pub(super) shape: PhantomData<(D, C, S)>,
}

#[derive(Clone)]
pub(super) struct Image {
    pub(super) path: String,
    pub(super) bytes: Vec<u8>,
    pub(super) chips: usize,
    pub(super) elements: usize,
}

thread_local! {
    pub(super) static IMAGES: RefCell<Vec<Image>> = const { RefCell::new(Vec::new()) };
}

impl Memory for Host {
    type Tensor<D: MaterializableScalar, C: M + 'static, S: M + 'static> = Leaf<D, C, S>;
}

impl<D: MaterializableScalar, C: M + 'static, S: M + 'static, E: M> Load<HostTensor<D, E>>
    for Leaf<D, C, S>
{
    fn load(cx: &mut Context<'_>, host: HostTensor<D, E>) -> Result<Self, LoadError> {
        assert_eq!(E::SIZE, C::SIZE * S::SIZE);
        let bytes = D::to_buf(&host.into_vec());
        IMAGES.with_borrow_mut(|images| {
            images.push(Image {
                path: cx.path(),
                // A capture must not add an owner to the leaf's residency storage.
                bytes: bytes.clone(),
                chips: C::SIZE,
                elements: S::SIZE,
            })
        });
        Ok(Self {
            bytes: bytes.into(),
            shape: PhantomData,
        })
    }
}

impl<'a, D: MaterializableScalar, C: M + 'static, S: M + 'static> Load<&'a Checkpoint>
    for Leaf<D, C, S>
{
    fn load(cx: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        cx.checkpoint(from, |cx| {
            let host = from.host::<D, S, m![{ C }, { S }]>(cx)?;
            Self::load(cx, host)
        })
    }
}

impl<'a, D, C, S> Load<&'a Leaf<D, C, S>> for &'a Leaf<D, C, S> {
    fn load(_: &mut Context<'_>, leaf: &'a Leaf<D, C, S>) -> Result<Self, LoadError> {
        Ok(leaf)
    }
}

impl<'a, D: MaterializableScalar, C: M + 'static, S: M + 'static, E: M>
    Load<&'a dyn Fn() -> Result<HostTensor<D, E>, LoadError>> for Leaf<D, C, S>
{
    fn load(
        cx: &mut Context<'_>,
        read: &'a dyn Fn() -> Result<HostTensor<D, E>, LoadError>,
    ) -> Result<Self, LoadError> {
        Self::load(cx, read()?)
    }
}
