//! Complete restored QKV operands and norm gain, hosted by the preceding MLP's down panels.
use super::*;
use crate::ops::linear::nvfp4::tiled::prefetch::{Issued, Points};
use crate::ops::linear::nvfp4::tiled::qkv::{self, Weight};

pub(super) type Norm = RmsNorm<Hidden, EPSILON, Chip, span::Scale<Hidden, Span>>;

pub(super) struct Projection {
    pub query: Weight<Hidden, QWidth>,
    pub key: Weight<Hidden, KvWidth>,
    pub value: Weight<Hidden, KvWidth>,
}

pub(super) struct Head<'a> {
    pub norm: &'a super::Norm,
    pub qkv: &'a Qkv,
    pub enabled: bool,
}

pub(super) struct First<'a> {
    qkv: &'a Qkv,
    enabled: bool,
    norm: Norm,
    value: Weight<Hidden, KvWidth>,
}

impl Head<'_> {
    pub(super) fn load(self, device: &mut Device) -> (Norm, Projection) {
        self.finish(device)
    }
}

impl<'a> Points<Device> for Head<'a> {
    type Next = First<'a>;
    type Output = (Norm, Projection);

    fn at(self, device: &mut Device) -> Self::Next {
        // No speculative loads on the final layer. Empty buffers are never consumed there.
        let (value, norm) = if self.enabled {
            (
                qkv::prepare(device, &self.qkv.value.0),
                crate::Op::forward(&crate::ops::normalization::span::Stage, device, self.norm),
            )
        } else {
            (DmTensor::new(), RmsNorm::new(DmTensor::new()))
        };
        First {
            qkv: self.qkv,
            enabled: self.enabled,
            norm,
            value,
        }
    }

    fn finish(self, device: &mut Device) -> Self::Output {
        self.at(device).finish(device)
    }
}

impl Points<Device> for First<'_> {
    type Next = Issued<(Norm, Projection)>;
    type Output = (Norm, Projection);

    fn at(self, device: &mut Device) -> Self::Next {
        let (key, query) = if self.enabled {
            // Queue both complete packed operands before either restoration stream.
            // Their storage stays at this boundary, rather than spanning all down panels.
            let key = qkv::stage(device, &self.qkv.key.0);
            let query = qkv::stage(device, &self.qkv.query.0);
            (qkv::restore(device, key), qkv::restore(device, query))
        } else {
            (DmTensor::new(), DmTensor::new())
        };
        Issued((
            self.norm,
            Projection {
                query,
                key,
                value: self.value,
            },
        ))
    }

    fn finish(self, device: &mut Device) -> Self::Output {
        self.at(device).finish(device)
    }
}
