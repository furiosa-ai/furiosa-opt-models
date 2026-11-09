use crate::Module;
use crate::ops::indexing::Offset;

use crate::ops::layout::axes::Single;

use crate::ops::vector::scalar;
use crate::ops::weight::Weight;
use furiosa_opt_std::prelude::*;

/// Look up token features by a row byte offset or by packet-padded token IDs.
#[derive(DeviceSend)]
pub struct Embedding<
    Vocab: AxisName,
    Hidden: AxisName,
    Chips: M = m![1],
    Parameter = Weight<m![Vocab, Hidden], Chips>,
> {
    weight: Parameter,
    features: core::marker::PhantomData<(Vocab, Hidden, Chips)>,
}

impl<Vocab: AxisName, Hidden: AxisName, Chips: M, Parameter>
    Embedding<Vocab, Hidden, Chips, Parameter>
{
    pub const fn new(weight: Parameter) -> Self {
        Self {
            weight,
            features: core::marker::PhantomData,
        }
    }

    pub const fn weight(&self) -> &Parameter {
        &self.weight
    }
}

/// Select one embedding row by its byte offset.
impl<Vocab: AxisName, Hidden: AxisName, Chips: M> Module<(&mut Device, &Offset<Chips>)>
    for Embedding<Vocab, Hidden, Chips>
{
    type Output = HbmTensor<bf16, Chips, m![Hidden]>;

    fn forward(&self, (device, offset): (&mut Device, &Offset<Chips>)) -> Self::Output {
        let row: DmTensor<bf16, Chips, m![1 # 2], m![1 # 256], m![Hidden]> = self
            .weight
            .tensor()
            .gather::<m![Vocab], m![Hidden]>()
            .by_byte_offsets(offset)
            .to_dm(&mut device.tdma);
        row.to_hbm(&mut device.tdma)
    }
}

/// Select the row of one token ID, at the byte offset of its row.
impl<Vocab: AxisName, Hidden: AxisName, Chips: M>
    Module<(&mut Device, &HbmTensor<i32, Chips, m![1]>)> for Embedding<Vocab, Hidden, Chips>
{
    type Output = HbmTensor<bf16, Chips, m![Hidden]>;

    fn forward(&self, (device, id): (&mut Device, &HbmTensor<i32, Chips, m![1]>)) -> Self::Output {
        let offset = scalar::offset::<bf16, m![Hidden], m![Single], _>(device, id);
        self.forward((device, &offset))
    }
}
