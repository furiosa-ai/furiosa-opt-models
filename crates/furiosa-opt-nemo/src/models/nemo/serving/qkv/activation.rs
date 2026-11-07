use furiosa_opt_std::prelude::*;

use super::contraction::{Activation, Cluster, Out, Slice};
use crate::compose::{Tile, Tiles};
use crate::models::nemo::serving::{Hidden, Rows, scale};
use crate::ops::{layout::Dm, normalization::RmsNorm};

type Time = m![Rows, Hidden % 80 / 8];
type Packet = m![Hidden % 8];

pub fn prepare<B: M>(
    device: &mut Device,
    input: &DmTensor<bf16, B, Cluster, Slice, m![Rows, Hidden % 80]>,
    parameters: (
        &RmsNorm<
            Hidden,
            { 1e-5_f32.to_bits() as usize },
            B,
            DmTensor<bf16, B, Cluster, Slice, m![Hidden % 80]>,
        >,
        &scale::InputScale<B, Cluster, Slice>,
    ),
) -> Activation<B> {
    let input = Dm::<Tile<Rows, Rows>, _> {
        tiles: Tiles::over(Rows::SIZE),
        value: input.view(),
    };
    let norm =
        RmsNorm::<Hidden, { 1e-5_f32.to_bits() as usize }, B, _>::new(parameters.0.weight().view());
    let gain = norm.gain::<m![Hidden % 80 / 8], Packet>(device);
    norm.replicated::<_, m![Rows], m![Hidden % 80], m![Out % 512 / 128, 64], m![1 # 2], false, _>(
        device, &input,
    )
    .inverse(device)
    .apply::<Time, Packet, m![Rows, Hidden % 80 / 4]>(device, &gain)
    .value
    .vector_fp_div(parameters.1.input_scale())
    .vector_widen_concat::<Time, Packet>()
    .vector_final()
    .cast::<f8e4m3, m![Hidden % 8 # 32]>()
    .commit_trim::<Packet>()
    .commit()
}
