use furiosa_opt_std::prelude::*;

/// The two inputs of an interleaved add.
#[derive(Debug, Clone)]
struct Term;

impl AxisName for Term {
    const NAME: Ident = Ident::new("Term");
    const SIZE: usize = 2;
}

/// Add the first `rows` entries of `Rows`, `Time`'s outermost axis, into `output`.
pub(crate) fn add_live<
    Rows: M,
    Time: M,
    Run: M,
    Packet: M,
    Chip: M,
    Cluster: M,
    Slice: M,
    Element: M,
    Output: M,
>(
    context: &mut Device,
    inputs: (
        &DmTensor<bf16, Chip, Cluster, Slice, Element>,
        &DmTensor<bf16, Chip, Cluster, Slice, Element>,
    ),
    (output, rows): (
        DmTensorViewMut<'_, bf16, Chip, Cluster, Slice, Output>,
        usize,
    ),
) {
    context
        .main
        .begin_interleaved::<Term, _, _, _, _, _>(inputs.0.view(), inputs.1.view())
        .fetch::<m![{ Time }, Term, { Run }], Packet>()
        .fetch_cast::<f32>()
        .collect::<m![{ Time }, Term, { Run }], Packet>()
        .vector_init()
        .vector_intra_slice_unzip::<Term, m![{ Time }, { Run }]>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .cast::<bf16, m![{ Packet } # 16]>()
        .commit_trim::<Packet>()
        .commit_prefix::<Rows, _, _>(output, rows);
}
