use furiosa_opt_std::prelude::*;

use crate::Module;
use crate::ops::layout::axes::Chip;

pub(crate) mod kernel;
use kernel as norm;

pub(crate) use norm::{Owned, Tiles};
pub(crate) type Scale<Width, Rows> =
    DmTensor<bf16, Chip, m![Rows / (Rows / 2)], m![Width / 16], m![Width % 16]>;

pub(crate) struct LiveRows<
    'a,
    Width: AxisName,
    Rows: AxisName,
    const CAPACITY: usize,
    const LAST: usize,
> {
    pub(crate) value: &'a Owned<Width, Rows>,
    pub(crate) rows: usize,
}

impl<
    Width: AxisName,
    Rows: AxisName,
    const CAPACITY: usize,
    const LAST: usize,
    const EPSILON: usize,
> Module<(&mut Device, LiveRows<'_, Width, Rows, CAPACITY, LAST>)>
    for super::RmsNorm<Width, EPSILON, Chip, Scale<Width, Rows>>
{
    type Output = Owned<Width, Rows>;

    /// The first `rows` rows of each cluster's half, capped at the half.
    fn forward(
        &self,
        (context, input): (&mut Device, LiveRows<'_, Width, Rows, CAPACITY, LAST>),
    ) -> Self::Output {
        let (half, rows) = (Rows::SIZE / 2, input.rows);
        let reach = rows + rows / (half + 1) * half - rows / (half + 1) * rows;
        norm::compute_live::<
            Width,
            m![Rows / (Rows / 2)],
            m![Width / 16],
            m![256],
            m![Rows % (Rows / 2)],
            m![Width % 16],
            m![Width / 16],
            Tiles<CAPACITY, LAST, EPSILON>,
        >(context, (input.value, &self.weight), reach)
    }
}

pub(crate) struct Wave<'i, 'o, Width: AxisName, Rows: M> {
    pub(crate) input: DmTensorView<
        'i,
        bf16,
        Chip,
        super::Cluster,
        super::FeatureSlices<Width>,
        m![{ Rows }, Width % 16],
    >,
    pub(crate) output: DmTensorViewMut<
        'o,
        bf16,
        Chip,
        super::Cluster,
        super::FeatureSlices<Width>,
        m![{ Rows }, Width % 16],
    >,
}

impl<Width: AxisName, Rows: M, const EPSILON: usize, G>
    Module<(&mut Device, Wave<'_, '_, Width, Rows>)> for super::RmsNorm<Width, EPSILON, Chip, G>
where
    G: super::Learned<Width, Chip>,
{
    type Output = ();

    fn forward(&self, (context, input): (&mut Device, Wave<'_, '_, Width, Rows>)) {
        super::kernel::rms_norm::<Width, Rows, EPSILON, Chip, _>(
            context,
            input.input,
            &self.weight,
        )
        .commit_view(input.output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::ops::normalization::{EPSILON, RmsNorm};

    axes![Width = 4096];

    #[test]
    fn accepts_wave_views_and_landed_gain() {
        fn accepts<N>()
        where
            N: for<'d, 'i, 'o> Module<(&'d mut Device, Wave<'i, 'o, Width, m![128]>), Output = ()>,
        {
        }
        accepts::<RmsNorm<Width, EPSILON, Chip, super::super::Landed<Width>>>();
    }
}
