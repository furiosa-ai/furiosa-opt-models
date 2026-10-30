use super::*;
use crate::Op;
use crate::ops::vector::Quantization;

/// A product panel's first `rows` rows as E4M3 rows and their BF16 row scales.
pub(super) struct QuantizeProduct;

/// The quantization of a product panel's rows.
type Panel<Hidden, Intermediate, Rows> = Quantization<
    m![Rows, Intermediate % 64 / 8],
    m![Rows, Intermediate % 64 / 4],
    Intermediate,
    m![Rows],
    m![Hidden / (Hidden / 8), Copies],
    m![Rows / 4],
    m![Rows % 4],
>;

/// A product panel and the quantized rows and scales its quantization writes.
type Quantized<'a, Hidden, Intermediate, Rows> = (
    &'a Product<Hidden, Intermediate, Rows>,
    usize,
    (
        &'a mut Activation<Hidden, Intermediate, Rows>,
        &'a mut Scale<Hidden, Intermediate, Rows>,
    ),
);

impl<'a, Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>
    Op<Quantized<'a, Hidden, Intermediate, Rows>> for QuantizeProduct
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (product, rows, (quantized, scales)): Quantized<'a, Hidden, Intermediate, Rows>,
    ) {
        // SAFETY: both mappings name every slice of the cluster; the reduction writes each one.
        let output = unsafe {
            scales
                .view_mut()
                .reshape::<
                    Chip,
                    Cluster<Intermediate>,
                    m![Hidden / (Hidden / 8), Copies],
                    m![Rows],
                >()
        };
        Panel::<Hidden, Intermediate, Rows>::scales(&mut context.main, product.view())
            .commit_prefix::<m![Rows / 4], _, _>(output, rows.div_ceil(4));
        // SAFETY: every slice of a cluster holds the same scales.
        let broadcast = unsafe {
            scales
                .view()
                .reshape::<Chip, Cluster<Intermediate>, DownSlice<Hidden, Intermediate>, m![Rows]>()
        };
        let scale: VrfTensor<
            f32,
            Chip,
            Cluster<Intermediate>,
            DownSlice<Hidden, Intermediate>,
            m![Rows],
        > = context
            .sub
            .begin(broadcast)
            .fetch::<m![Rows / 8], m![Rows % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Rows / 8], m![Rows % 8]>()
            .to_vrf();
        Panel::<Hidden, Intermediate, Rows>::rows(&mut context.main, product.view(), &scale)
            .commit_prefix::<m![Rows], _, _>(quantized.view_mut(), rows);
    }
}
