use super::*;

pub(crate) struct Gather<'a, All: AxisName, S: AxisName, Half: AxisName> {
    pub(crate) positions: &'a HbmTensor<i32, Chip, m![All]>,
    pub(crate) coefficients: &'a mut Rotation<S, Half>,
    pub(crate) live: usize,
}

impl<Head: AxisName, Half: AxisName, Position: AxisName, All: AxisName, S: AxisName>
    Module<(&mut Device, Gather<'_, All, S, Half>)> for SplitHalf<Head, &Table<Position, Half>>
{
    type Output = ();

    fn forward(
        &self,
        (
            context,
            Gather {
                positions,
                coefficients,
                live,
            },
        ): (&mut Device, Gather<'_, All, S, Half>),
    ) {
        const {
            assert!(All::SIZE >= Tile::SIZE);
            assert!(S::SIZE == All::SIZE.div_ceil(Tile::SIZE));
            assert!(Head::SIZE == 2 * Half::SIZE);
        }
        for wave in 0..live {
            self.gather(context, positions, (coefficients, wave, wave * Tile::SIZE));
        }
    }
}

impl<Head: AxisName, Half: AxisName, Position: AxisName> SplitHalf<Head, &Table<Position, Half>> {
    pub(crate) fn gather<All: AxisName, S: AxisName>(
        &self,
        context: &mut Device,
        positions: &HbmTensor<i32, Chip, m![All]>,
        output: (&mut Rotation<S, Half>, usize, usize),
    ) {
        type Halves = m![Tile / (Tile / 2)];
        type Copies = m![Tile % (Tile / 2), 512 / Tile];
        let (rotation, wave, start) = output;
        let positions = positions
            .view()
            .tile::<m![All], { Tile::SIZE }, m![All = { Tile::SIZE } # All]>(start);
        // SAFETY: the window's rows are the wave's, in order.
        let positions: DmTensor<i32, Chip, Halves, Copies, m![1 # 2]> =
            unsafe { positions.reshape::<Chip, m![Tile # All]>() }.to_dm(&mut context.tdma);
        let coefficients: DmTensor<bf16, Chip, Halves, Copies, m![Half, OutputHalf, InputHalf]> =
            self.frequency.dma_gather_unscaled(&positions);
        let coefficients: DmTensor<bf16, Chip, Halves, Copies, m![OutputHalf, InputHalf, Half]> =
            context
                .main
                .begin(coefficients.view())
                .fetch::<m![Half], m![OutputHalf, InputHalf]>()
                .collect::<m![Half], m![(OutputHalf, InputHalf) # 16]>()
                .transpose::<m![Half / 4, OutputHalf, InputHalf], m![Half % 4 # 16]>()
                .commit_trim::<m![Half % 4]>()
                .commit();
        coefficients.view().to_hbm_view(
            &mut context.tdma,
            rotation
                .view_mut()
                .tile::<m![S], 1, m![S = 1 #{!} S, Tile, OutputHalf, InputHalf, Half]>(wave),
        );
    }
}
