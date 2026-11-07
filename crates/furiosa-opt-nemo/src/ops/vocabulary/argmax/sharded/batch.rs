use super::*;

macro_rules! batch {
    ($name:ident, $groups:literal, [$($group:literal),*]) => {
        pub mod $name {
            use super::*;

            axes![Group = $groups];
            type Batch<D, Rows, Chips> = DmTensor<D, Chips, Cluster, m![1 # 256], m![Group, Rows, 1 # 8]>;

            pub(super) fn apply<V: AxisName, Rows: AxisName, Chips: M>(
                greedy: &Argmax<V, Chips>,
                device: &mut Device,
                logits: [&Scores<V, Rows, Chips>; $groups],
            ) -> [Tokens<Rows, Chips>; $groups] {
                let mut local: Batch<f32, Rows, Chips> = DmTensor::new();
                $(
                    greedy.maximum(device, logits[$group]).view().to_dm_view(
                        &mut device.tdma,
                        local.view_mut().tile::<m![Group], 1, m![
                            Group = 1 #{!} Group,
                            Rows,
                            1 # 8
                        ]>($group),
                    );
                )*
                let peer = local
                    .chip_shuffle([1, 0])
                    .to_dm::<m![Group, Rows, 1 # 8]>(&mut device.tdma);
                let maximum: Batch<f32, Rows, Chips> = device
                    .main
                    .begin_interleaved::<Term, _, _, _, _, _>(local.view(), peer.view())
                    .fetch::<m![Group, Rows / 2, Term, Rows % 2], m![1 # 8]>()
                    .collect::<m![Group, Rows / 2, Term, Rows % 2], m![1 # 8]>()
                    .vector_init()
                    .vector_intra_slice_unzip::<Term, m![Group, Rows]>()
                    .vector_clip_zip(ClipBinaryOpF32::Max)
                    .vector_final()
                    .commit_trim::<m![1 # 8]>()
                    .commit();
                let indices = greedy.register(device);
                let mut local: Batch<i32, Rows, Chips> = DmTensor::new();
                $(
                    let mut spread: Maximum<Rows, Chips> = DmTensor::new();
                    maximum
                        .view()
                        .tile::<m![Group], 1, m![Group = 1 # Group, Rows, 1 # 8]>($group)
                        .to_dm_view(&mut device.tdma, spread.view_mut());
                    greedy.rank(device, (logits[$group], &spread, &indices)).view().to_dm_view(
                        &mut device.tdma,
                        local.view_mut().tile::<m![Group], 1, m![
                            Group = 1 #{!} Group,
                            Rows,
                            1 # 8
                        ]>($group),
                    );
                )*
                let peer = local
                    .chip_shuffle([1, 0])
                    .to_dm::<m![Group, Rows, 1 # 8]>(&mut device.tdma);
                let rank: Batch<i32, Rows, Chips> = device
                    .main
                    .begin_interleaved::<Term, _, _, _, _, _>(local.view(), peer.view())
                    .fetch::<m![Group, Rows / 2, Term, Rows % 2], m![1 # 8]>()
                    .collect::<m![Group, Rows / 2, Term, Rows % 2], m![1 # 8]>()
                    .vector_init()
                    .vector_intra_slice_unzip::<Term, m![Group, Rows]>()
                    .vector_clip_zip(ClipBinaryOpI32::Max)
                    .vector_final()
                    .commit_trim::<m![1 # 8]>()
                    .commit();
                let tokens: DmTensor<i32, Chips, Cluster, m![1 # 256], m![Group, Rows, 1 # 4]> =
                    device
                        .main
                        .begin(rank.view())
                        .fetch::<m![Group, Rows], m![1 # 8]>()
                        .collect::<m![Group, Rows], m![1 # 8]>()
                        .vector_init()
                        .vector_intra_slice_tag(TagMode::Zero)
                        .vector_fxp_with_mode(FxpBinaryOp::SubFxp, BinaryArgMode::Mode10, Argmax::<V, Chips>::LAST)
                        .vector_final()
                        .commit_trim::<m![1 # 4]>()
                        .commit();
                [$({
                    let mut group: Tokens<Rows, Chips> = DmTensor::new();
                    tokens
                        .view()
                        .tile::<m![Group], 1, m![Group = 1 # Group, Rows, 1 # 4]>($group)
                        .to_dm_view(&mut device.tdma, group.view_mut());
                    group
                }),*]
            }
        }
        impl<V: AxisName, C: M, R: AxisName, S: Space> Module<(&mut Device, Dm<S, [&Scores<V, R, C>; $groups]>)>
            for Argmax<V, C>
        {
            type Output = [Tokens<R, C>; $groups];

            fn forward(&self, (device, input): (&mut Device, Dm<S, [&Scores<V, R, C>; $groups]>)) -> Self::Output {
                $name::apply(self, device, input.value)
            }
        }
    };
}

batch!(pair, 2, [0, 1]);
batch!(three, 3, [0, 1, 2]);
batch!(four, 4, [0, 1, 2, 3]);
