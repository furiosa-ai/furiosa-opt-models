use super::*;
use crate::modules::generic::schedule::Begin;
use crate::{Module, Op};

/// Seeds a group's state from its own fresh keys under the block's causal mask.
pub(super) struct CausalBlock<T>(pub(super) PhantomData<T>);

#[must_use]
pub(super) struct Scored<
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    T: Group<KvHeads, Groups, Head>,
> {
    scores: history::PanelScores<KvHeads, Groups, T::Fresh, T>,
    mask: GroupMask<KvHeads, Groups, Head, T>,
}

/// Scores the group's own fresh keys, then writes the block's causal mask.
impl<
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    T: Group<KvHeads, Groups, Head>,
    All: AxisName,
    W: AxisName,
    Q: QueryTiles<KvHeads, Groups, Head>,
>
    Op<
        Begin<(
            &Input<'_, KvHeads, Groups, Head, All, W>,
            (
                fresh::Keys<KvHeads, Groups, Head>,
                &mut Load<'_, '_, KvHeads, Groups, Head, All, W, T::Fresh>,
            ),
            &Q,
        )>,
    > for CausalBlock<T>
{
    type Output = Scored<KvHeads, Groups, Head, T>;

    fn forward(
        &self,
        context: &mut Device,
        Begin((input, (key, source), queries)): Begin<(
            &Input<'_, KvHeads, Groups, Head, All, W>,
            (
                fresh::Keys<KvHeads, Groups, Head>,
                &mut Load<'_, '_, KvHeads, Groups, Head, All, W, T::Fresh>,
            ),
            &Q,
        )>,
    ) -> Self::Output {
        let scores = T::scores(input.attention, context, (&mut *source, key), queries);
        // Behind the key loads, so it arrives during the contractions.
        let mask = T::mask::<All, W>(context, input);
        Scored { scores, mask }
    }
}

/// Seeds the group's state from the scored block under its mask, and returns the mask.
impl<
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    T: Group<KvHeads, Groups, Head>,
    All: AxisName,
    W: AxisName,
    C: state::Tiles<KvHeads, Groups, Head>,
    R: state::Tiles<KvHeads, Groups, Head>,
>
    Op<(
        Scored<KvHeads, Groups, Head, T>,
        (
            &mut Load<'_, '_, KvHeads, Groups, Head, All, W, T::Fresh>,
            (&mut C, &mut R),
        ),
    )> for CausalBlock<T>
{
    type Output = GroupMask<KvHeads, Groups, Head, T>;

    fn forward(
        &self,
        context: &mut Device,
        (scored, (source, state)): (
            Scored<KvHeads, Groups, Head, T>,
            (
                &mut Load<'_, '_, KvHeads, Groups, Head, All, W, T::Fresh>,
                (&mut C, &mut R),
            ),
        ),
    ) -> Self::Output {
        let Scored { scores, mask } = scored;
        T::seed(context, (&scores, &mask), source, state);
        mask
    }
}

impl<
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    T: Group<KvHeads, Groups, Head>,
    All: AxisName,
    W: AxisName,
    Q: QueryTiles<KvHeads, Groups, Head>,
    C: state::Tiles<KvHeads, Groups, Head>,
    R: state::Tiles<KvHeads, Groups, Head>,
    G: Fn() -> usize,
    H,
    S,
    Env: Copy,
>
    Module<
        (
            &Input<'_, KvHeads, Groups, Head, All, W>,
            (
                fresh::Keys<KvHeads, Groups, Head>,
                &mut Load<'_, '_, KvHeads, Groups, Head, All, W, T::Fresh>,
            ),
            &Q,
            (&mut C, &mut R),
            (G, (&H, &mut S), usize),
        ),
        Env,
        &mut (),
    > for CausalBlock<T>
where
    H: for<'d, 's> Module<&'s mut S, Env, &'d mut (), Output = ()>,
{
    type Output = GroupMask<KvHeads, Groups, Head, T>;

    fn forward(
        &self,
        context: &mut Device,
        env: Env,
        _: &mut (),
        (input, (key, source), queries, state, (guest, (boundary, place), keeps)): (
            &Input<'_, KvHeads, Groups, Head, All, W>,
            (
                fresh::Keys<KvHeads, Groups, Head>,
                &mut Load<'_, '_, KvHeads, Groups, Head, All, W, T::Fresh>,
            ),
            &Q,
            (&mut C, &mut R),
            (G, (&H, &mut S), usize),
        ),
    ) -> Self::Output {
        let scored = Module::forward(
            &self,
            &mut *context,
            env,
            &mut (),
            Begin((input, (key, &mut *source), queries)),
        );
        // Behind the block's key and mask loads, where a group's queue idles through its softmax.
        for _ in 0..keeps {
            for _ in 0..guest() {
                boundary.forward(&mut *context, env, &mut (), &mut *place);
            }
        }
        Module::forward(&self, context, env, &mut (), (scored, (source, state)))
    }
}
