//! Latent attention: queries attend over a cache of shared key/value latents, all rows or the
//! rows a selector picks.

use super::{Attention, KeyValue, Selected};
use crate::Module;
use furiosa_opt_std::prelude::{Device, DeviceSend};

/// Latent attention over the history rows a selector picks for each query.
pub type Sparse<Q, W, S, A, P> = Attention<Q, Selected<W, S>, A, P>;

/// Projects the queries and the shared key/value latent, rotating only their rotary channels.
#[derive(DeviceSend)]
pub struct LatentProjection<Query, KeyValue, Norm, Phases, Rotate, Absorb> {
    pub query: Query,
    pub key_value: KeyValue,
    pub norm: Norm,
    pub phases: Phases,
    pub rotate: Rotate,
    pub absorb: Absorb,
}

/// The state holds the query projection's slot, then the key/value projection's.
impl<
    's,
    X,
    P,
    Q,
    RotaryQuery,
    L,
    RotaryKey,
    NormalizedLatent,
    F,
    RotatedQuery,
    RotatedKey,
    AbsorbedQuery,
    QueryState,
    KeyValueState,
    Query,
    KeyValue,
    Norm,
    Phases,
    Rotate,
    Absorb,
    Env: Copy,
> Module<(&X, &P), Env, &'s mut (QueryState, KeyValueState)>
    for LatentProjection<Query, KeyValue, Norm, Phases, Rotate, Absorb>
where
    Query: for<'x> Module<&'x X, Env, &'s mut QueryState, Output = (Q, RotaryQuery)>,
    KeyValue: for<'x> Module<&'x X, Env, &'s mut KeyValueState, Output = (L, RotaryKey)>,
    Norm: for<'x, 'd> Module<&'x L, Env, &'d mut (), Output = NormalizedLatent>,
    Phases: for<'x, 'd> Module<&'x P, Env, &'d mut (), Output = F>,
    Rotate: for<'a, 'd> Module<
            (&'a RotaryQuery, &'a RotaryKey, &'a F),
            Env,
            &'d mut (),
            Output = (RotatedQuery, RotatedKey),
        >,
    Absorb: for<'x, 'd> Module<&'x Q, Env, &'d mut (), Output = AbsorbedQuery>,
{
    type Output = ((AbsorbedQuery, RotatedQuery), NormalizedLatent, RotatedKey);

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (query_state, key_value_state): &'s mut (QueryState, KeyValueState),
        (input, positions): (&X, &P),
    ) -> Self::Output {
        let (query, query_rotary) = self.query.forward(&mut *device, env, query_state, input);
        let (latent, key_rotary) =
            self.key_value
                .forward(&mut *device, env, key_value_state, input);
        let latent = self.norm.forward(&mut *device, env, &mut (), &latent);
        let phases = self.phases.forward(&mut *device, env, &mut (), positions);
        let (query_rotary, key_rotary) = self.rotate.forward(
            &mut *device,
            env,
            &mut (),
            (&query_rotary, &key_rotary, &phases),
        );
        let query = self.absorb.forward(device, env, &mut (), &query);
        ((query, query_rotary), latent, key_rotary)
    }
}

/// Attention over a cache of key/value latents, one per token.
pub type Latent<Q, W, A, P> = Attention<Q, KeyValue<W, ()>, A, P>;
