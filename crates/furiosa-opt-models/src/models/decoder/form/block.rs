use super::*;

impl Form for BlockScaled {
    type Stored = f8e4m3;

    type Of<In: AxisName, Out: AxisName, D: Scalar, Chips: M + 'static, X: Memory, V> =
        tensor::form::BlockScaled<
            In,
            Out,
            f8e4m3,
            f32,
            Scales<In, Out>,
            Block,
            Chips,
            V,
            X::Tensor<f32, Chips, Scales<In, Out>>,
        >;

    type Query<Hidden: AxisName, Heads: AxisName, Head: AxisName, X: Memory> = linear::Affine<
        query::fp8::Weights<Hidden, Heads, Head, query::fp8::HeadSquares<Heads, X>, X>,
        (),
    >;
    type Key<Hidden: AxisName, Heads: AxisName, Head: AxisName, X: Memory> = linear::Affine<
        (
            query::fp8::Weights<Hidden, Heads, Head, query::fp8::HeadSquares<Heads, X>, X>,
            (),
        ),
        (),
    >;
    type Value<Hidden: AxisName, Heads: AxisName, Head: AxisName, X: Memory> = linear::Affine<
        query::fp8::Weights<Hidden, Heads, Head, query::fp8::HeadScales<Heads, Head, X>, X>,
        (),
    >;
    type Output<Queries: AxisName, Hidden: AxisName, X: Memory> =
        output::fp8::Stored<Hidden, Queries, X>;
    type GateUp<Hidden: AxisName, Intermediate: AxisName, X: Memory> =
        swiglu::spans::fp8::GateUpWeights<Hidden, Intermediate, X>;
    type Down<Hidden: AxisName, Intermediate: AxisName, X: Memory> =
        swiglu::spans::fp8::DownWeights<Hidden, Intermediate, X>;

    const REQUIREMENT: Requirement = Requirement::Declared(Declaration::Fp8(Scheme::Block));
    type Caches = (bf16,);
}
