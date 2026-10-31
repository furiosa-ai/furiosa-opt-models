//! What the one catalog binding is generic over, private to Qwen3.

/// The catalog binding of a size `S` in projection form `P`, at the count of layers the size's
/// layer array holds, and its body.
pub trait Depth<S, P, C, X> {
    type Catalog;
    type Body;
}
