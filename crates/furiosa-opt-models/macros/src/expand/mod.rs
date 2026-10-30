//! The derived impls: the checkpoint shapes (build, update in place), which a `tree` struct writes
//! itself, and the tree shapes (borrow, copy in place).

use proc_macro2::{Ident, Span, TokenStream};
use quote::{ToTokens, format_ident, quote};
use syn::ext::IdentExt;
use syn::{
    Data, DeriveInput, Error, Fields, GenericParam, Generics, Member, Result, Type, TypeParamBound,
    WherePredicate,
};

use crate::attribute::{self, Read};

mod checkpoint;
mod substitute;
mod tree;

/// A struct being derived, with its parsed fields.
pub(crate) struct Shape<'a> {
    name: &'a Ident,
    generics: &'a Generics,
    /// The struct's fields as declared: named, a tuple's or none.
    layout: &'a Fields,
    fields: Vec<Field<'a>>,
    declared: attribute::Struct,
    /// The constant deciding a tie checked against the config, when a field declares one.
    tied: Option<TokenStream>,
}

struct Field<'a> {
    member: Member,
    /// The field name, or the index of a tuple field.
    tree: String,
    /// `rename`, else the field name; a tuple field adds no checkpoint segment.
    checkpoint: String,
    ty: &'a Type,
    read: Read,
    binding: Ident,
}

impl<'a> Shape<'a> {
    /// The `Load` impls of `input`, one per direction a tree is loaded in.
    pub(crate) fn derive(input: &'a DeriveInput) -> Result<TokenStream> {
        let Data::Struct(data) = &input.data else {
            return Err(Error::new(
                input.ident.span(),
                "`derive(Load)` supports structs",
            ));
        };
        let declared = attribute::Struct::parse(&input.attrs)?;
        let fields = data
            .fields
            .iter()
            .enumerate()
            .map(|(index, field)| {
                if declared.from.is_some() && field.attrs.iter().any(|a| a.path().is_ident("load"))
                {
                    return Err(Error::new_spanned(
                        field,
                        "fields of a `from` struct take no `load` attributes",
                    ));
                }
                let attribute::Field { rename, read } = attribute::Field::parse(&field.attrs)?;
                if declared.tree && !matches!(read, Read::Field) {
                    return Err(Error::new_spanned(
                        field,
                        "a `tree` struct reads the checkpoint itself, so its fields take only \
                         `rename`",
                    ));
                }
                let (member, tree, checkpoint) = match &field.ident {
                    Some(ident) => (
                        Member::Named(ident.clone()),
                        ident.unraw().to_string(),
                        ident.unraw().to_string(),
                    ),
                    None => (
                        Member::Unnamed(index.into()),
                        index.to_string(),
                        String::new(),
                    ),
                };
                Ok(Field {
                    member,
                    tree,
                    checkpoint: rename.map_or(checkpoint, |name| name.value()),
                    ty: &field.ty,
                    read,
                    binding: format_ident!("__f{index}"),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let tying = fields.iter().find_map(|field| match &field.read {
            Read::Tied(_, Some(key)) => Some(key),
            _ => None,
        });
        // The constant deciding a tie that a `config` key is checked against:
        // `G::TIED_EMBEDDINGS` of the one parameter bounded by `Geometry`.
        let decides = |span: Span| -> Result<TokenStream> {
            let generics = &input.generics;
            let geometry = |bound: &TypeParamBound| match bound {
                TypeParamBound::Trait(t) => t
                    .path
                    .segments
                    .last()
                    .is_some_and(|s| s.ident == "Geometry"),
                _ => false,
            };
            let declared = generics
                .where_clause
                .iter()
                .flat_map(|clause| &clause.predicates)
                .filter_map(|predicate| match predicate {
                    WherePredicate::Type(t) if t.bounds.iter().any(geometry) => Some(&t.bounded_ty),
                    _ => None,
                });
            let params: Vec<&Ident> = generics
                .type_params()
                .filter(|param| {
                    param.bounds.iter().any(geometry)
                        || declared
                            .clone()
                            .any(|ty| matches!(ty, Type::Path(p) if p.path.is_ident(&param.ident)))
                })
                .map(|param| &param.ident)
                .collect();
            match params.as_slice() {
                [param] => Ok(quote!(#param::TIED_EMBEDDINGS)),
                [] => Err(Error::new(
                    span,
                    "`tied` with `config` needs a type parameter bounded by `Geometry`, whose \
                     `TIED_EMBEDDINGS` decides the tie",
                )),
                _ => Err(Error::new(
                    span,
                    "more than one type parameter is bounded by `Geometry`",
                )),
            }
        };
        let tied = tying.map(|key| decides(key.span())).transpose()?;
        let shape = Shape {
            name: &input.ident,
            generics: &input.generics,
            layout: &data.fields,
            fields,
            declared,
            tied,
        };
        let checkpoint = match shape.declared.tree {
            true => Vec::new(),
            false => vec![shape.build(), shape.update()],
        };
        Ok(checkpoint
            .into_iter()
            .chain([shape.borrow(), shape.copy(), shape.footprint()])
            .collect())
    }

    fn this(&self) -> TokenStream {
        let name = self.name;
        let (_, arguments, _) = self.generics.split_for_impl();
        quote!(#name #arguments)
    }

    /// The struct's generics with `front` (lifetimes) before and `back` after its own parameters.
    fn generics(&self, front: &[TokenStream], back: &[TokenStream]) -> Generics {
        let mut generics = self.generics.clone();
        let own = std::mem::take(&mut generics.params);
        generics.params = front
            .iter()
            .map(|param| syn::parse2::<GenericParam>(param.clone()).expect("a generic parameter"))
            .chain(own)
            .chain(
                back.iter()
                    .map(|param| syn::parse2(param.clone()).expect("a generic parameter")),
            )
            .collect();
        generics.make_where_clause();
        generics
    }

    /// `name { a: x, b: y }`, `name(x, y)` or `name`, with `value` of each field.
    fn construct(&self, head: TokenStream, value: impl Fn(&Field) -> TokenStream) -> TokenStream {
        let values = self
            .fields
            .iter()
            .map(|field| (&field.member, value(field)));
        match self.layout {
            Fields::Named(_) => {
                let fields = values.map(|(member, value)| quote!(#member: #value));
                quote!(#head { #(#fields),* })
            }
            Fields::Unnamed(_) => {
                let values = values.map(|(_, value)| value);
                quote!(#head(#(#values),*))
            }
            Fields::Unit => head,
        }
    }

    /// The pattern binding every field of a value of the struct by value.
    fn destructure(&self) -> TokenStream {
        let name = self.name;
        self.construct(quote!(#name), |field| field.binding.to_token_stream())
    }

    fn ignore(&self) -> TokenStream {
        let prefixes = &self.declared.ignore;
        quote!(#(__cx.ignore(#prefixes);)*)
    }

    /// `T: Footprint`: the sum of its fields'. Each bound is over a lifetime, so a field type
    /// without a footprint leaves the impl unused rather than refused.
    fn footprint(&self) -> TokenStream {
        let load = quote!(::furiosa_opt_models::load);
        let this = self.this();
        let types: Vec<&Type> = self.fields.iter().map(|field| field.ty).collect();
        let mut generics = self.generics.clone();
        let predicates = &mut generics.make_where_clause().predicates;
        predicates.extend(
            types
                .iter()
                .map(|ty| -> WherePredicate { syn::parse_quote!(for<'__n> #ty: #load::Footprint) }),
        );
        let (impl_generics, _, where_clause) = generics.split_for_impl();
        quote! {
            impl #impl_generics #load::Footprint for #this #where_clause {
                const SIZE: usize = 0 #(+ <#types as #load::Footprint>::SIZE)*;
            }
        }
    }

    /// `visit` under the struct's config key, when it declares one.
    fn scoped(&self, visit: TokenStream) -> TokenStream {
        match &self.declared.config {
            Some(key) => quote!(__cx.config(#key, |__cx| #visit)),
            None => visit,
        }
    }

    /// The fields in visit order: the tied ones last, so their targets are read before them.
    fn visits(&self) -> impl Iterator<Item = &Field<'_>> {
        let tied = |field: &&Field| matches!(field.read, Read::Tied(..));
        let own = self.fields.iter().filter(move |field| !tied(field));
        own.chain(self.fields.iter().filter(tied))
    }

    /// The subtrees the tied fields read, relative to the struct.
    fn targets(&self) -> Vec<&syn::LitStr> {
        self.fields
            .iter()
            .filter_map(|field| match &field.read {
                Read::Tied(target, _) => Some(target),
                _ => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    fn refusal(input: DeriveInput) -> String {
        Shape::derive(&input)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default()
    }

    /// A `tree` struct derives only the tree shapes; its fields take no checkpoint read.
    #[test]
    fn derives_only_the_tree_shapes_of_a_tree_struct() {
        let derived = Shape::derive(&parse_quote! {
            #[load(tree)]
            struct Norm<W> { weight: W }
        })
        .unwrap()
        .to_string();
        assert!(!derived.contains("Checkpoint"), "{derived}");
        assert!(derived.contains("Footprint"), "{derived}");
        let read = refusal(parse_quote! {
            #[load(tree)]
            struct Norm<W> { #[load(config = "eps")] weight: W }
        });
        assert!(read.contains("take only `rename`"), "{read}");
    }

    /// A tie checked against the config needs exactly one parameter bounded by `Geometry`.
    #[test]
    fn refuses_a_checked_tie_without_one_geometry() {
        let none = refusal(parse_quote! {
            struct Head<X: Memory> {
                embed: Linear<X>,
                #[load(tied = "embed", config = "tie_word_embeddings")]
                head: Linear<X>,
            }
        });
        assert!(
            none.contains("needs a type parameter bounded by `Geometry`"),
            "{none}"
        );
        let two = refusal(parse_quote! {
            struct Head<G: Geometry, H, X: Memory> where H: Geometry {
                embed: Linear<G, H, X>,
                #[load(tied = "embed", config = "tie_word_embeddings")]
                head: Linear<G, H, X>,
            }
        });
        assert!(two.contains("more than one"), "{two}");
        assert!(
            Shape::derive(&parse_quote! {
                struct Head<G, X: Memory> where G: Geometry {
                    embed: Linear<G, X>,
                    #[load(tied = "embed", config = "tie_word_embeddings")]
                    head: Linear<G, X>,
                }
            })
            .is_ok()
        );
    }
}
