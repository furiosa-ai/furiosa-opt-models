//! Structural residency projection, adapted from the reference derive. Every declared field
//! participates; checkpoint names and layer stacking belong to other interfaces.

use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::ext::IdentExt;
use syn::spanned::Spanned;
use syn::visit_mut::{self, VisitMut};
use syn::{
    Attribute, Data, DeriveInput, Error, Fields, GenericParam, Ident, Index, Member, Type,
    parse_quote,
};

/// A declaration and what its attributes say about each generic position and field.
struct Catalog<'a> {
    input: &'a DeriveInput,
    children: Vec<Ident>,
    memory: Option<Ident>,
    shape: Shape,
}

enum Shape {
    Struct(Variant),
    Enum(Vec<(Ident, Variant)>),
}

/// One field list: a struct's, or a variant's.
struct Variant {
    style: Style,
    fields: Vec<Field>,
}

#[derive(Clone, Copy)]
enum Style {
    Named,
    Unnamed,
    Unit,
}

struct Field {
    member: Member,
    /// The local a destructuring binds the field to.
    binding: Ident,
    ty: Type,
    part: Part,
}

/// How a field takes part in the projection.
enum Part {
    Projected(Segment),
}

/// The path segment a projected field adds to its weights' paths.
enum Segment {
    Field(String),
    Index(usize),
    Transparent,
    Role(Box<Type>),
}

impl<'a> Catalog<'a> {
    fn parse(input: &'a DeriveInput) -> syn::Result<Self> {
        let (mut children, mut memory) = (Vec::new(), None);
        for attribute in residency_attributes(&input.attrs) {
            attribute.parse_nested_meta(|meta| {
                if meta.path.is_ident("children") {
                    meta.parse_nested_meta(|child| {
                        let child = type_parameter(input, child.path.require_ident()?)?;
                        if children.contains(&child) {
                            return Err(Error::new(
                                child.span(),
                                "a child position is declared once",
                            ));
                        }
                        children.push(child);
                        Ok(())
                    })
                } else if meta.path.is_ident("memory") {
                    meta.parse_nested_meta(|slot| {
                        let slot = type_parameter(input, slot.path.require_ident()?)?;
                        match memory.replace(slot.clone()) {
                            Some(_) => Err(Error::new(slot.span(), "a tree has one memory")),
                            None => Ok(()),
                        }
                    })
                } else {
                    Err(meta.error("expected `children(..)` or `memory(..)`"))
                }
            })?;
        }
        if let Some(slot) = &memory
            && children.contains(slot)
        {
            return Err(Error::new(
                slot.span(),
                "a memory position cannot also be a child",
            ));
        }
        let bounded = input
            .generics
            .type_params()
            .find(|parameter| children.contains(&parameter.ident) && (!parameter.bounds.is_empty()
                || input.generics.where_clause.iter().flat_map(|clause| &clause.predicates).any(|predicate| {
                    matches!(predicate, syn::WherePredicate::Type(bound)
                        if matches!(&bound.bounded_ty, Type::Path(path) if path.path.is_ident(&parameter.ident)))
                })));
        if let Some(parameter) = bounded {
            return Err(Error::new(
                parameter.span(),
                "a child position holds any catalog tree, so its declaration bounds it by nothing",
            ));
        }
        let shape = match &input.data {
            Data::Struct(data) => Shape::Struct(Variant::parse(&data.fields)?),
            Data::Enum(data) => Shape::Enum(
                (data.variants.iter())
                    .map(|variant| Ok((variant.ident.clone(), Variant::parse(&variant.fields)?)))
                    .collect::<syn::Result<_>>()?,
            ),
            Data::Union(_) => {
                return Err(Error::new(
                    input.ident.span(),
                    "a union has no fields to walk",
                ));
            }
        };
        Ok(Self {
            input,
            children,
            memory,
            shape,
        })
    }

    fn expand(&self) -> TokenStream2 {
        let paths = Paths::new();
        let Paths {
            residency,
            memory,
            hbm,
            leaves,
            visit,
            path,
        } = &paths;
        let ident = &self.input.ident;
        let parameters = &self.input.generics.params;
        let arguments = |child: &dyn Fn(&Ident) -> TokenStream2, slot: TokenStream2| {
            let arguments = parameters.iter().map(|parameter| match parameter {
                GenericParam::Lifetime(lifetime) => {
                    let lifetime = &lifetime.lifetime;
                    quote!(#lifetime)
                }
                GenericParam::Type(parameter) if self.memory.as_ref() == Some(&parameter.ident) => {
                    slot.clone()
                }
                GenericParam::Type(parameter) if self.children.contains(&parameter.ident) => {
                    child(&parameter.ident)
                }
                GenericParam::Type(parameter) => {
                    let ident = &parameter.ident;
                    quote!(#ident)
                }
                GenericParam::Const(parameter) => {
                    let ident = &parameter.ident;
                    quote!(#ident)
                }
            });
            match parameters.is_empty() {
                true => quote!(#ident),
                false => quote!(#ident<#(#arguments),*>),
            }
        };
        let this = arguments(&|child| quote!(#child), quote!(#hbm));
        let projected = arguments(
            &|child| quote!(<#child as #residency>::In<__R>),
            quote!(__R),
        );
        let mut generics = self.input.generics.clone();
        generics.params = (generics.params.into_iter())
            .filter(|parameter| {
                !matches!(parameter, GenericParam::Type(parameter)
                    if self.memory.as_ref() == Some(&parameter.ident))
            })
            .map(|mut parameter| {
                if let GenericParam::Type(parameter) = &mut parameter
                    && self.children.contains(&parameter.ident)
                {
                    parameter.bounds.push(parse_quote!(#residency));
                }
                parameter
            })
            .collect();
        // Bounds naming the memory slot must refer to the canonical HBM form too.
        if let Some(slot) = &self.memory {
            Substitute {
                slot,
                paths: &paths,
            }
            .visit_generics_mut(&mut generics);
        }
        let (generics, _, where_clause) = generics.split_for_impl();
        let [map, visit_body] = self.bodies(&paths);
        quote! {
            #[automatically_derived]
            impl #generics #residency for #this #where_clause {
                type In<__R: #memory> = #projected;

                #[allow(unused_variables)]
                fn map<'__t, __F: #memory, __T: #memory, __L: #leaves<'__t, __F, __T>>(
                    __tree: &'__t Self::In<__F>,
                    __leaves: &mut __L,
                    __path: #path<'_>,
                ) -> ::core::result::Result<Self::In<__T>, __L::Error> {
                    ::core::result::Result::Ok(#map)
                }

                #[allow(unused_variables)]
                fn visit<__R: #memory, __V: #visit<__R>>(
                    __tree: &Self::In<__R>,
                    __visitor: &mut __V,
                    __path: #path<'_>,
                ) -> ::core::result::Result<(), __V::Error> {
                    #visit_body
                    ::core::result::Result::Ok(())
                }


            }
        }
    }

    /// Map and visit share the declaration's field walk and enum branch structure.
    fn bodies(&self, paths: &Paths) -> [TokenStream2; 2] {
        let ident = &self.input.ident;
        match &self.shape {
            Shape::Struct(variant) => {
                let walk = Walk {
                    catalog: self,
                    paths,
                    base: quote!(__path),
                };
                let source = |field: &Field| {
                    let member = &field.member;
                    quote!(&__tree.#member)
                };
                [
                    variant.construct(&quote!(#ident), |field| walk.map(field, &source(field))),
                    variant
                        .fields
                        .iter()
                        .map(|field| walk.visit(field, &source(field)))
                        .collect(),
                ]
            }
            Shape::Enum(variants) => {
                let mut maps = Vec::new();
                let mut visits = Vec::new();
                for (name, variant) in variants {
                    let constructor = quote!(#ident::#name);
                    let segment = snake(name);
                    let walk = Walk {
                        catalog: self,
                        paths,
                        base: quote!(__variant),
                    };
                    let bound = |field: &Field| {
                        let binding = &field.binding;
                        quote!(#binding)
                    };
                    let pattern = variant.pattern(&constructor);
                    let map =
                        variant.construct(&constructor, |field| walk.map(field, &bound(field)));
                    let visit: TokenStream2 = variant
                        .fields
                        .iter()
                        .map(|field| walk.visit(field, &bound(field)))
                        .collect();
                    maps.push(quote!(#pattern => { let __variant = __path.field(#segment); #map }));
                    visits.push(
                        quote!(#pattern => { let __variant = __path.field(#segment); #visit }),
                    );
                }
                [
                    quote!(match __tree { #(#maps,)* }),
                    quote!(match __tree { #(#visits,)* }),
                ]
            }
        }
    }
}

/// The projection of one field list's fields, each under `base`, the path of their parent.
struct Walk<'a> {
    catalog: &'a Catalog<'a>,
    paths: &'a Paths,
    base: TokenStream2,
}

impl Walk<'_> {
    fn map(&self, field: &Field, source: &TokenStream2) -> TokenStream2 {
        let residency = &self.paths.residency;
        match &field.part {
            Part::Projected(segment) => {
                let (ty, segment) = (self.projected(&field.ty), self.segment(segment));
                quote!(<#ty as #residency>::map(#source, __leaves, #segment)?)
            }
        }
    }

    fn visit(&self, field: &Field, source: &TokenStream2) -> TokenStream2 {
        let residency = &self.paths.residency;
        match &field.part {
            Part::Projected(segment) => {
                let (ty, segment) = (self.projected(&field.ty), self.segment(segment));
                quote!(<#ty as #residency>::visit(#source, __visitor, #segment)?;)
            }
        }
    }

    fn segment(&self, segment: &Segment) -> TokenStream2 {
        let base = &self.base;
        match segment {
            Segment::Field(name) => quote!(#base.field(#name)),
            Segment::Index(index) => quote!(#base.index(#index)),
            Segment::Transparent => quote!(#base),
            Segment::Role(role) => quote!(#base.role::<#role>()),
        }
    }

    /// The field's type in this tree's HBM form: its memory slot, if any, in HBM.
    fn projected(&self, ty: &Type) -> Type {
        let mut ty = ty.clone();
        if let Some(slot) = &self.catalog.memory {
            Substitute {
                slot,
                paths: self.paths,
            }
            .visit_type_mut(&mut ty);
        }
        ty
    }
}

impl Variant {
    fn parse(fields: &Fields) -> syn::Result<Self> {
        let style = match fields {
            Fields::Named(_) => Style::Named,
            Fields::Unnamed(_) => Style::Unnamed,
            Fields::Unit => Style::Unit,
        };
        let fields = (fields.iter().enumerate())
            .map(|(index, field)| {
                let (member, binding, segment) = match &field.ident {
                    Some(ident) => (
                        Member::Named(ident.clone()),
                        format_ident!("__field{index}"),
                        Segment::Field(ident.unraw().to_string()),
                    ),
                    None => (
                        Member::Unnamed(Index::from(index)),
                        format_ident!("__{index}"),
                        Segment::Index(index),
                    ),
                };
                Ok(Field {
                    member,
                    binding,
                    ty: field.ty.clone(),
                    part: part(&field.attrs, segment)?,
                })
            })
            .collect::<syn::Result<_>>()?;
        Ok(Self { style, fields })
    }

    /// `constructor` applied to each field's `value`.
    fn construct(
        &self,
        constructor: &TokenStream2,
        value: impl Fn(&Field) -> TokenStream2,
    ) -> TokenStream2 {
        let values = self.fields.iter().map(|field| {
            let value = value(field);
            match &field.member {
                Member::Named(ident) => quote!(#ident: #value),
                Member::Unnamed(_) => value,
            }
        });
        match self.style {
            Style::Named => quote!(#constructor { #(#values),* }),
            Style::Unnamed => quote!(#constructor(#(#values),*)),
            Style::Unit => quote!(#constructor),
        }
    }

    /// The pattern binding every field to its local.
    fn pattern(&self, constructor: &TokenStream2) -> TokenStream2 {
        self.construct(constructor, |field| {
            let binding = &field.binding;
            quote!(#binding)
        })
    }
}

/// Replaces the memory slot of a field's type with the HBM memory.
struct Substitute<'a> {
    slot: &'a Ident,
    paths: &'a Paths,
}

impl VisitMut for Substitute<'_> {
    fn visit_type_mut(&mut self, ty: &mut Type) {
        if let Type::Path(path) = ty
            && path.qself.is_none()
            && path.path.leading_colon.is_none()
            && let Some(first) = path.path.segments.first()
            && first.ident == *self.slot
        {
            let Paths { hbm, memory, .. } = self.paths;
            let rest = path.path.segments.iter().skip(1);
            *ty = match path.path.segments.len() {
                1 => parse_quote!(#hbm),
                _ => parse_quote!(<#hbm as #memory>#(::#rest)*),
            };
        }
        visit_mut::visit_type_mut(self, ty);
    }
}

/// The items generated code names, by their paths from the catalog crate's root.
struct Paths {
    residency: TokenStream2,
    memory: TokenStream2,
    hbm: TokenStream2,
    leaves: TokenStream2,
    visit: TokenStream2,
    path: TokenStream2,
}

impl Paths {
    fn new() -> Self {
        let memory = quote!(::furiosa_opt_models::tensor::memory);
        let residency = quote!(#memory::residency);
        Self {
            residency: quote!(#residency::Residency),
            memory: quote!(#memory::Memory),
            hbm: quote!(#memory::Hbm),
            leaves: quote!(#residency::Leaves),
            visit: quote!(#residency::Visit),
            path: quote!(#residency::Path),
        }
    }
}

pub(crate) fn derive(input: &DeriveInput) -> syn::Result<TokenStream2> {
    Catalog::parse(input).map(|catalog| catalog.expand())
}

/// What a field's `#[residency(..)]` attributes make it, by default projected under `segment`.
fn part(attributes: &[Attribute], segment: Segment) -> syn::Result<Part> {
    let mut segment = segment;
    let mut declared = false;
    for attribute in residency_attributes(attributes) {
        attribute.parse_nested_meta(|meta| {
            if declared {
                return Err(meta.error("a field takes one of `transparent` and `role`"));
            }
            segment = if meta.path.is_ident("transparent") {
                Segment::Transparent
            } else if meta.path.is_ident("role") {
                Segment::Role(Box::new(meta.value()?.parse()?))
            } else {
                return Err(
                    meta.error("expected `transparent` or `role = R`; every field is traversed")
                );
            };
            declared = true;
            Ok(())
        })?;
    }
    Ok(Part::Projected(segment))
}

fn residency_attributes(attributes: &[Attribute]) -> impl Iterator<Item = &Attribute> {
    (attributes.iter()).filter(|attribute| attribute.path().is_ident("residency"))
}

/// `ident`, which must name a type parameter of the declaration.
fn type_parameter(input: &DeriveInput, ident: &Ident) -> syn::Result<Ident> {
    match input
        .generics
        .type_params()
        .any(|parameter| parameter.ident == *ident)
    {
        true => Ok(ident.clone()),
        false => Err(Error::new(
            ident.span(),
            "not a type parameter of this declaration",
        )),
    }
}

/// `Vocabulary` as `vocabulary`, `OutputHead` as `output_head`.
fn snake(ident: &Ident) -> String {
    let name = ident.unraw().to_string();
    name.char_indices()
        .flat_map(|(at, letter)| {
            let separator = (at > 0 && letter.is_uppercase()).then_some('_');
            separator.into_iter().chain(letter.to_lowercase())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refusal(input: DeriveInput) -> String {
        derive(&input)
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default()
    }

    #[test]
    fn accepts_struct_enum_and_explicit_projection_positions() {
        for input in [
            parse_quote!(
                #[residency(memory(X))]
                struct Norm<X: Memory> {
                    weight: X::Tensor<f32, C, S>,
                }
            ),
            parse_quote!(
                #[residency(children(A, B))]
                struct Layer<A, B>(A, B);
            ),
            parse_quote!(
                #[residency(children(T))]
                enum Branch<T> {
                    Named { value: T },
                    Tuple(T),
                    Empty,
                }
            ),
            parse_quote!(
                struct Empty;
            ),
        ] {
            let expanded = derive(&input).unwrap();
            syn::parse2::<syn::ItemImpl>(expanded).unwrap();
        }
    }

    #[test]
    fn refuses_ambiguous_or_unknown_generic_positions() {
        for (input, expected) in [
            (
                parse_quote!(
                    #[residency(memory())]
                    struct Layer<X>(X);
                ),
                "expected nested attribute",
            ),
            (
                parse_quote!(
                    #[residency(memory(Y))]
                    struct Layer<X>(X);
                ),
                "not a type parameter",
            ),
            (
                parse_quote!(
                    #[residency(children(X, X))]
                    struct Layer<X>(X);
                ),
                "declared once",
            ),
            (
                parse_quote!(
                    #[residency(memory(X), children(X))]
                    struct Layer<X>(X);
                ),
                "cannot also be a child",
            ),
            (
                parse_quote!(
                    #[residency(memory(X, Y))]
                    struct Layer<X, Y>(X, Y);
                ),
                "one memory",
            ),
            (
                parse_quote!(
                    #[residency(children(X))]
                    struct Layer<X: Clone>(X);
                ),
                "bounds it by nothing",
            ),
            (
                parse_quote!(
                    #[residency(children(X))]
                    struct Layer<X>(X)
                    where
                        X: Clone;
                ),
                "bounds it by nothing",
            ),
        ] {
            let error = refusal(input);
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn refuses_skipped_fields_and_conflicting_path_attributes() {
        for (input, expected) in [
            (
                parse_quote!(
                    struct Layer {
                        #[residency(keep)]
                        weight: String,
                    }
                ),
                "every field is traversed",
            ),
            (
                parse_quote!(
                    struct Layer {
                        #[residency(skip)]
                        weight: String,
                    }
                ),
                "every field is traversed",
            ),
            (
                parse_quote!(
                    struct Layer {
                        #[residency(transparent, role = R)]
                        weight: W,
                    }
                ),
                "takes one of",
            ),
            (
                parse_quote!(union Layer { first: usize, second: usize }),
                "union has no fields",
            ),
        ] {
            let error = refusal(input);
            assert!(error.contains(expected), "{error}");
        }
    }
}
