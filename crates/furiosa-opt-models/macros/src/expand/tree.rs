//! The tree shapes: a tree built from a borrowed tree of another memory, and a tree written in
//! place from a value of its own type.

use proc_macro2::TokenStream;
use quote::quote;
use syn::parse_quote;

use super::Shape;
use super::substitute::Counterparts;

impl Shape<'_> {
    /// `T<X..>: Load<&T<Y..>>`, field-wise, the converted parameters renamed in each field's type.
    pub(super) fn borrow(&self) -> TokenStream {
        let load = quote!(::furiosa_opt_models::load);
        let this = self.this();
        let name = self.name;
        let counterparts = Counterparts::of(self.generics);
        let arguments = counterparts.arguments(self.generics);
        let from = quote!(#name<#(#arguments),*>);
        let mut generics = self.generics(&[quote!('__a)], counterparts.params());
        let predicates = &mut generics.make_where_clause().predicates;
        predicates.extend(counterparts.predicates().iter().cloned());
        for field in &self.fields {
            let (ty, converted) = (field.ty, counterparts.convert(field.ty));
            predicates.push(parse_quote!(#ty: #load::Load<&'__a #converted>));
        }
        let value = self.construct(quote!(Self), |field| {
            let (tree, checkpoint, member, ty) =
                (&field.tree, &field.checkpoint, &field.member, field.ty);
            let converted = counterparts.convert(ty);
            quote! {
                __context.field(#tree, #checkpoint, |__cx| {
                    <#ty as #load::Load<&'__a #converted>>::load(__cx, &__from.#member)
                })?
            }
        });
        let (impl_generics, _, where_clause) = generics.split_for_impl();
        quote! {
            impl #impl_generics #load::Load<&'__a #from> for #this #where_clause {
                fn load(
                    __context: &mut #load::Context<'_>,
                    __from: &'__a #from,
                ) -> ::core::result::Result<Self, #load::LoadError> {
                    ::core::result::Result::Ok(#value)
                }
            }
        }
    }

    /// `&mut T: Load<(&mut T, T)>`: every field written in place from the value's.
    pub(super) fn copy(&self) -> TokenStream {
        let load = quote!(::furiosa_opt_models::load);
        let this = self.this();
        let mut generics = self.generics(&[quote!('__t)], &[]);
        let predicates = &mut generics.make_where_clause().predicates;
        for field in &self.fields {
            let ty = field.ty;
            predicates.push(parse_quote! {
                for<'__u> &'__u mut #ty: #load::Load<(&'__u mut #ty, #ty)>
            });
        }
        let pattern = self.destructure();
        let writes = self.fields.iter().map(|field| {
            let (tree, checkpoint, member) = (&field.tree, &field.checkpoint, &field.member);
            let (ty, binding) = (field.ty, &field.binding);
            quote! {
                __context.field(#tree, #checkpoint, |__cx| {
                    <&mut #ty as #load::Load<_>>::load(__cx, (&mut __into.#member, #binding))
                })?;
            }
        });
        let (impl_generics, _, where_clause) = generics.split_for_impl();
        quote! {
            impl #impl_generics #load::Load<(&'__t mut #this, #this)> for &'__t mut #this
            #where_clause
            {
                fn load(
                    __context: &mut #load::Context<'_>,
                    (__into, __from): (&'__t mut #this, #this),
                ) -> ::core::result::Result<Self, #load::LoadError> {
                    let #pattern = __from;
                    #(#writes)*
                    ::core::result::Result::Ok(__into)
                }
            }
        }
    }
}
