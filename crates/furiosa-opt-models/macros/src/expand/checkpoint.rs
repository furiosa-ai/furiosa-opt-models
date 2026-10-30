//! The checkpoint shapes: a new tree from a checkpoint, and an update of an existing tree in place.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Type, WherePredicate, parse_quote};

use super::{Field, Shape};
use crate::attribute::Read;

impl Shape<'_> {
    /// `T: Load<&Checkpoint>`: every field asked in its scope, the tied ones last, `Missing` of all
    /// of them together.
    pub(super) fn build(&self) -> TokenStream {
        let load = quote!(::furiosa_opt_models::load);
        let this = self.this();
        let mut generics = self.generics(&[quote!('__a)], &[]);
        let predicates = &mut generics.make_where_clause().predicates;
        let reads = |ty: &dyn quote::ToTokens| -> WherePredicate {
            parse_quote!(#ty: #load::Load<&'__a #load::Checkpoint>)
        };
        let converts =
            |ty: &Type, from: &Type| -> WherePredicate { parse_quote!(#ty: #load::Load<#from>) };
        let ignore = self.ignore();
        let body = match &self.declared.from {
            Some(from) => {
                predicates.push(reads(from));
                predicates.push(converts(&parse_quote!(#this), from));
                quote! {
                    #ignore
                    let __parts =
                        <#from as #load::Load<&'__a #load::Checkpoint>>::load(__cx, __from)?;
                    <#this as #load::Load<#from>>::load(__cx, __parts)
                }
            }
            None => {
                for field in &self.fields {
                    match &field.read {
                        Read::Field | Read::Config(_) | Read::Tied(..) => {
                            predicates.push(reads(field.ty))
                        }
                        Read::From(from) => {
                            predicates.extend([reads(from), converts(field.ty, from)])
                        }
                        Read::With(_) => {}
                    }
                }
                let asks = self.visits().map(|field| {
                    let (binding, ty, read) = (&field.binding, field.ty, self.read(field));
                    quote! {
                        let #binding: ::core::option::Option<#ty> = __children.ask(#read)?;
                    }
                });
                let bindings: Vec<_> = self.fields.iter().map(|field| &field.binding).collect();
                let value = self.construct(quote!(Self), |field| {
                    let binding = &field.binding;
                    quote!(#binding)
                });
                quote! {
                    #ignore
                    let mut __children = #load::Children::default();
                    #(#asks)*
                    #[allow(unreachable_patterns)]
                    let __node = match (#(#bindings,)*) {
                        (#(::core::option::Option::Some(#bindings),)*) => {
                            ::core::option::Option::Some(#value)
                        }
                        _ => ::core::option::Option::None,
                    };
                    __children.node(__node)
                }
            }
        };
        let targets = self.targets();
        let visit = self.scoped(quote!(__cx.ties(&[#(#targets),*], |__cx| { #body })));
        let (impl_generics, _, where_clause) = generics.split_for_impl();
        quote! {
            impl #impl_generics #load::Load<&'__a #load::Checkpoint> for #this #where_clause {
                fn load(
                    __context: &mut #load::Context<'_>,
                    __from: &'__a #load::Checkpoint,
                ) -> ::core::result::Result<Self, #load::LoadError> {
                    __context.checkpoint(__from, |__cx| #visit)
                }
            }
        }
    }

    /// `&mut T: Load<(&mut T, &Checkpoint)>`: each field updated in place, tied ones last, a `from`
    /// or `with` field and a `from` struct as one rule; config fields and ties stay as loaded.
    pub(super) fn update(&self) -> TokenStream {
        let load = quote!(::furiosa_opt_models::load);
        let this = self.this();
        let copy = |ty: &Type| -> WherePredicate {
            parse_quote!(for<'__u> &'__u mut #ty: #load::Load<(&'__u mut #ty, #ty)>)
        };
        let reads = |ty: &dyn quote::ToTokens| -> WherePredicate {
            parse_quote!(#ty: #load::Load<&'__s #load::Checkpoint>)
        };
        let write = |field: &Field| {
            let (member, ty) = (&field.member, field.ty);
            quote! {
                |__cx, __new: #ty| {
                    <&mut #ty as #load::Load<_>>::load(__cx, (&mut __into.#member, __new))
                }
            }
        };
        let mut predicates: Vec<WherePredicate> = Vec::new();
        let mut steps = Vec::new();
        match &self.declared.from {
            Some(_) => {
                predicates.push(reads(&this));
                predicates.extend(self.fields.iter().map(|field| copy(field.ty)));
                let pattern = self.destructure();
                let writes = self.fields.iter().map(|field| {
                    let (tree, checkpoint) = (&field.tree, &field.checkpoint);
                    let (member, ty, binding) = (&field.member, field.ty, &field.binding);
                    let into = quote!((&mut __into.#member, #binding));
                    quote! {
                        __cx.field(#tree, #checkpoint, |__cx| {
                            <&mut #ty as #load::Load<_>>::load(__cx, #into)
                        })?;
                    }
                });
                steps.push(quote! {
                    __cx.rule(
                        |__cx| <#this as #load::Load<&'__s #load::Checkpoint>>::load(__cx, __from),
                        |__cx, __new: #this| {
                            let #pattern = __new;
                            #(#writes)*
                            ::core::result::Result::Ok(())
                        },
                    )?;
                });
            }
            None => {
                for field in self.visits() {
                    let (tree, checkpoint, member, ty) =
                        (&field.tree, &field.checkpoint, &field.member, field.ty);
                    let in_place = quote! {
                        <&mut #ty as #load::Load<_>>::load(__cx, (&mut __into.#member, __from))
                    };
                    let bound: WherePredicate = parse_quote! {
                        for<'__u> &'__u mut #ty:
                            #load::Load<(&'__u mut #ty, &'__s #load::Checkpoint)>
                    };
                    let own = quote!(__cx.field(#tree, #checkpoint, |__cx| #in_place)?;);
                    match &field.read {
                        Read::Config(_) => {}
                        Read::Field => {
                            predicates.push(bound);
                            steps.push(own);
                        }
                        Read::Tied(target, key) => {
                            predicates.push(bound);
                            let tied = quote! {
                                __cx.tied(#tree, #checkpoint, #target, |__cx| #in_place)?;
                            };
                            steps.push(match key {
                                None => tied,
                                Some(_) => {
                                    let decided = &self.tied;
                                    quote!(match #decided { true => { #tied } false => { #own } })
                                }
                            });
                        }
                        Read::With(with) => {
                            predicates.push(copy(ty));
                            let write = write(field);
                            steps.push(quote! {
                                __cx.field(#tree, #checkpoint, |__cx| {
                                    __cx.rule(|__cx| #with(__from, __cx), #write)
                                })?;
                            });
                        }
                        Read::From(from) => {
                            let converts: WherePredicate = parse_quote!(#ty: #load::Load<#from>);
                            predicates.extend([reads(from), converts, copy(ty)]);
                            let write = write(field);
                            steps.push(quote! {
                                __cx.field(#tree, #checkpoint, |__cx| __cx.rule(
                                    |__cx| {
                                        let __parts = <#from as #load::Load<
                                            &'__s #load::Checkpoint,
                                        >>::load(__cx, __from)?;
                                        <#ty as #load::Load<#from>>::load(__cx, __parts)
                                    },
                                    #write,
                                ))?;
                            });
                        }
                    }
                }
            }
        }
        let mut generics = self.generics(&[quote!('__t), quote!('__s)], &[]);
        generics.make_where_clause().predicates.extend(predicates);
        let (impl_generics, _, where_clause) = generics.split_for_impl();
        let ignore = self.ignore();
        let targets = self.targets();
        let visit = self.scoped(quote! {
            __cx.ties(&[#(#targets),*], |__cx| {
                #ignore
                #(#steps)*
                ::core::result::Result::Ok(())
            })
        });
        quote! {
            impl #impl_generics #load::Load<(&'__t mut #this, &'__s #load::Checkpoint)>
            for &'__t mut #this
            #where_clause
            {
                fn load(
                    __context: &mut #load::Context<'_>,
                    (__into, __from): (&'__t mut #this, &'__s #load::Checkpoint),
                ) -> ::core::result::Result<Self, #load::LoadError> {
                    __context.checkpoint(__from, |__cx| #visit)?;
                    ::core::result::Result::Ok(__into)
                }
            }
        }
    }

    /// The expression reading `field` in the checkpoint build.
    fn read(&self, field: &Field) -> TokenStream {
        let load = quote!(::furiosa_opt_models::load);
        let (tree, checkpoint, ty) = (&field.tree, &field.checkpoint, field.ty);
        let read = quote!(<#ty as #load::Load<&'__a #load::Checkpoint>>::load(__cx, __from));
        let own = quote!(__cx.field(#tree, #checkpoint, |__cx| #read));
        match &field.read {
            Read::Field => own,
            Read::Config(key) => {
                quote!(__cx.field(#tree, #checkpoint, |__cx| __cx.config(#key, |__cx| #read)))
            }
            Read::Tied(target, None) => {
                quote!(__cx.tied(#tree, #checkpoint, #target, |__cx| #read))
            }
            Read::Tied(target, Some(key)) => {
                let tied = &self.tied;
                quote! {
                    __cx.tying(__from, #key, #tied).and_then(|()| match #tied {
                        true => __cx.tied(#tree, #checkpoint, #target, |__cx| #read),
                        false => #own,
                    })
                }
            }
            Read::From(from) => quote! {
                __cx.field(#tree, #checkpoint, |__cx| {
                    let __parts =
                        <#from as #load::Load<&'__a #load::Checkpoint>>::load(__cx, __from)?;
                    <#ty as #load::Load<#from>>::load(__cx, __parts)
                })
            },
            Read::With(with) => quote!(__cx.field(#tree, #checkpoint, |__cx| #with(__from, __cx))),
        }
    }
}
