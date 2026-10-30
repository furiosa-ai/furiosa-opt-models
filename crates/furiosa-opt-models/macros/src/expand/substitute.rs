//! The parameters a tree build converts, renamed to fresh counterparts in the types that name them.

use proc_macro2::{Group, Ident, TokenStream, TokenTree};
use quote::{ToTokens, format_ident};
use syn::visit_mut::VisitMut;
use syn::{GenericParam, Generics, Type, WherePredicate};

/// Fresh counterparts of the parameters a tree build converts: those bounded by `Memory` and the
/// unbounded ones (generic children); the struct type with them; their bounds.
pub(super) struct Counterparts {
    map: Vec<(Ident, Ident)>,
    params: Vec<TokenStream>,
    predicates: Vec<WherePredicate>,
}

impl Counterparts {
    pub(super) fn of(generics: &Generics) -> Self {
        let memory = |bound: &syn::TypeParamBound| match bound {
            syn::TypeParamBound::Trait(t) => {
                t.path.segments.last().is_some_and(|s| s.ident == "Memory")
            }
            _ => false,
        };
        let declared = |param: &syn::TypeParam| -> Vec<syn::TypeParamBound> {
            let predicates = generics.where_clause.iter().flat_map(|w| &w.predicates);
            predicates
                .filter_map(|predicate| match predicate {
                    WherePredicate::Type(t) => Some(t),
                    _ => None,
                })
                .filter(|t| matches!(&t.bounded_ty, Type::Path(p) if p.path.is_ident(&param.ident)))
                .flat_map(|t| t.bounds.iter().cloned())
                .collect()
        };
        let converted: Vec<&syn::TypeParam> = generics
            .type_params()
            .filter(|param| {
                let bounds: Vec<_> = param
                    .bounds
                    .iter()
                    .cloned()
                    .chain(declared(param))
                    .collect();
                bounds.is_empty() || bounds.iter().any(memory)
            })
            .collect();
        let map: Vec<(Ident, Ident)> = converted
            .iter()
            .enumerate()
            .map(|(i, param)| (param.ident.clone(), format_ident!("__Y{i}")))
            .collect();
        let mut substitute = Substitute(&map);
        let params = converted
            .iter()
            .zip(&map)
            .map(|(param, (_, fresh))| {
                let mut param = (*param).clone();
                param.ident = fresh.clone();
                param.eq_token = None;
                param.default = None;
                param
                    .bounds
                    .iter_mut()
                    .for_each(|bound| substitute.visit_type_param_bound_mut(bound));
                param.to_token_stream()
            })
            .collect();
        let predicates = generics
            .where_clause
            .iter()
            .flat_map(|w| &w.predicates)
            .filter_map(|predicate| {
                let mut converted = predicate.clone();
                substitute.visit_where_predicate_mut(&mut converted);
                let changed = converted.to_token_stream().to_string()
                    != predicate.to_token_stream().to_string();
                changed.then_some(converted)
            })
            .collect();
        Self {
            map,
            params,
            predicates,
        }
    }

    /// The counterparts, declared with the bounds of the parameters they stand for.
    pub(super) fn params(&self) -> &[TokenStream] {
        &self.params
    }

    /// The struct's where-clause predicates that name a converted parameter, renamed.
    pub(super) fn predicates(&self) -> &[WherePredicate] {
        &self.predicates
    }

    pub(super) fn convert(&self, ty: &Type) -> Type {
        let mut ty = ty.clone();
        Substitute(&self.map).visit_type_mut(&mut ty);
        ty
    }

    /// The struct's arguments with the counterparts in place of the converted parameters.
    pub(super) fn arguments(&self, generics: &Generics) -> Vec<TokenStream> {
        generics
            .params
            .iter()
            .map(|param| match param {
                GenericParam::Lifetime(l) => l.lifetime.to_token_stream(),
                GenericParam::Type(t) => {
                    let fresh = self
                        .map
                        .iter()
                        .find(|(from, _)| *from == t.ident)
                        .map(|(_, to)| to);
                    fresh.unwrap_or(&t.ident).to_token_stream()
                }
                GenericParam::Const(c) => c.ident.to_token_stream(),
            })
            .collect()
    }
}

/// Renames type parameters in paths and in macro arguments (`m![{ C }, H]`).
struct Substitute<'a>(&'a [(Ident, Ident)]);

impl Substitute<'_> {
    fn tokens(&self, tokens: TokenStream) -> TokenStream {
        tokens
            .into_iter()
            .map(|token| match token {
                TokenTree::Ident(ident) => {
                    let fresh = self
                        .0
                        .iter()
                        .find(|(from, _)| *from == ident)
                        .map(|(_, to)| to.clone());
                    TokenTree::Ident(fresh.unwrap_or(ident))
                }
                TokenTree::Group(group) => {
                    let mut renamed = Group::new(group.delimiter(), self.tokens(group.stream()));
                    renamed.set_span(group.span());
                    TokenTree::Group(renamed)
                }
                other => other,
            })
            .collect()
    }
}

impl VisitMut for Substitute<'_> {
    fn visit_path_mut(&mut self, path: &mut syn::Path) {
        if path.leading_colon.is_none()
            && let Some(first) = path.segments.first_mut()
            && let Some((_, fresh)) = self.0.iter().find(|(from, _)| first.ident == *from)
        {
            first.ident = fresh.clone();
        }
        syn::visit_mut::visit_path_mut(self, path);
    }

    fn visit_macro_mut(&mut self, mac: &mut syn::Macro) {
        mac.tokens = self.tokens(std::mem::take(&mut mac.tokens));
    }
}
