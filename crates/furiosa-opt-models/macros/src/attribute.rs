//! The `#[load(..)]` attributes of a struct and of its fields.

use proc_macro2::Span;
use syn::{Attribute, Error, LitStr, Path, Result, Type};

/// How a field is read from a checkpoint.
pub(crate) enum Read {
    /// Its own subtree, under its checkpoint segment.
    Field,
    /// The config value `key` instead of a tensor.
    Config(LitStr),
    /// The subtree at another path of the struct, its stored copy under the field's name compared
    /// when present; with a config key, the geometry decides the tie and the key is checked.
    Tied(LitStr, Option<LitStr>),
    /// A checkpoint-shaped type, converted by `Field: Load<T>`.
    From(Box<Type>),
    /// A function of the source and the context.
    With(Path),
}

/// What a field declares.
pub(crate) struct Field {
    pub(crate) rename: Option<LitStr>,
    pub(crate) read: Read,
}

/// What a struct declares.
pub(crate) struct Struct {
    pub(crate) from: Option<Box<Type>>,
    pub(crate) ignore: Vec<LitStr>,
    /// The config key its fields read their config under.
    pub(crate) config: Option<LitStr>,
    /// Whether only the tree directions are derived, the struct's own impls reading the checkpoint.
    pub(crate) tree: bool,
}

/// One `key = value` (or bare `tree`) of a `#[load(..)]` list.
enum Entry {
    Rename(LitStr),
    From(Box<Type>),
    With(Path),
    Config(LitStr),
    Tied(LitStr),
    Ignore(LitStr),
    Tree,
}

impl Entry {
    fn key(&self) -> &'static str {
        match self {
            Self::Rename(_) => "rename",
            Self::From(_) => "from",
            Self::With(_) => "with",
            Self::Config(_) => "config",
            Self::Tied(_) => "tied",
            Self::Ignore(_) => "ignore",
            Self::Tree => "tree",
        }
    }

    fn all(attrs: &[Attribute]) -> Result<Vec<(Span, Self)>> {
        let mut entries = Vec::new();
        for attr in attrs.iter().filter(|attr| attr.path().is_ident("load")) {
            attr.parse_nested_meta(|meta| {
                let key = meta
                    .path
                    .get_ident()
                    .map(ToString::to_string)
                    .unwrap_or_default();
                let entry = match key.as_str() {
                    "rename" => Self::Rename(meta.value()?.parse()?),
                    "from" => Self::From(Box::new(meta.value()?.parse()?)),
                    "with" => Self::With(meta.value()?.parse()?),
                    "config" => Self::Config(meta.value()?.parse()?),
                    "tied" => Self::Tied(meta.value()?.parse()?),
                    "ignore" => Self::Ignore(meta.value()?.parse()?),
                    "tree" => Self::Tree,
                    _ => return Err(meta.error(format!("unknown `load` attribute `{key}`"))),
                };
                entries.push((
                    meta.path
                        .get_ident()
                        .map_or_else(Span::call_site, |i| i.span()),
                    entry,
                ));
                Ok(())
            })?;
        }
        let mut seen = Vec::new();
        for (span, entry) in &entries {
            let key = entry.key();
            if key != "ignore" && seen.contains(&key) {
                return Err(Error::new(*span, format!("duplicate `{key}`")));
            }
            seen.push(key);
        }
        Ok(entries)
    }
}

impl Field {
    pub(crate) fn parse(attrs: &[Attribute]) -> Result<Self> {
        let mut rename = None;
        let mut reads = Vec::new();
        for (span, entry) in Entry::all(attrs)? {
            match entry {
                Entry::Rename(name) => rename = Some(name),
                Entry::Ignore(_) => return Err(Error::new(span, "`ignore` applies to the struct")),
                Entry::Tree => return Err(Error::new(span, "`tree` applies to the struct")),
                Entry::Config(key) if key.value().is_empty() => {
                    return Err(Error::new(key.span(), "`config` names no config key"));
                }
                Entry::Config(key) => reads.push((span, Read::Config(key))),
                Entry::Tied(path) if path.value().split('.').any(str::is_empty) => {
                    return Err(Error::new(
                        path.span(),
                        "`tied` names a subtree below the struct, as `a.b`",
                    ));
                }
                Entry::Tied(path) => reads.push((span, Read::Tied(path, None))),
                Entry::From(ty) => reads.push((span, Read::From(ty))),
                Entry::With(path) => reads.push((span, Read::With(path))),
            }
        }
        let mut reads = reads.into_iter();
        let read = match (reads.next(), reads.next(), reads.next()) {
            (None, _, _) => Read::Field,
            (Some((_, read)), None, _) => read,
            (Some((_, Read::Tied(path, None))), Some((_, Read::Config(key))), None)
            | (Some((_, Read::Config(key))), Some((_, Read::Tied(path, None))), None) => {
                Read::Tied(path, Some(key))
            }
            (_, Some((span, _)), _) => {
                return Err(Error::new(
                    span,
                    "`from`, `with`, `config` and `tied` exclude each other, but for `tied` with \
                     `config`",
                ));
            }
        };
        Ok(Self { rename, read })
    }
}

impl Struct {
    pub(crate) fn parse(attrs: &[Attribute]) -> Result<Self> {
        let mut parsed = Self {
            from: None,
            ignore: Vec::new(),
            config: None,
            tree: false,
        };
        let entries = Entry::all(attrs)?;
        let tree = entries
            .iter()
            .any(|(_, entry)| matches!(entry, Entry::Tree));
        for (span, entry) in entries {
            match entry {
                Entry::Tree => parsed.tree = true,
                Entry::From(_) | Entry::Ignore(_) | Entry::Config(_) if tree => {
                    let key = entry.key();
                    return Err(Error::new(
                        span,
                        format!("a `tree` struct reads the checkpoint itself, so takes no `{key}`"),
                    ));
                }
                Entry::From(ty) => parsed.from = Some(ty),
                Entry::Ignore(prefix) if prefix.value().is_empty() => {
                    return Err(Error::new(prefix.span(), "`ignore` names no prefix"));
                }
                Entry::Ignore(prefix) => parsed.ignore.push(prefix),
                Entry::Config(key) if key.value().is_empty() => {
                    return Err(Error::new(key.span(), "`config` names no config key"));
                }
                Entry::Config(key) => parsed.config = Some(key),
                other => {
                    let key = other.key();
                    return Err(Error::new(span, format!("`{key}` applies to fields")));
                }
            }
        }
        Ok(parsed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    fn field(attrs: Vec<Attribute>) -> String {
        Field::parse(&attrs)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default()
    }

    fn structure(attrs: Vec<Attribute>) -> String {
        Struct::parse(&attrs)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default()
    }

    /// Every attribute a field takes parses into what it declares.
    #[test]
    fn parses_each_field_attribute() {
        let parsed = Field::parse(&[parse_quote!(#[load(rename = "q_norm")])]).unwrap();
        assert_eq!(parsed.rename.map(|s| s.value()), Some("q_norm".into()));
        let read = |attr: Attribute| Field::parse(&[attr]).unwrap().read;
        assert!(
            matches!(read(parse_quote!(#[load(config = "e")])), Read::Config(k) if k.value() == "e")
        );
        let tied = read(parse_quote!(#[load(tied = "a.b")]));
        assert!(matches!(tied, Read::Tied(p, None) if p.value() == "a.b"));
        assert!(matches!(
            read(parse_quote!(#[load(from = Parts<X>)])),
            Read::From(_)
        ));
        assert!(matches!(
            read(parse_quote!(#[load(with = scales)])),
            Read::With(_)
        ));
        let both = Field::parse(&[parse_quote!(#[load(rename = "", tied = "a")])]).unwrap();
        assert!(both.rename.is_some() && matches!(both.read, Read::Tied(_, _)));
    }

    /// A tied field may name the config value checked against its type's tying constant.
    #[test]
    fn accepts_checked_tying() {
        assert!(
            Field::parse(&[parse_quote!(
                #[load(tied = "model.embed_tokens", config = "tie_word_embeddings")]
            )])
            .is_ok()
        );
    }

    /// A struct takes `from` and `config` once and any number of `ignore` prefixes.
    #[test]
    fn parses_struct_attributes() {
        let parsed = Struct::parse(&[
            parse_quote!(#[load(from = Parts<X>, ignore = "mtp")]),
            parse_quote!(#[load(ignore = "vision", config = "text_config")]),
        ])
        .unwrap();
        assert!(parsed.from.is_some());
        let prefixes: Vec<String> = parsed.ignore.iter().map(LitStr::value).collect();
        assert_eq!(prefixes, ["mtp", "vision"]);
        assert_eq!(
            parsed.config.map(|key| key.value()),
            Some("text_config".into())
        );
    }

    /// An unknown key, a repeated key and a value of the wrong kind are refused by name.
    #[test]
    fn refuses_unknown_repeated_and_mistyped_keys() {
        assert!(
            field(vec![parse_quote!(#[load(name = "x")])])
                .contains("unknown `load` attribute `name`")
        );
        assert!(
            field(vec![parse_quote!(#[load(rename = "a", rename = "b")])])
                .contains("duplicate `rename`")
        );
        assert!(
            field(vec![
                parse_quote!(#[load(rename = "a")]),
                parse_quote!(#[load(rename = "b")])
            ])
            .contains("duplicate `rename`")
        );
        assert!(field(vec![parse_quote!(#[load(rename = 1)])]).contains("expected string literal"));
        assert!(
            structure(vec![
                parse_quote!(#[load(ignore = "a", from = A, from = B)])
            ])
            .contains("duplicate `from`")
        );
    }

    /// A `tree` struct reads the checkpoint itself: no struct attribute of a checkpoint read.
    #[test]
    fn parses_a_tree_struct_without_checkpoint_attributes() {
        assert!(Struct::parse(&[parse_quote!(#[load(tree)])]).unwrap().tree);
        for attr in [
            parse_quote!(#[load(tree, from = A)]),
            parse_quote!(#[load(ignore = "a", tree)]),
            parse_quote!(#[load(tree, config = "c")]),
        ] {
            assert!(structure(vec![attr]).contains("reads the checkpoint itself"));
        }
    }

    /// `ignore` and `tree` belong to the struct; the field attributes belong to fields.
    #[test]
    fn refuses_attributes_at_the_wrong_level() {
        assert!(
            field(vec![parse_quote!(#[load(ignore = "mtp")])])
                .contains("`ignore` applies to the struct")
        );
        assert!(field(vec![parse_quote!(#[load(tree)])]).contains("`tree` applies to the struct"));
        for attr in [
            parse_quote!(#[load(rename = "a")]),
            parse_quote!(#[load(tied = "a")]),
            parse_quote!(#[load(with = f)]),
        ] {
            assert!(structure(vec![attr]).contains("applies to fields"));
        }
    }

    /// A field is read one way: `from`, `with`, `config` and `tied` exclude each other.
    #[test]
    fn refuses_two_ways_of_reading_a_field() {
        for attr in [
            parse_quote!(#[load(from = A, with = f)]),
            parse_quote!(#[load(with = f, config = "eps")]),
            parse_quote!(#[load(from = A, tied = "a")]),
            parse_quote!(#[load(config = "eps", from = A)]),
        ] {
            assert!(field(vec![attr]).contains("exclude each other"));
        }
    }

    /// A tied path names a subtree below the struct; a config key and an ignored prefix are named.
    #[test]
    fn refuses_paths_outside_the_struct_and_empty_names() {
        for path in ["", ".a", "a.", "a..b", "..a"] {
            let attr: Attribute = parse_quote!(#[load(tied = #path)]);
            assert!(field(vec![attr]).contains("below the struct"), "{path}");
        }
        assert!(field(vec![parse_quote!(#[load(config = "")])]).contains("names no config key"));
        assert!(
            structure(vec![parse_quote!(#[load(config = "")])]).contains("names no config key")
        );
        assert!(structure(vec![parse_quote!(#[load(ignore = "")])]).contains("names no prefix"));
    }
}
