//! Where a weight sits in its tree, built while a pass descends and written out only when the pass
//! refuses a weight.

use std::fmt;

/// The fields, roles and indices from the root to one node, outermost first.
#[derive(Clone, Copy)]
pub struct Path<'a> {
    parent: Option<&'a Path<'a>>,
    segment: Segment<'a>,
}

#[derive(Clone, Copy)]
enum Segment<'a> {
    Root,
    Field(&'a str),
    Index(usize),
    /// A role's type name, written as the role's snake-case name.
    Role(&'static str),
}

impl<'a> Path<'a> {
    /// The tree itself.
    pub const ROOT: Path<'static> = Path {
        parent: None,
        segment: Segment::Root,
    };

    /// The child in field `name`.
    pub fn field(&'a self, name: &'a str) -> Self {
        self.child(Segment::Field(name))
    }

    /// The child at `index`.
    pub fn index(&'a self, index: usize) -> Self {
        self.child(Segment::Index(index))
    }

    /// The child in role `R`.
    pub fn role<R>(&'a self) -> Self {
        self.child(Segment::Role(std::any::type_name::<R>()))
    }

    fn child(&'a self, segment: Segment<'a>) -> Self {
        Self {
            parent: Some(self),
            segment,
        }
    }
}

/// Segments joined by `.`, a role by the last name of its type: `layers.3.attention_norm.weight`.
impl fmt::Display for Path<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(parent) = self.parent {
            write!(formatter, "{parent}")?;
            if parent.parent.is_some() {
                formatter.write_str(".")?;
            }
        }
        match self.segment {
            Segment::Root => Ok(()),
            Segment::Field(name) => formatter.write_str(name),
            Segment::Index(index) => write!(formatter, "{index}"),
            Segment::Role(name) => {
                // A role pair such as `(Feedforward, FeedforwardTree)` names its first role.
                let first = name
                    .trim_start_matches('(')
                    .split(',')
                    .next()
                    .unwrap_or(name);
                let last = first.rsplit("::").next().unwrap_or(first);
                last.chars().enumerate().try_for_each(|(at, letter)| {
                    if letter.is_uppercase() && at > 0 {
                        formatter.write_str("_")?;
                    }
                    write!(formatter, "{}", letter.to_ascii_lowercase())
                })
            }
        }
    }
}
