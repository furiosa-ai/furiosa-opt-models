use super::Vocabulary;
use crate::models::nemo::serving::{Chips, Rows};
use crate::ops::vocabulary::{Argmax, argmax::sharded};

pub type Greedy = Argmax<Vocabulary, Chips>;
pub type Scores = sharded::Scores<Vocabulary, Rows, Chips>;
pub type Tokens = sharded::Tokens<Rows, Chips>;
