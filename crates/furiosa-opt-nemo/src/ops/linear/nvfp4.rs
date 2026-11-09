//! NVFP4 projections whose input rows quantize as their weights do.
//!
//! - A weight row holds E2M1 codes with one E4M3 scale per 16 inputs, under one FP32 tensor scale.
//! - Each call quantizes its input row the same way: a 16-input block's E4M3 scale comes from its
//!   largest magnitude under the input's FP32 tensor scale, and its codes round to nearest even,
//!   bit for bit as the reference quantizer `ref_nvfp4_quant` does.
//! - Both operands enter the contraction as code times block scale, which BF16 holds exactly, and
//!   the FP32 sum takes both tensor scales once, as a block-scaled FP4 product does.
//!
//! Source: <https://github.com/vllm-project/vllm/blob/ee0da84ab9e04ac7610e28580af62c365e898389/vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py>.

use furiosa_opt_std::prelude::*;

// The 16 inputs one E4M3 scale covers, apart from the block they belong to, and the four output
// rows a slice decodes at a time, apart from its block of rows.
axes![Group = 16, Row = 4];
