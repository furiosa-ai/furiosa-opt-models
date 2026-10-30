//! The registry: every upstream architecture name once, by family level, with the decoders and
//! media encoders that implement it, or none when the name is refused.
//!
//! - `ARCHITECTURES` lists each name, and each alias as a name of its own, with its
//!   implementations: a decoder or media encoder type at one size each.
//! - `Implementation::listed` finds the first of some architecture names that the list holds, with
//!   its implementations, and `choose` picks the one whose scope serves a checkpoint, from its
//!   config and names alone; its `load` then reads the checkpoint strictly at that scope into HBM,
//!   as the `Model` a framework serves.
//! - A decoder reads a whole model, a media encoder a load of the encoder part alone; each refuses
//!   a checkpoint read for the other part.
//! - A refused name is `Unsupported`, an unknown one `InvalidData`.

use crate::decoder::{Decoder, Instance, Served};
use crate::vision_language::Encoder;
use furiosa_opt_models::load::{Checkpoint, Context, Load, LoadError, Part};
use furiosa_opt_models::models::decoder::attention::{llama, qwen3};
use furiosa_opt_models::models::decoder::form;
use furiosa_opt_models::models::vision_language::gemma4_unified;
use furiosa_opt_models::models::{Class, Implements};
use furiosa_opt_models::ops::attention::full::paged::Tiling;
use furiosa_opt_models::tensor::cache;
use furiosa_opt_models::tensor::memory::Hbm;
use furiosa_opt_models::tensor::parallel::Parallel;

/// Every upstream architecture name, by family level, with the decoders that implement it; a
/// refused name has none.
pub const ARCHITECTURES: &[(&str, &[Implementation])] = &[
    // # Attention decoders
    ("AXK1ForCausalLM", &[]),
    ("AfmoeForCausalLM", &[]),
    ("ApertusForCausalLM", &[]),
    ("ArceeForCausalLM", &[]),
    ("Audio8TTSSlowARForConditionalGeneration", &[]),
    ("BaichuanForCausalLM", &[]),
    ("BailingMoEForCausalLM", &[]),
    ("BailingMoeForCausalLM", &[]),
    ("BailingMoeForCausalLMNextN", &[]),
    ("BailingMoeV25MTPModel", &[]),
    ("BailingMoeV2ForCausalLM", &[]),
    ("BailingMoeV2_5ForCausalLM", &[]),
    ("BailingMoeV3MTPModel", &[]),
    ("BloomForCausalLM", &[]),
    ("BreezeForConditionalGeneration", &[]),
    ("ChatGLMForConditionalGeneration", &[]),
    ("ChatGLMModel", &[]),
    ("Cohere2ForCausalLM", &[]),
    ("Cohere2MoeForCausalLM", &[]),
    ("CohereForCausalLM", &[]),
    ("CwmForCausalLM", &[]),
    ("DFlash2DraftModel", &[]),
    ("DFlashDraftModel", &[]),
    ("DFlashLagunaForCausalLM", &[]),
    ("DFlashMuseGlimmerAssistantModel", &[]),
    ("DSparkDraftModel", &[]),
    ("DSparkV41DraftModel", &[]),
    ("DbrxForCausalLM", &[]),
    ("DeciLMForCausalLM", &[]),
    ("DeepSeekMTPModel", &[]),
    ("DeepSeekV4MTPModel", &[]),
    ("DeepseekForCausalLM", &[]),
    ("DeepseekV2ForCausalLM", &[]),
    ("DeepseekV32ForCausalLM", &[]),
    ("DeepseekV32MTPModel", &[]),
    ("DeepseekV3BidirectionalModel", &[]),
    ("DeepseekV3ForCausalLM", &[]),
    ("DeepseekV3ForCausalLMNextN", &[]),
    ("DeepseekV4ForCausalLM", &[]),
    ("DeepseekV4ForCausalLMDSpark", &[]),
    ("DeepseekV4ForCausalLMNextN", &[]),
    ("DiffusionGemmaForBlockDiffusion", &[]),
    ("Dots3NoteForCausalLMNextN", &[]),
    ("Dots3NoteMTPModel", &[]),
    ("Eagle3DeepseekV2ForCausalLM", &[]),
    ("Eagle3DeepseekV3ForCausalLM", &[]),
    ("Eagle3LlamaForCausalLM", &[]),
    ("Eagle3MiniMaxM2ForCausalLM", &[]),
    ("Eagle3Qwen2_5vlForCausalLM", &[]),
    ("Eagle3Qwen3ForCausalLM", &[]),
    ("Eagle3Qwen3vlForCausalLM", &[]),
    ("EagleCohereForCausalLM", &[]),
    ("EagleDeepSeekMTPModel", &[]),
    ("EagleLlama4ForCausalLM", &[]),
    ("EagleLlamaForCausalLM", &[]),
    ("EagleMiniCPMForCausalLM", &[]),
    ("EagleMistralForCausalLM", &[]),
    ("EagleMistralLarge3ForCausalLM", &[]),
    ("EmbeddingGemmaModel", &[]),
    ("Ernie4_5ForCausalLM", &[]),
    ("Ernie4_5_ForCausalLM", &[]),
    ("Ernie4_5_MoeForCausalLM", &[]),
    ("Ernie4_5_MoeForCausalLMMTP", &[]),
    ("ErnieMTPModel", &[]),
    ("Exaone4ForCausalLM", &[]),
    ("Exaone4_5_MTP", &[]),
    ("ExaoneForCausalLM", &[]),
    ("ExaoneMoEForCausalLM", &[]),
    ("ExaoneMoEForCausalLMMTP", &[]),
    ("ExaoneMoeForCausalLM", &[]),
    ("ExaoneMoeMTP", &[]),
    ("FalconForCausalLM", &[]),
    // Source: <https://github.com/vllm-project/vllm-omni/blob/6daf5b30f07cc0b3db87766d22fdf433f8a1a7c9/vllm_omni/model_executor/models/fish_speech/fish_speech_slow_ar.py>.
    ("FishSpeechFastAR", &[]),
    ("FishSpeechSlowARForConditionalGeneration", &[]),
    ("FlexOlmoForCausalLM", &[]),
    ("GPT2ForSequenceClassification", &[]),
    ("GPT2LMHeadModel", &[]),
    ("GPTBigCodeForCausalLM", &[]),
    ("GPTJForCausalLM", &[]),
    ("GPTNeoXForCausalLM", &[]),
    ("Gemma2ForCausalLM", &[]),
    ("Gemma2ForSequenceClassification", &[]),
    // Source: <https://github.com/vllm-project/vllm/blob/153242a314153637999eb6ebe8dd830e63433bb6/vllm/model_executor/models/gemma2.py>.
    ("Gemma2Model", &[]),
    ("Gemma3ForCausalLM", &[]),
    ("Gemma3TextModel", &[]),
    ("Gemma3nForCausalLM", &[]),
    ("Gemma4AssistantForCausalLM", &[]),
    ("Gemma4DSparkModel", &[]),
    ("Gemma4ForCausalLM", &[]),
    ("Gemma4ForConditionalGeneration", &[]),
    ("Gemma4MTPModel", &[]),
    ("Gemma4UnifiedAssistantForCausalLM", &[]),
    ("GemmaForCausalLM", &[]),
    ("GepardTalkerForConditionalGeneration", &[]),
    ("GigaChat35ForCausalLM", &[]),
    ("GigaChat35ForCausalLMNextN", &[]),
    ("Glm4ForCausalLM", &[]),
    ("Glm4MoeForCausalLM", &[]),
    ("Glm4MoeForCausalLMNextN", &[]),
    ("Glm4MoeLiteForCausalLM", &[]),
    ("Glm4MoeLiteForCausalLMNextN", &[]),
    ("Glm4MoeLiteMTPModel", &[]),
    ("Glm4MoeMTPModel", &[]),
    ("GlmForCausalLM", &[]),
    ("GlmMoeDsaForCausalLM", &[]),
    ("GlmMoeDsaForCausalLMNextN", &[]),
    ("GlmOcrForConditionalGenerationNextN", &[]),
    ("GlmOcrMTPModel", &[]),
    ("GptOssForCausalLM", &[]),
    ("GraniteForCausalLM", &[]),
    ("GraniteMoeForCausalLM", &[]),
    ("GraniteMoeSWAForCausalLM", &[]),
    ("GraniteMoeSharedForCausalLM", &[]),
    ("GraniteSWAForCausalLM", &[]),
    ("Grok1ForCausalLM", &[]),
    ("Grok1ModelForCausalLM", &[]),
    ("HYV3ForCausalLM", &[]),
    ("HYV3ForCausalLMNextN", &[]),
    ("HYV3MTPModel", &[]),
    ("HYV4ForCausalLM", &[]),
    ("HYV4ForCausalLMNextN", &[]),
    ("HYV4MTPModel", &[]),
    ("HiggsAudioV2ForConditionalGeneration", &[]),
    ("HiggsAudioV2TalkerForConditionalGeneration", &[]),
    ("HiggsAudioV3TalkerForConditionalGeneration", &[]),
    ("HiggsMultimodalQwen3ForConditionalGeneration", &[]),
    ("HrmTextForCausalLM", &[]),
    ("HunYuanDenseV1ForCausalLM", &[]),
    ("HunYuanMoEV1ForCausalLM", &[]),
    ("HyperCLOVAXForCausalLM", &[]),
    ("IQuestCoderForCausalLM", &[]),
    ("IQuestLoopCoderForCausalLM", &[]),
    ("IndexTTS25TalkerForConditionalGeneration", &[]),
    ("IndexTTS2TalkerForConditionalGeneration", &[]),
    ("InternLM2ForCausalLM", &[]),
    ("InternLM2ForRewardModel", &[]),
    ("InternLM3ForCausalLM", &[]),
    ("Jais2ForCausalLM", &[]),
    ("JetNemotronForCausalLM", &[]),
    ("JinaEmbeddingsV5Model", &[]),
    ("K2HorizonForCausalLM", &[]),
    ("KimiK3MTPModel", &[]),
    ("LLaDA2MoeModelLM", &[]),
    ("LLaMAForCausalLM", &[]),
    ("LagunaForCausalLM", &[]),
    ("Lfm2DSparkDraftModel", &[]),
    ("LingDSparkModel", &[]),
    ("Llama4ForCausalLM", &[]),
    ("LlamaBidirectionalForSequenceClassification", &[]),
    ("LlamaBidirectionalModel", &[]),
    ("LlamaEmbeddingModel", &[]),
    (
        "LlamaForCausalLM",
        &[Implementation::of::<
            llama::ForCausalLM<llama::Size<"8B">, form::Affine, Parallel<1>, cache::Bf16, Hbm>,
        >()],
    ),
    ("LlamaForCausalLMEagle", &[]),
    ("LlamaForCausalLMEagle3", &[]),
    // Source: <https://github.com/sgl-project/sglang/blob/81cb895b85c1ac3361bfd9a241f362b48b75301f/python/sglang/srt/models/llama_classification.py>.
    ("LlamaForClassification", &[]),
    ("LlamaForSequenceClassification", &[]),
    ("LlamaForSequenceClassificationWithNormal_Weights", &[]),
    ("LlamaModel", &[]),
    ("LongCatFlashMTPModel", &[]),
    ("LongcatFlashForCausalLM", &[]),
    ("LongcatFlashForCausalLMNextN", &[]),
    ("LongcatFlashNgramForCausalLM", &[]),
    ("MammothModa2Qwen2ForCausalLM", &[]),
    ("MammothModa2Qwen3ForCausalLM", &[]),
    ("MedusaModel", &[]),
    ("MellumForCausalLM", &[]),
    ("MiMoAudioLLMModel", &[]),
    ("MiMoForCausalLM", &[]),
    ("MiMoMTP", &[]),
    ("MiMoMTPModel", &[]),
    ("MiMoV2FlashForCausalLM", &[]),
    ("MiMoV2ForCausalLM", &[]),
    ("MiMoV2MTP", &[]),
    ("MiMoV2MTPModel", &[]),
    ("MiMoV2OmniMTPModel", &[]),
    ("MingLLMModel", &[]),
    ("MiniCPM3ForCausalLM", &[]),
    ("MiniCPMForCausalLM", &[]),
    ("MiniCPMO45OmniTTSForConditionalGeneration", &[]),
    ("MiniCPMSALAForCausalLM", &[]),
    ("MiniMaxM2ForCausalLM", &[]),
    ("MiniMaxM3MTP", &[]),
    ("MiniMaxM3SparseForCausalLM", &[]),
    // Source: <https://github.com/vllm-project/vllm-omni/blob/6daf5b30f07cc0b3db87766d22fdf433f8a1a7c9/vllm_omni/model_executor/models/minimax_music3/talker.py>.
    ("MiniMaxMusic3TalkerForConditionalGeneration", &[]),
    ("Ministral3ForCausalLM", &[]),
    (
        "MistralForCausalLM",
        &[Implementation::of::<
            llama::ForCausalLM<
                llama::Size<"Mistral-7B">,
                form::Affine,
                Parallel<1>,
                cache::Bf16,
                Hbm,
            >,
        >()],
    ),
    ("MistralForCausalLMEagle", &[]),
    ("MistralLarge3ForCausalLM", &[]),
    ("MistralLarge3ForCausalLMEagle", &[]),
    ("MistralModel", &[]),
    ("MixtralForCausalLM", &[]),
    ("MossTTSDelayModel", &[]),
    ("MossTTSLocalModel", &[]),
    ("MossTTSNanoForCausalLM", &[]),
    ("MossTTSRealtime", &[]),
    ("MuseGlimmerAssistantModel", &[]),
    ("MuseGlimmerForCausalLM", &[]),
    ("NanbeigeForCausalLM", &[]),
    ("NanbeigeForSequenceClassification", &[]),
    // Source: <https://github.com/vllm-project/vllm-omni/blob/6daf5b30f07cc0b3db87766d22fdf433f8a1a7c9/vllm_omni/model_executor/models/audex/audex_thinker.py>.
    ("NemotronDenseForCausalLM", &[]),
    ("NemotronForCausalLM", &[]),
    ("NemotronHForCausalLM", &[]),
    ("NemotronHForCausalLMMTP", &[]),
    ("NemotronHMTPModel", &[]),
    ("NemotronHPuzzleForCausalLM", &[]),
    ("NemotronVoiceChatTalkerForConditionalGeneration", &[]),
    ("OPTForCausalLM", &[]),
    ("Olmo2ForCausalLM", &[]),
    ("Olmo3ForCausalLM", &[]),
    ("OlmoForCausalLM", &[]),
    ("OlmoeForCausalLM", &[]),
    ("OpenAIPrivacyFilterForTokenClassification", &[]),
    ("OpenPanguMTPModel", &[]),
    ("OrionForCausalLM", &[]),
    ("PEagleDraftModel", &[]),
    ("PanguEmbeddedForCausalLM", &[]),
    ("PanguProMoEV2ForCausalLM", &[]),
    ("PanguUltraMoEForCausalLM", &[]),
    ("Param2MoEForCausalLM", &[]),
    ("PeagleLlamaForCausalLM", &[]),
    ("PeagleQwen3ForCausalLM", &[]),
    ("PersimmonForCausalLM", &[]),
    ("PersonaPlexTalkerForConditionalGeneration", &[]),
    ("Phi3ForCausalLM", &[]),
    ("Phi3SmallForCausalLM", &[]),
    ("PhiForCausalLM", &[]),
    ("PhiMoEForCausalLM", &[]),
    ("Plamo3ForCausalLM", &[]),
    ("QWenLMHeadModel", &[]),
    ("QuantMixtralForCausalLM", &[]),
    ("Qwen2ForCausalLM", &[]),
    ("Qwen2ForCausalLMEagle", &[]),
    // Source: <https://github.com/vllm-project/vllm-omni/blob/6daf5b30f07cc0b3db87766d22fdf433f8a1a7c9/vllm_omni/model_executor/models/qwen2_5_omni/qwen2_old.py>.
    ("Qwen2ForCausalLM_old", &[]),
    ("Qwen2ForProcessRewardModel", &[]),
    ("Qwen2ForRewardModel", &[]),
    ("Qwen2ForSequenceClassification", &[]),
    ("Qwen2Model", &[]),
    ("Qwen2MoeForCausalLM", &[]),
    ("Qwen2_5OmniTalkerModel", &[]),
    ("Qwen3DSparkModel", &[]),
    (
        "Qwen3ForCausalLM",
        &[
            Implementation::of::<
                qwen3::ForCausalLM<
                    qwen3::Size<"0.6B">,
                    form::Affine,
                    Parallel<1>,
                    cache::Bf16,
                    Hbm,
                >,
            >(),
            Implementation::of::<
                qwen3::ForCausalLM<
                    qwen3::Size<"1.7B">,
                    form::Affine,
                    Parallel<1>,
                    cache::Bf16,
                    Hbm,
                >,
            >(),
            Implementation::of::<
                qwen3::ForCausalLM<qwen3::Size<"4B">, form::Affine, Parallel<1>, cache::Bf16, Hbm>,
            >(),
            Implementation::of::<
                qwen3::ForCausalLM<qwen3::Size<"8B">, form::Affine, Parallel<1>, cache::Bf16, Hbm>,
            >(),
            Implementation::of::<qwen3::fp8::ForCausalLM>(),
            Implementation::of::<
                qwen3::ForCausalLM<
                    qwen3::Size<"0.6B">,
                    form::ChannelScaled,
                    Parallel<1>,
                    cache::Bf16,
                    Hbm,
                >,
            >(),
            Implementation::of::<
                qwen3::ForCausalLM<
                    qwen3::Size<"1.7B">,
                    form::ChannelScaled,
                    Parallel<1>,
                    cache::Bf16,
                    Hbm,
                >,
            >(),
            Implementation::of::<
                qwen3::ForCausalLM<
                    qwen3::Size<"4B">,
                    form::ChannelScaled,
                    Parallel<1>,
                    cache::Bf16,
                    Hbm,
                >,
            >(),
        ],
    ),
    ("Qwen3ForRewardModel", &[]),
    (
        "Qwen3ForSequenceClassification",
        &[
            Implementation::of::<
                qwen3::ForSequenceClassification<
                    qwen3::Size<"0.6B">,
                    form::Affine,
                    Parallel<1>,
                    cache::Bf16,
                    Hbm,
                >,
            >(),
            Implementation::of::<
                qwen3::ForSequenceClassification<
                    qwen3::Size<"1.7B">,
                    form::Affine,
                    Parallel<1>,
                    cache::Bf16,
                    Hbm,
                >,
            >(),
            Implementation::of::<
                qwen3::ForSequenceClassification<
                    qwen3::Size<"4B">,
                    form::Affine,
                    Parallel<1>,
                    cache::Bf16,
                    Hbm,
                >,
            >(),
            Implementation::of::<
                qwen3::ForSequenceClassification<
                    qwen3::Size<"8B">,
                    form::Affine,
                    Parallel<1>,
                    cache::Bf16,
                    Hbm,
                >,
            >(),
        ],
    ),
    (
        "Qwen3Model",
        &[
            Implementation::of::<
                qwen3::Model<qwen3::Size<"0.6B">, form::Affine, Parallel<1>, cache::Bf16, Hbm>,
            >(),
            Implementation::of::<
                qwen3::Model<qwen3::Size<"1.7B">, form::Affine, Parallel<1>, cache::Bf16, Hbm>,
            >(),
            Implementation::of::<
                qwen3::Model<qwen3::Size<"4B">, form::Affine, Parallel<1>, cache::Bf16, Hbm>,
            >(),
            Implementation::of::<
                qwen3::Model<qwen3::Size<"8B">, form::Affine, Parallel<1>, cache::Bf16, Hbm>,
            >(),
        ],
    ),
    ("Qwen3MoeForCausalLM", &[]),
    ("Qwen3MoeForCausalLMMTP", &[]),
    // Head-free Hugging Face Qwen3 MoE decoder.
    ("Qwen3MoeModel", &[]),
    ("Qwen3NextForCausalLM", &[]),
    ("Qwen3NextForCausalLMMTP", &[]),
    ("Qwen3NextMTP", &[]),
    ("Qwen3OmniDSparkModel", &[]),
    ("Qwen3OmniMoeTalkerForConditionalGeneration", &[]),
    ("Qwen3TTSForConditionalGeneration", &[]),
    ("Qwen3TTSTalkerForConditionalGeneration", &[]),
    ("Qwen3_5ForCausalLM", &[]),
    ("Qwen3_5ForCausalLMMTP", &[]),
    ("Qwen3_5MTP", &[]),
    ("Qwen3_5MoeForCausalLM", &[]),
    ("Qwen3_5MoeMTP", &[]),
    ("Qwen4ExpForCausalLM", &[]),
    ("Rnj1ForCausalLM", &[]),
    ("SDARForCausalLM", &[]),
    ("SDARMoeForCausalLM", &[]),
    ("SarvamMLAForCausalLM", &[]),
    ("SarvamMoEForCausalLM", &[]),
    ("SeedOssForCausalLM", &[]),
    ("SglangDSparkDraftModel", &[]),
    ("SmolLM3ForCausalLM", &[]),
    ("SolarForCausalLM", &[]),
    ("Spark2_5ForCausalLM", &[]),
    ("StableLmForCausalLM", &[]),
    ("Starcoder2ForCausalLM", &[]),
    ("Step1ForCausalLM", &[]),
    ("Step3TextForCausalLM", &[]),
    ("Step3p5ForCausalLM", &[]),
    ("Step3p5MTP", &[]),
    ("TeleChat2ForCausalLM", &[]),
    ("TeleChat3ForCausalLM", &[]),
    ("TeleFLMForCausalLM", &[]),
    ("TorchNativeLlamaForCausalLM", &[]),
    ("TorchNativePhi3ForCausalLM", &[]),
    ("VaultGemmaForCausalLM", &[]),
    ("VoxCPM2TalkerForConditionalGeneration", &[]),
    ("VoyageQwen3BidirectionalEmbedModel", &[]),
    ("XllmForCausalLM", &[]),
    ("XverseForCausalLM", &[]),
    ("XverseMoeForCausalLM", &[]),
    ("ZayaForCausalLM", &[]),
    // # Recurrent decoders
    ("FalconMambaForCausalLM", &[]),
    ("Mamba2ForCausalLM", &[]),
    ("MambaForCausalLM", &[]),
    // # Hybrid decoders
    ("BailingMoeV3ForCausalLM", &[]),
    ("BambaForCausalLM", &[]),
    ("ColBERTLfm2Model", &[]),
    ("FalconH1ForCausalLM", &[]),
    ("Glm5NextForCausalLM", &[]),
    ("Glm5NextForConditionalGenerationNextN", &[]),
    ("Glm5NextMTPModel", &[]),
    ("GraniteMoeHybridForCausalLM", &[]),
    ("InklingForCausalLM", &[]),
    ("InklingForConditionalGeneration", &[]),
    ("InklingForConditionalGenerationMTP", &[]),
    ("InklingMTPModel", &[]),
    ("InternS2MobiusMTP", &[]),
    ("JambaForCausalLM", &[]),
    ("JambaForSequenceClassification", &[]),
    ("K3DSparkModel", &[]),
    ("KimiK3LinearForCausalLM", &[]),
    ("KimiLinearForCausalLM", &[]),
    ("Lfm2BidirectionalModel", &[]),
    ("Lfm2ForCausalLM", &[]),
    ("Lfm2MoeForCausalLM", &[]),
    ("OlmoHybridForCausalLM", &[]),
    ("Qwen4ExpForCausalLMMTP", &[]),
    ("Qwen4ExpMTP", &[]),
    ("Zamba2ForCausalLM", &[]),
    // # Encoders
    ("BertForMaskedLM", &[]),
    ("BertForSequenceClassification", &[]),
    ("BertForTokenClassification", &[]),
    ("BertModel", &[]),
    ("BertSpladeSparseEmbeddingModel", &[]),
    ("BgeM3EmbeddingModel", &[]),
    ("CLIPModel", &[]),
    ("CLIPTextModel", &[]),
    ("CLIPTextModelWithProjection", &[]),
    ("CLIPVisionModel", &[]),
    ("CLIPVisionModelWithProjection", &[]),
    ("ColBERTJinaRobertaModel", &[]),
    ("ColBERTModernBertModel", &[]),
    ("Contriever", &[]),
    ("GteModel", &[]),
    ("GteNewForSequenceClassification", &[]),
    ("GteNewModel", &[]),
    ("HF_ColBERT", &[]),
    ("JinaForRanking", &[]),
    ("ModernBertForSequenceClassification", &[]),
    ("ModernBertForTokenClassification", &[]),
    ("ModernBertModel", &[]),
    ("NomicBertModel", &[]),
    ("PixtralVisionModel", &[]),
    ("RobertaForMaskedLM", &[]),
    ("RobertaForSequenceClassification", &[]),
    ("RobertaForTokenClassification", &[]),
    ("RobertaModel", &[]),
    ("SiglipModel", &[]),
    ("SiglipVisionModel", &[]),
    ("XLMRobertaForSequenceClassification", &[]),
    ("XLMRobertaForTokenClassification", &[]),
    ("XLMRobertaModel", &[]),
    // # Encoder-decoders
    ("CohereAsrForConditionalGeneration", &[]),
    ("MT5ForConditionalGeneration", &[]),
    ("NemotronParseForConditionalGeneration", &[]),
    ("T5EncoderModel", &[]),
    ("T5ForConditionalGeneration", &[]),
    ("WhisperForConditionalGeneration", &[]),
    // # Vision-language models
    ("AriaForConditionalGeneration", &[]),
    ("AuKForConditionalGeneration", &[]),
    ("AudioFlamingo3ForConditionalGeneration", &[]),
    // Source: <https://github.com/vllm-project/vllm-omni/blob/6daf5b30f07cc0b3db87766d22fdf433f8a1a7c9/vllm_omni/model_executor/models/aura_omni/qwen3_vl.py>.
    ("AuraQwen3VLForConditionalGeneration", &[]),
    // Source: <https://github.com/vllm-project/vllm/blob/153242a314153637999eb6ebe8dd830e63433bb6/vllm/model_executor/models/bagel.py>.
    ("BagelForConditionalGeneration", &[]),
    ("BailingMM2NativeForConditionalGeneration", &[]),
    ("BailingMMNativeForConditionalGeneration", &[]),
    ("BailingMoeV3VLForConditionalGeneration", &[]),
    ("BeeForConditionalGeneration", &[]),
    ("Blip2ForConditionalGeneration", &[]),
    ("Cohere2VisionForConditionalGeneration", &[]),
    ("CohereCompassForConditionalGeneration", &[]),
    ("ColModernVBertForRetrieval", &[]),
    ("ColPaliForRetrieval", &[]),
    ("ColQwen3", &[]),
    ("ColQwen3_5", &[]),
    ("Cosmos3EdgeForConditionalGeneration", &[]),
    ("Cosmos3ForConditionalGeneration", &[]),
    // Source: <https://github.com/vllm-project/vllm-omni/blob/6daf5b30f07cc0b3db87766d22fdf433f8a1a7c9/vllm_omni/model_executor/models/covo_audio/covo_audio_llm.py>.
    ("CovoAudioLLMModel", &[]),
    ("DeepseekOCR2ForCausalLM", &[]),
    ("DeepseekOCRForCausalLM", &[]),
    ("DeepseekV41ForCausalLM", &[]),
    ("DeepseekV4ForConditionalGeneration", &[]),
    ("DeepseekVL2ForCausalLM", &[]),
    ("DeepseekVLV2ForCausalLM", &[]),
    ("Dots3NoteForCausalLM", &[]),
    ("DotsOCRForCausalLM", &[]),
    ("DotsVLMForCausalLM", &[]),
    ("Eagle2_5_VLForConditionalGeneration", &[]),
    ("Emu3ForConditionalGeneration", &[]),
    ("Ernie4_5_VLMoeForConditionalGeneration", &[]),
    ("Exaone4_5_ForConditionalGeneration", &[]),
    ("FireRedASR2ForConditionalGeneration", &[]),
    ("FunASRForConditionalGeneration", &[]),
    ("FunAudioChatForConditionalGeneration", &[]),
    ("GLM4VForCausalLM", &[]),
    ("Gemma3ForConditionalGeneration", &[]),
    ("Gemma3nForConditionalGeneration", &[]),
    (
        "Gemma4UnifiedForConditionalGeneration",
        &[Implementation::of::<gemma4_unified::Encoder<Hbm>>()],
    ),
    ("Glm4vForConditionalGeneration", &[]),
    ("Glm4vMoeForConditionalGeneration", &[]),
    ("Glm5NextForConditionalGeneration", &[]),
    ("GlmAsrForConditionalGeneration", &[]),
    ("GlmImageForConditionalGeneration", &[]),
    ("GlmOcrForConditionalGeneration", &[]),
    ("Granite4VisionForConditionalGeneration", &[]),
    ("GraniteSpeechForConditionalGeneration", &[]),
    ("GraniteSpeechPlusForConditionalGeneration", &[]),
    ("H2OVLChatModel", &[]),
    ("HCXVisionV2ForCausalLM", &[]),
    ("HfMoondream", &[]),
    ("HunYuanVLForConditionalGeneration", &[]),
    ("HunyuanImage3ForCausalMM", &[]),
    ("Idefics3ForConditionalGeneration", &[]),
    ("InternS1ForConditionalGeneration", &[]),
    ("InternS1ProForConditionalGeneration", &[]),
    ("InternS2MobiusForConditionalGeneration", &[]),
    ("InternS2PreviewForConditionalGeneration", &[]),
    ("InternVLChatModel", &[]),
    ("InternVLForConditionalGeneration", &[]),
    ("IsaacForConditionalGeneration", &[]),
    ("JetVLMForConditionalGeneration", &[]),
    ("JinaVLForRanking", &[]),
    ("KananaVForConditionalGeneration", &[]),
    ("KeyeForConditionalGeneration", &[]),
    ("KeyeVL1_5ForConditionalGeneration", &[]),
    ("KimiK25ForConditionalGeneration", &[]),
    ("KimiK3ForConditionalGeneration", &[]),
    ("KimiVLForConditionalGeneration", &[]),
    ("Lfm2VlForConditionalGeneration", &[]),
    ("LightOnOCRForConditionalGeneration", &[]),
    ("Llama4ForConditionalGeneration", &[]),
    ("LlamaNemotronVLForSequenceClassification", &[]),
    // Source: <https://github.com/vllm-project/vllm/blob/153242a314153637999eb6ebe8dd830e63433bb6/vllm/model_executor/models/nemotron_vl.py>.
    ("LlamaNemotronVLModel", &[]),
    ("Llama_Nemotron_Nano_VL", &[]),
    ("LlavaForConditionalGeneration", &[]),
    ("LlavaLlamaForCausalLM", &[]),
    ("LlavaMistralForCausalLM", &[]),
    ("LlavaNextForConditionalGeneration", &[]),
    // Source: <https://github.com/vllm-project/vllm/blob/153242a314153637999eb6ebe8dd830e63433bb6/vllm/model_executor/models/llava_next_video.py>.
    ("LlavaNextVideoForConditionalGeneration", &[]),
    ("LlavaOnevision2ForConditionalGeneration", &[]),
    ("LlavaOnevisionForConditionalGeneration", &[]),
    ("LlavaQwenForCausalLM", &[]),
    ("LlavaVidForCausalLM", &[]),
    ("LocateAnythingForConditionalGeneration", &[]),
    ("MammothModa2ARForConditionalGeneration", &[]),
    ("MammothModa2Qwen3ARForConditionalGeneration", &[]),
    ("MiDashengLMModel", &[]),
    ("MiMoV2ASRForCausalLM", &[]),
    ("MiMoV2OmniForCausalLM", &[]),
    // Source: <https://github.com/vllm-project/vllm-omni/blob/6daf5b30f07cc0b3db87766d22fdf433f8a1a7c9/vllm_omni/model_executor/models/ming_flash_omni/ming_flash_omni_thinker.py>.
    // Modality masks also reach the language body's experts, after input embedding fusion.
    ("MingFlashOmniThinkerForConditionalGeneration", &[]),
    // Source: <https://github.com/vllm-project/vllm-omni/blob/6daf5b30f07cc0b3db87766d22fdf433f8a1a7c9/vllm_omni/model_executor/models/ming_image/model.py>.
    ("MingImageForConditionalGeneration", &[]),
    // Source: <https://github.com/vllm-project/vllm/blob/153242a314153637999eb6ebe8dd830e63433bb6/vllm/model_executor/models/minicpmo.py>.
    ("MiniCPMO", &[]),
    // Source: <https://github.com/vllm-project/vllm-omni/blob/6daf5b30f07cc0b3db87766d22fdf433f8a1a7c9/vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni_llm.py>.
    ("MiniCPMO45OmniLLMForConditionalGeneration", &[]),
    ("MiniCPMV", &[]),
    ("MiniCPMV4_6ForConditionalGeneration", &[]),
    ("MiniMaxM3SparseForConditionalGeneration", &[]),
    ("Mistral3ForConditionalGeneration", &[]),
    // Cross attention keeps immutable encoder memory and independently gated, masked residuals.
    // Source: <https://github.com/sgl-project/sglang/blob/v0.5.19/python/sglang/srt/models/mllama.py#L563>.
    ("MllamaForConditionalGeneration", &[]),
    ("Molmo2ForConditionalGeneration", &[]),
    ("MolmoForCausalLM", &[]),
    ("Moondream3ForCausalLM", &[]),
    ("MoonshotKimiaForCausalLM", &[]),
    ("MossAudioModel", &[]),
    ("MossTranscribeDiarizeForConditionalGeneration", &[]),
    ("MossVLForConditionalGeneration", &[]),
    ("MultiModalityCausalLM", &[]),
    ("MuseGlimmerForConditionalGeneration", &[]),
    ("NVILAForConditionalGeneration", &[]),
    ("NVILALiteForConditionalGeneration", &[]),
    ("NVLM_D", &[]),
    // Source: <https://github.com/vllm-project/vllm-omni/blob/6daf5b30f07cc0b3db87766d22fdf433f8a1a7c9/vllm_omni/model_executor/models/audex/audex_omni.py>.
    ("NemotronDenseAudexForConditionalGeneration", &[]),
    ("NemotronHAudexForConditionalGeneration", &[]),
    ("NemotronH_Nano_Omni_Reasoning_V3", &[]),
    ("NemotronH_Nano_VL_V2", &[]),
    ("NemotronH_Omni_Reasoning_V3", &[]),
    ("NemotronH_Super_Omni_Reasoning_V3", &[]),
    ("NemotronVoiceChatThinkerForConditionalGeneration", &[]),
    ("OmniBagelForConditionalGeneration", &[]),
    ("OpenCUAForConditionalGeneration", &[]),
    ("OpenPanguVLForConditionalGeneration", &[]),
    ("OpenVLAForActionPrediction", &[]),
    ("OpsColQwen3Model", &[]),
    ("Ovis", &[]),
    ("Ovis2_5", &[]),
    ("Ovis2_6ForCausalLM", &[]),
    ("Ovis2_6_MoeForCausalLM", &[]),
    ("POINTSV15ChatModel", &[]),
    ("PaddleOCRVLForConditionalGeneration", &[]),
    ("PaliGemmaForConditionalGeneration", &[]),
    ("Phi3VForCausalLM", &[]),
    ("Phi4ForCausalLMV", &[]),
    ("Phi4MMForCausalLM", &[]),
    ("PixtralForConditionalGeneration", &[]),
    ("QianfanOCRForConditionalGeneration", &[]),
    ("Qwen2AudioForConditionalGeneration", &[]),
    // Source: <https://github.com/vllm-project/vllm/blob/153242a314153637999eb6ebe8dd830e63433bb6/vllm/model_executor/models/qwen2_vl.py>.
    ("Qwen2VLForConditionalGeneration", &[]),
    ("Qwen2_5OmniForConditionalGeneration", &[]),
    ("Qwen2_5OmniModel", &[]),
    // Source: <https://github.com/vllm-project/vllm-omni/blob/6daf5b30f07cc0b3db87766d22fdf433f8a1a7c9/vllm_omni/model_executor/models/qwen2_5_omni/qwen2_5_omni_thinker.py>.
    ("Qwen2_5OmniThinkerModel", &[]),
    ("Qwen2_5_VLForConditionalGeneration", &[]),
    ("Qwen3ASRForConditionalGeneration", &[]),
    ("Qwen3ASRForcedAlignerForTokenClassification", &[]),
    ("Qwen3ASRRealtimeGeneration", &[]),
    ("Qwen3OmniMoeForConditionalGeneration", &[]),
    ("Qwen3OmniMoeThinkerForConditionalGeneration", &[]),
    ("Qwen3VLForConditionalGeneration", &[]),
    ("Qwen3VLMoeForConditionalGeneration", &[]),
    ("Qwen3VLNemotronEmbedModel", &[]),
    ("Qwen3_5ForConditionalGeneration", &[]),
    ("Qwen3_5MoeForConditionalGeneration", &[]),
    ("Qwen4ExpForConditionalGeneration", &[]),
    ("RForConditionalGeneration", &[]),
    ("Sarashina2VisionForCausalLM", &[]),
    ("SkyworkR1VChatModel", &[]),
    ("SmolVLMForConditionalGeneration", &[]),
    ("Step3VLForConditionalGeneration", &[]),
    ("Step3p7ForConditionalGeneration", &[]),
    ("StepAudio2ThinkerForConditionalGeneration", &[]),
    ("StepVLForConditionalGeneration", &[]),
    ("UltravoxModel", &[]),
    ("UnlimitedOCRForCausalLM", &[]),
    ("VibeVoiceAsrForConditionalGeneration", &[]),
    ("VoxtralForConditionalGeneration", &[]),
    ("VoxtralRealtimeGeneration", &[]),
    ("YiVLForCausalLM", &[]),
    // # Diffusion
    ("AnimaPipeline", &[]),
    ("AuKPipeline", &[]),
    ("AudexCode2Wav", &[]),
    ("AudexXCodec1", &[]),
    // Source: <https://github.com/vllm-project/vllm-omni/blob/6daf5b30f07cc0b3db87766d22fdf433f8a1a7c9/vllm_omni/model_executor/models/audio8_tts/audio8_tts_codec_decoder.py>.
    ("Audio8TTSCodecDecoder", &[]),
    ("AutoencoderKL", &[]),
    // Unsized posterior composition; latent normalization is chosen by its children.
    ("AutoencoderKLFlux2", &[]),
    // Unsized posterior composition; the alias does not specify temporal kernels.
    ("AutoencoderKLQwenImage", &[]),
    // Unsized posterior composition; the alias does not specify temporal kernels.
    ("AutoencoderKLWan", &[]),
    ("BagelPipeline", &[]),
    ("BooguImagePipeline", &[]),
    ("BooguImageTurboPipeline", &[]),
    // Source: <https://github.com/vllm-project/vllm-omni/blob/6daf5b30f07cc0b3db87766d22fdf433f8a1a7c9/vllm_omni/model_executor/models/breeze_tts_2/code2wav.py>.
    ("BreezeCode2Wav", &[]),
    ("CausalWanTransformer3DModel", &[]),
    ("Cosmos3ActionPipeline", &[]),
    ("Cosmos3OmniDiffusersPipeline", &[]),
    ("Cosmos3OmniPipeline", &[]),
    ("Cosmos3SoundPipeline", &[]),
    ("Cosmos3TransferPipeline", &[]),
    ("Cosmos3VisualPipeline", &[]),
    ("CosyVoice3Model", &[]),
    // Source: <https://github.com/vllm-project/vllm-omni/blob/6daf5b30f07cc0b3db87766d22fdf433f8a1a7c9/vllm_omni/model_executor/models/covo_audio/covo_audio_code2wav.py>.
    ("CovoAudioCode2WavModel", &[]),
    // Source: <https://github.com/vllm-project/vllm-omni/blob/6daf5b30f07cc0b3db87766d22fdf433f8a1a7c9/vllm_omni/model_executor/models/covo_audio/covo_audio.py>.
    ("CovoAudioForCausalLM", &[]),
    // Source: <https://github.com/vllm-project/vllm-omni/blob/6daf5b30f07cc0b3db87766d22fdf433f8a1a7c9/vllm_omni/model_executor/models/covo_audio/covo_audio.py>.
    ("CovoAudioForConditionalGeneration", &[]),
    // Source: <https://github.com/vllm-project/vllm-omni/blob/6daf5b30f07cc0b3db87766d22fdf433f8a1a7c9/vllm_omni/model_executor/models/covo_audio/covo_audio.py>.
    ("CovoAudioModel", &[]),
    ("DiTPipeline", &[]),
    // Video and actions have separate solver histories and output decoding within one chunk trajectory.
    ("DreamZeroPipeline", &[]),
    ("ErnieImagePipeline", &[]),
    ("FishSpeechDACDecoder", &[]),
    ("Flux2KleinPipeline", &[]),
    ("Flux2Pipeline", &[]),
    ("Flux2Transformer2DModel", &[]),
    ("FluxDMD2Pipeline", &[]),
    ("FluxKontextPipeline", &[]),
    ("FluxPipeline", &[]),
    ("FluxTransformer2DModel", &[]),
    // Source: <https://github.com/vllm-project/vllm-omni/blob/6daf5b30f07cc0b3db87766d22fdf433f8a1a7c9/vllm_omni/model_executor/models/glm_tts/glm_tts.py>.
    ("GLMTTSForConditionalGeneration", &[]),
    ("GlmImagePipeline", &[]),
    ("Gr00tN1d7", &[]),
    ("Gr00tN1d7Pipeline", &[]),
    ("HeliosPipeline", &[]),
    ("HeliosPyramidPipeline", &[]),
    ("HeliosTransformer3DModel", &[]),
    ("HiDreamImagePipeline", &[]),
    // The encoder packs text and reference-image features; the predictor returns only generation positions.
    ("HiDreamO1ImagePipeline", &[]),
    // Source: <https://github.com/vllm-project/vllm-omni/blob/6daf5b30f07cc0b3db87766d22fdf433f8a1a7c9/vllm_omni/model_executor/models/higgs_audio_v2/higgs_audio_v2_code2wav.py>.
    ("HiggsAudioV2Code2WavForConditionalGeneration", &[]),
    // Source: <https://github.com/vllm-project/vllm-omni/blob/6daf5b30f07cc0b3db87766d22fdf433f8a1a7c9/vllm_omni/model_executor/models/higgs_audio_v3/higgs_audio_v3_code2wav.py>.
    ("HiggsAudioV3Code2WavForConditionalGeneration", &[]),
    ("Hunyuan3D2DiT", &[]),
    // AR planning selects image geometry before the separately callable image trajectory.
    ("HunyuanImage3Pipeline", &[]),
    ("HunyuanVideo15ImageToVideoPipeline", &[]),
    ("HunyuanVideo15Pipeline", &[]),
    ("HunyuanVideoTransformer3DModel", &[]),
    ("IndexTTS25S2MelDecoder", &[]),
    // Source: <https://github.com/vllm-project/vllm-omni/blob/6daf5b30f07cc0b3db87766d22fdf433f8a1a7c9/vllm_omni/model_executor/models/indextts2/indextts2_s2mel_decoder.py>.
    // Semantic embedding/length regulation prepares conditioning; velocity and vocoding follow.
    ("IndexTTS2S2MelDecoder", &[]),
    ("InternVLAA1", &[]),
    ("InternVLAA1Pipeline", &[]),
    ("Krea2Pipeline", &[]),
    ("LTX2DistilledOneStagePipeline", &[]),
    ("LTX2DistilledPipeline", &[]),
    ("LTX2DistilledTwoStagePipeline", &[]),
    ("LTX2I2VDMD2Pipeline", &[]),
    ("LTX2Pipeline", &[]),
    ("LTX2T2VDMD2Pipeline", &[]),
    ("LTX2TwoStagePipeline", &[]),
    ("LTX2VideoTransformer3DModel", &[]),
    // Understanding and generation have distinct readouts of shared model parameters.
    ("LancePipeline", &[]),
    ("LingBotVideoPipeline", &[]),
    // Chunk prediction reads the clean prefix; the commit function writes the completed chunk's KV.
    // The caller owns session resets, chunk boundaries and independently indexed decoder histories.
    ("LingBotWorldCausalDMDPipeline", &[]),
    ("LongCatImageEditPipeline", &[]),
    ("LongCatImagePipeline", &[]),
    ("LongCatImageTransformer2DModel", &[]),
    ("LongCatVideoAvatarPipeline", &[]),
    ("Magi2Pipeline", &[]),
    ("MammothModa2DiTPipeline", &[]),
    ("MammothModa2ForConditionalGeneration", &[]),
    ("Mammothmoda2Model", &[]),
    ("MiMoAudioModel", &[]),
    ("MiMoAudioToken2WavModel", &[]),
    ("MingAudioVAEModel", &[]),
    ("MingFlashOmniForConditionalGeneration", &[]),
    // Source: <https://github.com/vllm-project/vllm-omni/blob/6daf5b30f07cc0b3db87766d22fdf433f8a1a7c9/vllm_omni/model_executor/models/ming_flash_omni/ming_flash_omni_talker.py>.
    ("MingFlashOmniTalkerForConditionalGeneration", &[]),
    ("MingImageDiffusionPipeline", &[]),
    ("MingImagePipeline", &[]),
    ("MingTTSForConditionalGeneration", &[]),
    // Source: <https://github.com/vllm-project/vllm-omni/blob/6daf5b30f07cc0b3db87766d22fdf433f8a1a7c9/vllm_omni/model_executor/models/minicpmo_4_5/batched_token2wav.py>.
    ("MiniCPMO45Code2Wav", &[]),
    ("MiniCPMO45OmniForConditionalGeneration", &[]),
    ("MiniMaxMusic3AcousticForConditionalGeneration", &[]),
    ("MossTTSCodecDecoder", &[]),
    ("NemotronVoiceChatCode2Wav", &[]),
    ("NextStep11Pipeline", &[]),
    ("OmniGen2Pipeline", &[]),
    ("OmniVoice", &[]),
    ("OmniVoiceDecoder", &[]),
    ("OmniVoiceGenerator", &[]),
    ("OmniVoiceModel", &[]),
    ("OmniVoicePipeline", &[]),
    ("OvisImagePipeline", &[]),
    ("PersonaPlexCode2Wav", &[]),
    ("Pi05ForActionPrediction", &[]),
    ("Pi05Pipeline", &[]),
    ("Pi0ForActionPrediction", &[]),
    ("Pi0Pipeline", &[]),
    ("Qwen2_5OmniDenoiser", &[]),
    ("Qwen2_5OmniPipeline", &[]),
    ("Qwen2_5OmniToken2WavBigVGANModel", &[]),
    ("Qwen2_5OmniToken2WavDiTModel", &[]),
    ("Qwen2_5OmniToken2WavModel", &[]),
    // Source: <https://github.com/vllm-project/vllm-omni/blob/6daf5b30f07cc0b3db87766d22fdf433f8a1a7c9/vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_code2wav.py>.
    ("Qwen3OmniMoeCode2Wav", &[]),
    ("Qwen3OmniPipeline", &[]),
    ("Qwen3TTSCode2Wav", &[]),
    ("QwenImage21Transformer2DModel", &[]),
    ("QwenImageDMD2Pipeline", &[]),
    ("QwenImageEditPipeline", &[]),
    ("QwenImageEditPlusPipeline", &[]),
    ("QwenImageLayeredPipeline", &[]),
    ("QwenImagePipeline", &[]),
    ("QwenImageTransformer2DModel", &[]),
    ("SD3Transformer2DModel", &[]),
    ("SDXLUNet2DConditionModel", &[]),
    ("SanaImageToVideoPipeline", &[]),
    ("SanaTransformer2DModel", &[]),
    ("SanaVideoPipeline", &[]),
    // The clean first-frame latent is a condition restored by each numerical update.
    ("SanaWmPipeline", &[]),
    // Pixel reconstruction replaces VAE decoding; AR thinking and image denoising share model parameters.
    ("SenseNovaU1Pipeline", &[]),
    ("StableAudioPipeline", &[]),
    ("StableDiffusion3Pipeline", &[]),
    ("StableDiffusionXLPipeline", &[]),
    ("StepAudio2ForCausalLM", &[]),
    ("StepAudio2ForConditionalGeneration", &[]),
    ("StepAudio2Token2WavForConditionalGeneration", &[]),
    ("StepAudio2Token2WavModel", &[]),
    // Diffusers image transformer; configuration selects the supported function.
    ("Transformer2DModel", &[]),
    ("DiTTransformer2DModel", &[]),
    ("VoxtralTTSAudioGeneration", &[]),
    ("VoxtralTTSAudioTokenizer", &[]),
    ("VoxtralTTSForConditionalGeneration", &[]),
    ("WanAudioModel", &[]),
    ("WanDMDPipeline", &[]),
    ("WanI2VDMD2Pipeline", &[]),
    ("WanImageToVideoPipeline", &[]),
    ("WanModel", &[]),
    ("WanPipeline", &[]),
    ("WanS2VPipeline", &[]),
    ("WanT2VDMD2Pipeline", &[]),
    ("WanTransformer3DModel", &[]),
    ("WanVACEPipeline", &[]),
    ("WanVACETransformer3DModel", &[]),
    ("ZImagePipeline", &[]),
    ("ZImageTransformer2DModel", &[]),
];

/// A type that implements an architecture at one size: its size, its class, the scope it reads a
/// checkpoint at, the adapters its calls run, and its strict load into HBM.
#[derive(Debug)]
pub struct Implementation {
    size: &'static str,
    class: Class,
    scope: fn(&Checkpoint) -> Result<&'static str, LoadError>,
    adapters: usize,
    load: fn(&mut Context<'_>, &Checkpoint, Tiling) -> Result<Model, LoadError>,
}

/// A model a load returns, as a framework serves it.
pub enum Model {
    /// A decoder of requests on a cache.
    Decoder(Box<dyn Served>),
    /// A media encoder of images, video frames and audio.
    Encoder(Box<dyn Encoder>),
}

impl Model {
    /// The decoder the model is, if it is one.
    pub fn decoder(self) -> Option<Box<dyn Served>> {
        match self {
            Self::Decoder(decoder) => Some(decoder),
            Self::Encoder(_) => None,
        }
    }
}

/// A type `ARCHITECTURES` lists: the part of a model a load of it reads, and the model it serves.
pub trait Listed: Implements + for<'a> Load<&'a Checkpoint> + Sized + 'static {
    /// Adapters a call of the model runs at most: none for a model whose projections take none.
    const ADAPTERS: usize = 0;

    /// Refuses a load of `part`, which the type does not read.
    fn reads(part: Part) -> Result<(), LoadError>;

    /// The type as a framework serves it, its calls attending tiled as `tiling` says.
    fn served(self, tiling: Tiling) -> Model;
}

/// A decoder reads a whole model and serves requests as an instance of the load's tiling.
impl<D: Implements + for<'a> Load<&'a Checkpoint> + Decoder + 'static> Listed for D {
    const ADAPTERS: usize = <D as Decoder>::ADAPTERS;

    fn reads(part: Part) -> Result<(), LoadError> {
        match part {
            Part::Whole => Ok(()),
            Part::Encoder => Err(LoadError::Unsupported {
                name: "encoder".into(),
                reason: "a decoder holds no media encoder that loads alone".into(),
            }),
        }
    }

    fn served(self, tiling: Tiling) -> Model {
        Model::Decoder(Box::new(Instance::new(self, tiling)))
    }
}

/// The media encoder reads a load of the encoder part alone, and serves media; its rows are
/// computed alone, so no tiling applies to them. It is listed by type: the decoders' impl takes
/// every type parameter.
impl Listed for gemma4_unified::Encoder<Hbm> {
    fn reads(part: Part) -> Result<(), LoadError> {
        match part {
            Part::Encoder => Ok(()),
            Part::Whole => Err(LoadError::Unsupported {
                name: "encoder".into(),
                reason: "a media encoder loads alone, so choose the encoder".into(),
            }),
        }
    }

    fn served(self, _: Tiling) -> Model {
        Model::Encoder(Box::new(self))
    }
}

impl Implementation {
    /// The implementation the type `T` is.
    pub const fn of<T: Listed>() -> Self {
        Self {
            size: T::SIZE,
            class: T::CLASS,
            scope: |source| {
                T::reads(source.part())?;
                T::scope(source)
            },
            adapters: T::ADAPTERS,
            load: |context, source, tiling| Ok(T::load(context, source)?.served(tiling)),
        }
    }

    /// The first of `names` that `ARCHITECTURES` lists, as it spells it, with its implementations;
    /// refuses names it lists none of as invalid, and a first listed one without any as unsupported.
    pub fn listed(names: &[&str]) -> Result<(&'static str, &'static [Self]), LoadError> {
        let row = (names.iter())
            .find_map(|&name| ARCHITECTURES.iter().find(|&&(listed, _)| listed == name));
        match row {
            Some(&(name, [])) => Err(LoadError::Unsupported {
                name: name.into(),
                reason: "no model type is implemented".into(),
            }),
            Some(&(listed, implementations)) => Ok((listed, implementations)),
            None => Err(LoadError::InvalidData {
                name: names.join(", "),
                reason: "unknown architecture".into(),
            }),
        }
    }

    /// The one of `implementations`, those listed for `name`, whose scope serves `source`, from its
    /// config and names alone; when none serves, a lone refusal is returned as it is.
    pub fn choose(
        name: &str,
        implementations: &'static [Self],
        source: &Checkpoint,
    ) -> Result<&'static Self, LoadError> {
        let (mut served, mut refused) = (Vec::new(), Vec::new());
        for implementation in implementations {
            match (implementation.scope)(source) {
                Ok(_) => served.push(implementation),
                Err(refusal) => refused.push(refusal),
            }
        }
        let invalid = |reason: String| LoadError::InvalidData {
            name: name.into(),
            reason,
        };
        let joined = |parts: Vec<String>| parts.join("; ");
        match served.as_slice() {
            [implementation] => Ok(implementation),
            [] if refused.len() == 1 => Err(refused.remove(0)),
            [] => Err(invalid(format!(
                "no implementation serves the checkpoint: {}",
                joined(refused.iter().map(ToString::to_string).collect())
            ))),
            _ => Err(invalid(format!(
                "implementations of sizes {} all serve the checkpoint",
                joined(served.iter().map(|chosen| chosen.size.into()).collect())
            ))),
        }
    }

    /// A model of this implementation's type, loaded strictly from `source` at its scope, serving
    /// its calls with the attention tiled as `tiling` says.
    pub fn load(
        &self,
        context: &mut Context<'_>,
        source: &Checkpoint,
        tiling: Tiling,
    ) -> Result<Model, LoadError> {
        let scope = (self.scope)(source)?;
        context.field("", scope, |cx| (self.load)(cx, source, tiling))
    }

    /// The size, as the hub spells it.
    pub fn size(&self) -> &'static str {
        self.size
    }

    /// The class of the type.
    pub fn class(&self) -> Class {
        self.class
    }

    /// Adapters a call of the decoder runs at most: none when its projections take none.
    pub fn adapters(&self) -> usize {
        self.adapters
    }
}

#[cfg(test)]
mod tests {
    use super::{ARCHITECTURES, Implementation, Tiling};
    use crate::decoder::{Batch, Decoder, Limits, Output, Rows};
    use furiosa_opt_models::load::{Checkpoint, Context, Load, LoadError};
    use furiosa_opt_models::models::{Class, Implements};
    use furiosa_opt_std::prelude::{Broadcast, Device, HbmBuffer, m};
    use std::collections::HashSet;
    use std::io;

    /// A decoder that serves the checkpoints whose config's `size` is its `SIZE` or `every`: a
    /// hidden-row decoder when large, a token decoder when small.
    struct Fixture<const LARGE: bool>;

    impl<const LARGE: bool> Implements for Fixture<LARGE> {
        const SIZE: &'static str = if LARGE { "large" } else { "small" };
        const CLASS: Class = Class {
            family: "fixture",
            name: "Fixture",
        };

        fn scope(source: &Checkpoint) -> Result<&'static str, LoadError> {
            match source.config()["size"].as_str() {
                Some(size) if size == Self::SIZE || size == "every" => Ok(""),
                _ => Err(LoadError::InvalidData {
                    name: "config.json:size".into(),
                    reason: format!("not {}", Self::SIZE),
                }),
            }
        }
    }

    impl<'a, const LARGE: bool> Load<&'a Checkpoint> for Fixture<LARGE> {
        fn load(_: &mut Context<'_>, _: &'a Checkpoint) -> Result<Self, LoadError> {
            Ok(Self)
        }
    }

    impl<const LARGE: bool> Decoder for Fixture<LARGE> {
        const OUTPUT: Output = if LARGE {
            Output::Hidden
        } else {
            Output::Tokens
        };
        type Output = Rows<i32, m![1]>;
        type Layers = ();
        type Adapter = std::convert::Infallible;

        fn limits(&self) -> Limits {
            unreachable!("a registry test calls no decoder")
        }

        fn layers(_: &[HbmBuffer]) -> io::Result<()> {
            Ok(())
        }

        async fn call(
            &self,
            _: &mut Device,
            (): &mut (),
            _: Batch<'_, Self::Adapter>,
            _: Tiling,
        ) -> io::Result<Self::Output> {
            unreachable!("a registry test calls no decoder")
        }
    }

    /// The implementations of a name: a small decoder and a large one.
    const SIZES: &[Implementation] = &[
        Implementation::of::<Fixture<false>>(),
        Implementation::of::<Fixture<true>>(),
    ];

    /// Every name is listed once, and parses to its row: one with implementations to them, a
    /// refused one as unsupported, by its name.
    #[test]
    fn lists_every_name_once_by_its_kind() {
        let names: HashSet<_> = ARCHITECTURES.iter().map(|(name, _)| name).collect();
        assert_eq!((names.len(), ARCHITECTURES.len()), (670, 670));
        for &(name, implementations) in ARCHITECTURES {
            match (implementations, Implementation::listed(&[name])) {
                ([], Err(refusal @ LoadError::Unsupported { .. })) => {
                    let reason = format!("{name}: unsupported, no model type is implemented");
                    assert_eq!(refusal.to_string(), reason)
                }
                ([_, ..], Ok((listed, found))) => {
                    assert_eq!((listed, found.len()), (name, implementations.len()))
                }
                (_, parsed) => panic!("{name}: {parsed:?}"),
            }
        }
    }

    /// A name the list does not hold is `InvalidData`, by its name.
    #[test]
    fn refuses_unknown_names_as_invalid() {
        for name in ["UnknownModel", "", "qwen3forcausallm"] {
            let parsed = Implementation::listed(&[name]);
            assert!(
                matches!(&parsed, Err(LoadError::InvalidData { name: refused, reason })
                    if refused == name && reason == "unknown architecture"),
                "{name}: {parsed:?}"
            );
        }
    }

    /// The one implementation whose scope serves the checkpoint's config is chosen and loads as its
    /// decoder; a config none serves is refused with each one's refusal, or a lone one's own, and
    /// one that several serve is refused.
    #[test]
    fn chooses_the_one_implementation_that_serves_the_config() {
        let empty = tempfile::tempdir().unwrap();
        let source = |size: &str| {
            let config = Some(serde_json::json!({ "size": size }));
            Checkpoint::new(empty.path(), config).unwrap()
        };
        let large = source("large");
        let chosen = Implementation::choose("Causal", SIZES, &large).unwrap();
        assert_eq!(chosen.size(), "large");
        let loaded = chosen.load(&mut Context::new(None), &large, Tiling::Adaptive);
        assert_eq!(loaded.unwrap().decoder().unwrap().output(), Output::Hidden);
        let refusal = |implementations, size| {
            let chosen = Implementation::choose("Causal", implementations, &source(size));
            match chosen {
                Err(LoadError::InvalidData { name, reason }) => (name, reason),
                other => panic!("{size}: {:?}", other.map(Implementation::size)),
            }
        };
        let unserved = concat!(
            "no implementation serves the checkpoint: ",
            "config.json:size: not small; config.json:size: not large",
        );
        assert_eq!(refusal(SIZES, "huge"), ("Causal".into(), unserved.into()));
        let several = "implementations of sizes small; large all serve the checkpoint";
        assert_eq!(refusal(SIZES, "every"), ("Causal".into(), several.into()));
        let lone = refusal(&SIZES[..1], "huge");
        assert_eq!(lone, ("config.json:size".into(), "not small".into()));
    }

    /// A load of the media encoder alone chooses an encoder's implementation, and a whole load a
    /// decoder's; each refuses a checkpoint read for the other part.
    #[test]
    fn chooses_by_the_part_a_load_reads() {
        let empty = tempfile::tempdir().unwrap();
        let config = serde_json::json!({
            "model_type": "gemma4_unified",
            "text_config": {"hidden_size": 3840},
            "vision_config": {
                "mm_embed_dim": 3840, "mm_posemb_size": 1120, "patch_size": 16,
                "pooling_kernel_size": 3, "rms_norm_eps": 1e-6,
            },
            "audio_config": {"audio_embed_dim": 640, "rms_norm_eps": 1e-6},
        });
        let media = Checkpoint::new(empty.path(), Some(config)).unwrap();
        let (name, encoders) =
            Implementation::listed(&["Gemma4UnifiedForConditionalGeneration"]).unwrap();
        let chosen = Implementation::choose(name, encoders, &media.encoder()).unwrap();
        assert_eq!(chosen.size(), "12B");
        let small = Checkpoint::new(empty.path(), Some(serde_json::json!({ "size": "small" })));
        let small = small.unwrap();
        assert_eq!(
            Implementation::choose("Causal", &SIZES[..1], &small)
                .unwrap()
                .size(),
            "small"
        );
        for refused in [
            Implementation::choose(name, encoders, &media),
            Implementation::choose("Causal", &SIZES[..1], &small.encoder()),
        ] {
            assert!(
                matches!(&refused, Err(LoadError::Unsupported { name, .. }) if name == "encoder"),
                "{refused:?}"
            );
        }
    }
}
