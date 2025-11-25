from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, PreTrainedModel, AutoModelForSeq2SeqLM

from constants import MODEL_CHECKPOINT


def initialize_tokenizer() -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """
    Initialize a tokenizer for sequence-to-sequence tasks.

    Returns:
        A tokenizer for sequence-to-sequence tasks.

    NOTE: You are free to change this. But make sure the tokenizer is the same as the model.
    """
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODEL_CHECKPOINT
    )
    
    # Set source and target language codes for NLLB models
    # This is crucial for the model to know which language pair to translate
    tokenizer.src_lang = "zho_Hans"  # Simplified Chinese (source)
    tokenizer.tgt_lang = "eng_Latn"  # English (target)
    
    return tokenizer


def initialize_model() -> PreTrainedModel:
    """
    Initialize a model for sequence-to-sequence tasks. You are free to change this,
    not only seq2seq models, but also other models like BERT, or even LLMs.

    Returns:
        A model for sequence-to-sequence tasks.

    NOTE: You are free to change this.
    """
    model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_CHECKPOINT
    )
       
    # Enable gradient checkpointing to reduce memory usage
    # This allows larger batch sizes at the cost of ~20% slower training
    # if hasattr(model, 'gradient_checkpointing_enable'):
    #     model.gradient_checkpointing_enable()
   
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    if "nllb" in MODEL_CHECKPOINT.lower():
        # 注意：不同版本的 NLLB 英文代码可能是 eng_Latn
        tgt_lang_id = tokenizer.convert_tokens_to_ids("eng_Latn")
        model.config.forced_bos_token_id = tgt_lang_id
        print(f"Set forced_bos_token_id to {tgt_lang_id} (eng_Latn)")


    return model
