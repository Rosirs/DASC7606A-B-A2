from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, PreTrainedModel, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType

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
    
    return tokenizer


def initialize_model() -> PreTrainedModel:
    """
    Initialize a model for sequence-to-sequence tasks with LoRA optimization.

    Returns:
        A model for sequence-to-sequence tasks with LoRA adapters.

    NOTE: You are free to change this.
    """
    # Load base model
    model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_CHECKPOINT
    )
       
    # Enable gradient checkpointing to reduce memory usage
    # This allows larger batch sizes at the cost of ~20% slower training
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    print(f"✓ Loaded base model: {MODEL_CHECKPOINT}")
    
    # Configure LoRA - Optimized for translation tasks
    # Choose one of the following configurations:
    
    # === Configuration 1: Balanced (Recommended) ===
    # Good trade-off between performance and efficiency
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=32,                              # LoRA rank: 32 for better performance
        lora_alpha=64,                     # Alpha: 2*r for stable training
        lora_dropout=0.05,                 # Lower dropout for better convergence
        target_modules=[
            "q_proj",                      # Query projection (essential)
            "v_proj",                      # Value projection (essential)
            "k_proj",                      # Key projection
            "out_proj",                    # Output projection
            "fc1",                         # Feed-forward layer 1
            "fc2",                         # Feed-forward layer 2
        ],
        bias="none",
        inference_mode=False,
    )
    
    # === Configuration 2: High Performance (Uncomment to use) ===
    # More parameters, better quality but slower training
    # lora_config = LoraConfig(
    #     task_type=TaskType.SEQ_2_SEQ_LM,
    #     r=64,                            # Higher rank for better capacity
    #     lora_alpha=128,                  # Alpha: 2*r
    #     lora_dropout=0.05,
    #     target_modules=[
    #         "q_proj", "v_proj", "k_proj", "out_proj",
    #         "fc1", "fc2",
    #     ],
    #     bias="none",
    #     inference_mode=False,
    # )
    
    # === Configuration 3: Memory Efficient (Uncomment to use) ===
    # Minimal parameters, fastest training, lower memory usage
    # lora_config = LoraConfig(
    #     task_type=TaskType.SEQ_2_SEQ_LM,
    #     r=8,                             # Lower rank for efficiency
    #     lora_alpha=16,                   # Alpha: 2*r
    #     lora_dropout=0.1,
    #     target_modules=[
    #         "q_proj",                    # Only essential modules
    #         "v_proj",
    #     ],
    #     bias="none",
    #     inference_mode=False,
    # )
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model
