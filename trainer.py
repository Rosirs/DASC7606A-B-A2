from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from constants import OUTPUT_DIR
from evaluation import compute_metrics


def create_training_arguments() -> TrainingArguments:
    """
    Create and return the training arguments for the model.

    Returns:
        Training arguments for the model.

    NOTE: You can change the training arguments as needed.
    # Below is an example of how to create training arguments. You are free to change this.
    # ref: https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments
    """
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        
        # Training epochs and batch size
        num_train_epochs=3,
        per_device_train_batch_size=16,  # Reduced for NLLB-600M with gradient checkpointing
        per_device_eval_batch_size=32,   # Can be larger during eval (no gradients)
        gradient_accumulation_steps=8,   # Effective batch size = 16 * 8 = 128
        
        # Learning rate and optimization
        learning_rate=3e-5,              # Optimal for NLLB fine-tuning
        weight_decay=0.01,               # L2 regularization
        warmup_ratio=0.1,                # 10% warmup steps
        warmup_steps=0,
        max_grad_norm=1.0,               # Gradient clipping
        
        # Evaluation and logging
        eval_strategy="steps",
        eval_steps=500,                  # Evaluate every 500 steps
        logging_steps=100,               # Log every 100 steps
        logging_first_step=True,
        
        # Checkpointing
        save_strategy="steps",
        save_steps=500,                  # Save checkpoint every 500 steps
        save_total_limit=3,              # Keep only 3 best checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        
        # Generation settings for NLLB
        predict_with_generate=True,
        generation_max_length=256,       # Match MAX_TARGET_LENGTH
        generation_num_beams=5,          # Beam search for better quality
        
        # Regularization
        label_smoothing_factor=0.1,      # Prevent overconfidence
        
        # Performance optimizations
        fp16=True,                       # Mixed precision training
        dataloader_num_workers=4,        # Parallel data loading
        dataloader_pin_memory=True,      # Speed up data transfer to GPU
        group_by_length=True,            # Group similar lengths for efficiency
        
        # Memory optimizations (enabled gradient checkpointing in model)
        gradient_checkpointing=True,     # Must match model settings
        
        # Additional settings
        report_to="none",                # Disable wandb/tensorboard if not needed
        remove_unused_columns=True,
        push_to_hub=False,
    )

    return training_args


def create_data_collator(tokenizer, model):
    """
    Create data collator for sequence-to-sequence tasks optimized for NLLB.

    Args:
        tokenizer: Tokenizer object.
        model: Model object.

    Returns:
        DataCollatorForSeq2Seq instance.

    NOTE: You are free to change this. But make sure the data collator is the same as the model.
    """
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model,
        padding=True,              # Dynamic padding for efficiency
        pad_to_multiple_of=8,      # Optimize for tensor cores (fp16)
    )


def build_trainer(model, tokenizer, tokenized_datasets) -> Trainer:
    """
    Build and return the trainer object for training and evaluation.

    Args:
        model: Model for sequence-to-sequence tasks.
        tokenizer: Tokenizer object.
        tokenized_datasets: Tokenized datasets.

    Returns:
        Trainer object for training and evaluation.

    NOTE: You are free to change this. But make sure the trainer is the same as the model.
    """
    data_collator = create_data_collator(tokenizer, model)
    training_args: TrainingArguments = create_training_arguments()

    return Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
    )
