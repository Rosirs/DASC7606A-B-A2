from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset, concatenate_datasets
from transformers import DataCollatorForSeq2Seq

from constants import MAX_INPUT_LENGTH, MAX_TARGET_LENGTH


def filter_low_quality_data(example):
    """
    Filter out low-quality translation pairs.
    
    Filtering criteria:
    - Too short sentences (< 5 characters)
    - Too long sentences (> 500 characters)
    - Abnormal length ratio between source and target (> 5x difference)
    - Empty or whitespace-only content
    
    Args:
        example: Dataset example containing 'translation' field
        
    Returns:
        bool: True if the example should be kept, False otherwise
    """
    try:
        zh_text = example["translation"]["zh"].strip()
        en_text = example["translation"]["en"].strip()
        
        # Check for empty content
        if not zh_text or not en_text:
            return False
        
        zh_len = len(zh_text)
        en_len = len(en_text)
        
        # Filter too short sentences
        if zh_len < 5 or en_len < 5:
            return False
        
        # Filter too long sentences (likely noise or formatting issues)
        if zh_len > 500 or en_len > 500:
            return False
        
        # Filter abnormal length ratios (might indicate misalignment)
        length_ratio = max(zh_len / en_len, en_len / zh_len)
        if length_ratio > 5.0:
            return False
        
        # Filter sentences with excessive special characters (potential noise)
        zh_special_ratio = sum(1 for c in zh_text if not c.isalnum() and not c.isspace()) / zh_len
        en_special_ratio = sum(1 for c in en_text if not c.isalnum() and not c.isspace()) / en_len
        if zh_special_ratio > 0.5 or en_special_ratio > 0.5:
            return False
        
        return True
    except Exception:
        # If any error occurs during filtering, exclude the example
        return False


def build_dataset() -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    """
    Build the dataset using only WMT19 to avoid domain shift.

    Returns:
        The dataset with WMT19 training set.

    NOTE: You can replace this with your own dataset. Make sure to include
    the `validation` split and ensure that it is the same as the test split from the WMT19 dataset,
    Which means that:
        raw_datasets["validation"] = load_dataset('wmt19', 'zh-en', split="validation")
    """
    # Load WMT19 dataset (primary dataset)
    print("Loading WMT19 zh-en dataset...")
    dataset = load_dataset("wmt19", "zh-en")
    
    # Use only WMT19 to avoid domain shift from OPUS-100
    # Increase WMT19 to 1M samples for better generalization
    print("Selecting 1M samples from WMT19 train set...")
    wmt19_train = dataset["train"].select(range(1000000))
    
    # Filter low-quality data
    print("Filtering low-quality data from WMT19 training set...")
    train_dataset = wmt19_train.filter(filter_low_quality_data, num_proc=8)
    print(f"WMT19 after filtering: {len(wmt19_train)} -> {len(train_dataset)} samples")
    
    # Use WMT19 validation set for validation
    print("Using WMT19 validation set...")
    validation_dataset = dataset["validation"]

    train_dataset = train_dataset.select(range(20))
    validation_dataset = validation_dataset.select(range(20))
    # NOTE: You should not change the test dataset
    test_dataset = dataset["validation"]

    return DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset
    })


def create_data_collator(tokenizer, model):
    """
    Create data collator for sequence-to-sequence tasks.

    Args:
        tokenizer: Tokenizer object.
        model: Model object.

    Returns:
        DataCollatorForSeq2Seq instance.
    """
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


def preprocess_function(examples, prefix, tokenizer, max_input_length, max_target_length):
    """
    Preprocess the data for sequence-to-sequence models.

    Args:
        examples: Examples.
        prefix: Prefix to add before inputs (optional).
        tokenizer: Tokenizer object.
        max_input_length: Maximum input length.
        max_target_length: Maximum target length.

    Returns:
        Model inputs dict.
    """
    # Extract Chinese (source) and English (target) texts
    inputs = [prefix + ex["zh"] for ex in examples["translation"]]
    targets = [ex["en"] for ex in examples["translation"]]

    # Tokenize inputs (source)
    model_inputs = tokenizer(
        inputs, 
        max_length=max_input_length, 
        truncation=True, 
        padding=False
    )
    
    # Tokenize targets (labels)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length, 
            truncation=True, 
            padding=False
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def preprocess_data(raw_datasets: DatasetDict, tokenizer) -> DatasetDict:
    """
    Preprocess the data.

    Args:
        raw_datasets: Raw datasets.
        tokenizer: Tokenizer object.

    Returns:
        Tokenized datasets.
    """
    column_names = raw_datasets["train"].column_names
    
    tokenized_datasets: DatasetDict = raw_datasets.map(
        function=lambda examples: preprocess_function(
            examples=examples,
            prefix="",
            tokenizer=tokenizer,
            max_input_length=MAX_INPUT_LENGTH,
            max_target_length=MAX_TARGET_LENGTH,
        ),
        batched=True,
        remove_columns=column_names
    )
    return tokenized_datasets