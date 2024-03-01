from boolean_datasets import ParityFunctionDataset, KSparseBooleanFunctionDataset, MajorityBooleanFunctionDataset, \
    MaxBooleanFunctionDataset, StaircaseFunctionDataset
from boolean_tokenizer import BooleanTokenizer
from utils import load_config
import os
import wandb
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import BertTokenizer, BertConfig, set_seed, BertForSequenceClassification, \
    AutoModelForSequenceClassification, TrainingArguments, Trainer

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# Set up the Trainer
set_seed(42)
device = 'cuda:0'



os.environ["WANDB_PROJECT"] = "boolean_function_analysis"  # log to your project
os.environ["WANDB_LOG_MODEL"] = "all"  # log your models

tokenizer = BooleanTokenizer()

id2label, label2id = {1: '-1', 0: '1'}, {'-1': 1, '1': 0}

dataset_name = "staircase_function"

if dataset_name == "parity":
    configs = load_config('configs/parity_config.json')
    train_dataset = ParityFunctionDataset(tokenizer, configs.max_length, 5, configs.num_train_examples, label2id)
    eval_dataset = ParityFunctionDataset(tokenizer, configs.max_length, 5, configs.num_eval_examples, label2id)
elif dataset_name == "staircase_function":
    configs = load_config('configs/staircase_config.json')
    train_dataset = StaircaseFunctionDataset(tokenizer, configs.max_length, 5,configs.num_train_examples, label2id)
    eval_dataset = StaircaseFunctionDataset(tokenizer, configs.max_length, 5,configs.num_eval_examples, label2id)
elif dataset_name == "ksparse_boolean":
    configs = load_config('configs/ksparse_config.json')
    train_dataset = KSparseBooleanFunctionDataset(tokenizer, configs.max_length, configs.num_train_examples, 3, label2id)
    eval_dataset = KSparseBooleanFunctionDataset(tokenizer, configs.max_length, configs.num_eval_examples, 3, label2id)
elif dataset_name == "majority":
    configs = load_config('configs/majority_config.json')
    train_dataset = MajorityBooleanFunctionDataset(tokenizer, configs.max_length, configs.num_train_examples, label2id)
    eval_dataset = MajorityBooleanFunctionDataset(tokenizer, configs.max_length, configs.num_eval_examples, label2id)
elif dataset_name == "max":
    configs = load_config('configs/max_config.json')
    train_dataset = MaxBooleanFunctionDataset(tokenizer, configs.max_length, configs.num_train_examples, label2id)
    eval_dataset = MaxBooleanFunctionDataset(tokenizer, configs.max_length, configs.num_eval_examples, label2id)


model_config = BertConfig(vocab_size=len(tokenizer.vocab),
                          max_position_embeddings=configs.max_length + 2,
                          num_labels=len(id2label),
                          # num_hidden_layers=4,
                          #num_attention_heads=4
                          )
print(model_config)

model_config.id2label = id2label
model_config.label2id = label2id
model_config.problem_type = "single_label_classification"

model = AutoModelForSequenceClassification.from_config(config=model_config).to(device)

working_dir = f"{configs.output_dir}_{configs.max_length}"

training_args = TrainingArguments(
        output_dir=working_dir,
        overwrite_output_dir=False,
        num_train_epochs=configs.n_epochs,
        per_device_train_batch_size=configs.train_batch_size,
        per_device_eval_batch_size=configs.eval_batch_size,
        learning_rate=configs.learning_rate,
        logging_dir=working_dir,
        dataloader_num_workers=1,
        logging_steps=3,
        save_strategy="steps",  # save a checkpoint every save_steps
        save_steps=int(configs.save_steps * len(train_dataset)),
        save_total_limit=5,
        evaluation_strategy="steps",  # evaluation is done every eval_steps
        eval_steps=int(configs.save_steps * len(train_dataset)),
        load_best_model_at_end=False,
        metric_for_best_model="f1",
        # fp16=True
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

wandb.finish()
