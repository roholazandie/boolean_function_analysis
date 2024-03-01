from torch.utils.data import DataLoader

from boolean_datasets import ParityFunctionDataset, RandomBooleanFunctionDataset
from boolean_tokenizer import BooleanTokenizer
from utils import load_config
import os, torch
import wandb
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import BertTokenizer, BertConfig, set_seed, BertForSequenceClassification, \
    AutoModelForSequenceClassification, TrainingArguments, Trainer

wandb.init()

device = 'cuda:0'

configs = load_config('configs/random_boolean_function_sensitivity_config.json')

os.environ["WANDB_PROJECT"] = "random_boolean_function_sensitivity"  # log to your project
os.environ["WANDB_LOG_MODEL"] = "all"  # log your models

tokenizer = BooleanTokenizer()

id2label, label2id = {1: '-1', 0: '1'}, {'-1': 1, '1': 0}

random_boolean_dataset = RandomBooleanFunctionDataset(tokenizer, configs.max_length, configs.num_train_examples, label2id)

train_dataloader = DataLoader(random_boolean_dataset, batch_size=configs.train_batch_size, shuffle=True)

num_initializations = 1000 # Number of times you want to initialize the model


for i in range(num_initializations):
    torch.manual_seed(42+i)

    model_config = BertConfig(vocab_size=len(tokenizer.vocab),
                              max_position_embeddings=configs.max_length,
                              num_labels=len(id2label),
                              # num_hidden_layers=4,
                              #num_attention_heads=4
                              )

    model_config.id2label = id2label
    model_config.label2id = label2id
    model_config.problem_type = "single_label_classification"

    model = AutoModelForSequenceClassification.from_config(config=model_config).to(device)

    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set model to evaluation mode

    sensitivities = []
    # Evaluation loop
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to device
        original_input = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
        with torch.no_grad():  # No need to compute gradients
            outputs = model(**original_input)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        perturbed_input = {'input_ids': batch['perturbed_input_ids'], 'attention_mask': batch['attention_mask']}
        with torch.no_grad():  # No need to compute gradients
            perturbed_outputs = model(**perturbed_input)

        perturbed_logits = perturbed_outputs.logits
        perturbed_predictions = torch.argmax(perturbed_logits, dim=-1)

        # calculate the sensitivity
        sensitivity = torch.mean(torch.abs(predictions - perturbed_predictions).float())
        sensitivities.append(sensitivity.item())

    print(f"Mean sensitivity for initialization {i+1}: {sum(sensitivities)/len(sensitivities)}")
    wandb.log({"sensitivity": sum(sensitivities)/len(sensitivities)})


wandb.finish()
