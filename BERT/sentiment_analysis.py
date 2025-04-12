# preprocessing of data
from datasets import load_dataset
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

# fetch the imdb dataset for sentiment analysis
dataset = load_dataset("imdb")
print(dataset)

# use the pre-trained Huggingface BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenizer_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# apply the tokenizer function on the dataset
tokenized_dataset = dataset.map(tokenizer_function, batched=True)

# split the dataset into train and test sets
train_testvalid = tokenized_dataset['train'].train_test_split(test_size=0.2)
train_dataset = train_testvalid['train']
test_dataset = train_testvalid['test']

# load the dataset into batched dataloader
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16)

# load the model pretrained on sequence classification task
# as far as sentiment analysis is concerned, we have two labels - positive and negative
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# set up the training config and training loop

training_args = TrainingArguments(
    output_dir = './results',
    evaluation_strategy="epoch",
    num_train_epochs = 3,
    learning_rate = 2e-5,
    weight_decay = 0.01,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 16,
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
)

trainer.train()

# get the metrics - F1 score, precision, recall
metrics = trainer.evaluate()
print(metrics)

predictions = trainer.predict(test_dataset[:5])
print(predictions)

