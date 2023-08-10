from transformers import BertForSequenceClassification, BertTokenizer
import torch.nn.functional as F
import torch


# initialize the tokenizer for BERT models
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
# initialize the model for sequence classification
model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')


tokens = tokenizer.encode_plus(txt, max_length=512, truncation=True, padding='max_length',
                               add_special_tokens=True, return_tensors='pt')

output = model(**tokens)
probs = F.softmax(output[0], dim=-1)
pred = torch.argmax(probs)