import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdapterType

print('a')

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model.load_adapter("sentiment/sst-2@ukp")


def predict(sentence):
    token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))
    input_tensor = torch.tensor([token_ids])

    outputs = model(input_tensor, adapter_names=['sst-2'])

    return 'possitive' if 1 == torch.argmax(outputs[0]).item() else 'negative'


print(predict("Those who find ugly meanings in beautiful things are corrupt without being charming."))
print(predict("There are slow and repetitive parts, but it has just enough spice to keep it interesting."))

