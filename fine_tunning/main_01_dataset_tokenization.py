import datasets
from transformers import AutoTokenizer

def main(db_name, model_ckpt):
    global dataset, tokenizer, ds_encoded
    
    dataset = datasets.load_dataset(db_name)
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)


    ds_encoded = dataset.map(tokenize, batched=True, batch_size=None)