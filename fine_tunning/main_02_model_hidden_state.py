import torch
from transformers import AutoModelForSequenceClassification, AutoModel


def main(num_labels, model_ckpt, tokenizer, device, ds_encoded):
    global model, ds_hidden

    num_labels = num_labels
    model = (AutoModelForSequenceClassification
            .from_pretrained(model_ckpt, num_labels=num_labels)
            .to(device))

    automodel = AutoModel.from_pretrained(model_ckpt).to(device)
    
    def extract_hidden_states(batch):
        # Place model inputs on the GPU
        inputs = {k: v.to(device) for k, v in batch.items()
                if k in tokenizer.model_input_names}
        # Extract last hidden states
        with torch.no_grad():
            last_hidden_state = automodel(**inputs).last_hidden_state
        # Return vector for [CLS] token
        return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}

    ds_hidden = ds_encoded.map(extract_hidden_states, batched=True)
