import fine_tunning.main_01_dataset_tokenization as ds_tokenization
import fine_tunning.main_02_model_hidden_state as model_hidden_state
import fine_tunning.main_03_trainer_metrics as trainer_metrics
import torch
import sys
import os
import pickle
from huggingface_hub import login
from dotenv import load_dotenv


load_dotenv()


# 0 ##  Checking GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    print("torch.cuda is not available, exiting...")
    sys.exit(0)


# 1 ## Tokenize de Dataset
print("# 1 ## Tokenize de Dataset")
dataset_name = "emotion"
model_ckpt = "distilbert-base-multilingual-cased"
ds_tokenization.main(dataset_name, model_ckpt)
emotions = ds_tokenization.dataset
tokenizer = ds_tokenization.tokenizer
emotions_encoded = ds_tokenization.ds_encoded
# convert the input_ids and attention_mask columns to the "torch" format
emotions_encoded.set_format("torch",
                            columns=["input_ids", "attention_mask", "label"])


# 2 ##  Loading a pretrained model. Extracting the last hidden state
print("# 2 ##  Loading a pretrained model. Extracting the last hidden state")
num_labels = emotions["train"].features["label"].num_classes
model_hidden_state.main(num_labels, model_ckpt, tokenizer, device, emotions_encoded)
model = model_hidden_state.model
emotions_hidden = model_hidden_state.ds_hidden


# 3 ##  Training the model
print("# 3 ##  Training the model")
batch_size = 64
trainer_metrics.main(batch_size, model_ckpt, emotions_encoded, tokenizer, model)
trainer = trainer_metrics.trainer
trainer.train()


# 4 ##  Making predictions. Saving the results
print("# 4 ##  Making predictions. Saving the results")
preds_output = trainer.predict(emotions_encoded["validation"])
results = {"dataset": emotions,
           "tokenizer": tokenizer,
           "ds_encoded": emotions_encoded,
           "model": model,
           "ds_hidden": emotions_hidden,
           "predictions": preds_output}

with open("./metrics/results.pkl", "wb") as f:
    pickle.dump(results, f)


# 5 ##  Saving the model
print("# 5 ##  Saving the model")
login(os.getenv("HUGGINGFACE_TOKEN"))
trainer.push_to_hub(commit_message="Training completed!")

print("\nDone!")
