import torch
import sys
import numpy as np
import pickle
from torch.nn.functional import cross_entropy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# Visualization
def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()


# Error Analysis
def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)


def forward_pass_with_label(batch):
    # Place all input tensors on the same device as the model
    inputs = {k: v.to(device) for k, v in batch.items()
              if k in tokenizer.model_input_names}
    with torch.no_grad():
        output = model(**inputs)
        pred_label = torch.argmax(output.logits, axis=-1)
        loss = cross_entropy(output.logits, batch["label"].to(device),
                             reduction="none")
    # Place outputs on CPU for compatibility with other dataset columns
    return {"loss": loss.cpu().numpy(),
            "predicted_label": pred_label.cpu().numpy()}


def main():
    global model, tokenizer, device, emotions

    # 0 ##  Checking GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("torch.cuda is not available, exiting...")
        sys.exit(0)

    # 1 ## Loading data
    with open("./metrics/results.pkl", "rb") as f:
        results = pickle.load(f)

    emotions = results["dataset"]
    tokenizer = results["tokenizer"]
    emotions_encoded = results["ds_encoded"]
    model = results["model"]

    # Convert our dataset back to PyTorch tensors
    emotions_encoded.set_format("torch",
                                columns=["input_ids", "attention_mask", "label"])
    # Compute loss values
    emotions_encoded["validation"] = emotions_encoded["validation"].map(
        forward_pass_with_label, batched=True, batch_size=16)

    # 2 ## Computing metrics
    emotions_encoded.set_format("pandas")
    cols = ["text", "label", "predicted_label", "loss"]
    df_test = emotions_encoded["validation"][:][cols]
    df_test["label"] = df_test["label"].apply(label_int2str)
    df_test["predicted_label"] = (
        df_test["predicted_label"].apply(label_int2str))

    return df_test


if __name__ == "__main__":
    df_test = main()

    # 4 ##  Saving the results
    df_test.to_csv("./metrics/results.csv", index=False)
