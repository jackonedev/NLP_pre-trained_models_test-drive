from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def main(batch_size, model_ckpt, ds_encoded, tokenizer, model):
    global trainer

    batch_size = batch_size
    logging_steps = len(ds_encoded["train"]) // batch_size
    model_name = f"{model_ckpt}-finetuned-emotion"
    training_args = TrainingArguments(output_dir=model_name,
                                    num_train_epochs=2,
                                    learning_rate=2e-5,
                                    per_device_train_batch_size=batch_size,
                                    per_device_eval_batch_size=batch_size,
                                    weight_decay=0.01,
                                    evaluation_strategy="epoch",
                                    disable_tqdm=False,
                                    logging_steps=logging_steps,
                                    push_to_hub=True,
                                    log_level="error")


    trainer = Trainer(model=model, args=training_args,
                    compute_metrics=compute_metrics,
                    train_dataset=ds_encoded["train"],
                    eval_dataset=ds_encoded["validation"],
                    tokenizer=tokenizer)