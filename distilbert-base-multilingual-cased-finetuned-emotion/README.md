---
license: apache-2.0
base_model: distilbert-base-multilingual-cased
tags:
- generated_from_trainer
datasets:
- emotion
metrics:
- accuracy
- f1
model-index:
- name: distilbert-base-multilingual-cased-finetuned-emotion
  results:
  - task:
      name: Text Classification
      type: text-classification
    dataset:
      name: emotion
      type: emotion
      config: split
      split: validation
      args: split
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.8895
    - name: F1
      type: f1
      value: 0.8900286564226987
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# distilbert-base-multilingual-cased-finetuned-emotion

This model is a fine-tuned version of [distilbert-base-multilingual-cased](https://huggingface.co/distilbert-base-multilingual-cased) on the emotion dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3596
- Accuracy: 0.8895
- F1: 0.8900

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 64
- eval_batch_size: 64
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 2

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy | F1     |
|:-------------:|:-----:|:----:|:---------------:|:--------:|:------:|
| 1.1585        | 1.0   | 250  | 0.5694          | 0.8265   | 0.8191 |
| 0.4367        | 2.0   | 500  | 0.3596          | 0.8895   | 0.8900 |


### Framework versions

- Transformers 4.33.2
- Pytorch 2.0.1+cu117
- Datasets 2.14.5
- Tokenizers 0.13.3
