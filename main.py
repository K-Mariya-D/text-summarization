from datasets import load_dataset
import pandas as pd
import numpy as np
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from rouge_score import rouge_scorer
from datasets import Dataset
from transformers import (T5Tokenizer, T5ForConditionalGeneration,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq,
                          TrainerCallback)
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

#Load dataset
train_data = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train")
valid_data = load_dataset("abisee/cnn_dailymail", "3.0.0", split="validation")
#test_data = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")

#Check data
def info_about_data():
  print(f"train size = {len(train_data)}")
  print(f"valid size = {len(valid_data)}")

  article_lens = train_data.map(lambda ex: {'text_len' : [len(t) for t in ex["article"]]}, batched=True)
  highlights_lens = train_data.map(lambda ex: {'text_len': [len(t) for t in ex["highlights"]]}, batched=True)
  mean_art_len = mean_hgl_len = 0
  for i in range(0, len(article_lens)):
    mean_art_len += article_lens[i]["text_len"]
    mean_hgl_len += highlights_lens[i]["text_len"]

  mean_art_len /= len(article_lens)
  mean_hgl_len /= len(highlights_lens)
  compression_percent = mean_hgl_len/mean_art_len * 100

  print("\nINFO ABOUT TRAIN DATA")
  print(f"Mean article size in train: {round(mean_art_len, 2)}")
  print(f"Mean highlights size in train: {round(mean_hgl_len, 2)}")
  print(f"The article is decreasing on {round(compression_percent, 2)} %")

info_about_data()

train_data = train_data.remove_columns("id")
valid_data = valid_data.remove_columns("id")

#Extractive summarization
def extractive_summarize(data, count = None):
    if count is None:
         count = len(data)

    summarizer = LsaSummarizer()
    lsa_highlights = []

    for i in range(count):
        article_parser = PlaintextParser.from_string(data[i]["article"], Tokenizer("english"))
        highlight_parser = PlaintextParser.from_string(data[i]["highlights"], Tokenizer("english"))

        summary_length = len(highlight_parser.document.sentences)

        lsa_highlights.append(" ".join(str(sent) for sent in summarizer(article_parser.document, summary_length)))

    return lsa_highlights

#ROUGE-score
def f1_rouge_score(new_highlights, old_highlights, count = None):
        if count is None:
             count = min(len(new_highlights), len(old_highlights))

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        keys = ["new highlight", "reference highlight", "rouge1", "rouge2", "rougeL"]
        scores_tabel = {key: [] for key in keys}

        for i in range(count):
            new_h = new_highlights[i]
            old_h = old_highlights[i]
            scores_tabel["new highlight"].append(new_h)
            scores_tabel["reference highlight"].append(old_h)
            sc = scorer.score(new_h, old_h)
            for key in sc:
                scores_tabel[key].append(sc[key].fmeasure)

        return scores_tabel

#count = 100
#lsa_h = extractive_summarize(train_data, count)
#scores = f1_rouge_score(lsa_h, train_data["highlights"], count)
#print("INFO ABOUT EXTRACTIVE SUMMARY")
#print(f"Mean LSA ROUGE-1: {sum(scores["rouge1"])/count}")
#print(f"Mean LSA ROUGE-2: {sum(scores["rouge2"])/count}")
#print(f"Mean LSA ROUGE-L: {sum(scores["rougeL"])/count}")


train = train_data
valid = valid_data
#test = Dataset.from_pandas(test_data)

def fine_tuning():
    tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-small')

    def tokenize_funct(examples):
        articles = examples['article']
        highlights = examples['highlights']

        #преобразует список строк в словарь с input_ids, attention_mask.
        inputs = tokenizer(articles, truncation = True, max_length=256)
        labels = tokenizer(text_target = highlights, truncation = True, max_length=128)

        inputs["labels"] = labels["input_ids"]
        '''
    полученный батч: {
    "input_ids": [...],           # входные токены
    "attention_mask": [...],      # маска внимания
    "labels": [...]                # токены суммированного текста
    }
        '''
        return inputs

    tokenized_train = train.map(tokenize_funct, batched = True)
    tokenized_valid = valid.map(tokenize_funct, batched = True)
    #tokenized_test = test.map(tokenize_funct, batched = True)

    model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')

    training_args = Seq2SeqTrainingArguments(output_dir= 'trainer_logs',
                                    logging_strategy="steps",
                                    logging_steps = 24,
                                    #eval_strategy= 'epoch',
                                    per_device_train_batch_size = 16, #2,
                                    #per_device_eval_batch_size = 1,
                                    #gradient_accumulation_steps = 4,
                                    #eval_accumulation_steps=4,
                                    predict_with_generate=True,
                                    num_train_epochs = 5,
                                    gradient_checkpointing = True,
                                    report_to = 'none',
                                    #torch_empty_cache_steps= 4,
                                    fp16 = True,
                                    #fp16_full_eval = True,
                                    weight_decay = 0.01,
                                    learning_rate = 5e-03)


    def compute_metrics(eval_preds):
        predictions, labels = eval_preds

        decod_preds = tokenizer.batch_decode(predictions, skip_special_tokens = True)
        #преобразование исходных текстов с учётом padding'ов
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decod_labels = tokenizer.batch_decode(labels, skip_special_tokens = True)

        rouge1 = []
        rouge2 = []
        rougeL = []
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        for p, l in zip(decod_preds, decod_labels):
          score = scorer.score(p.strip(), l.strip())
          rouge1.append(score["rouge1"].fmeasure)
          rouge2.append(score["rouge2"].fmeasure)
          rougeL.append(score["rougeL"].fmeasure)

        return {"rouge1": np.mean(rouge1),
                "rouge2": np.mean(rouge2),
                "rougeL": np.mean(rougeL)}

    collator = DataCollatorForSeq2Seq(model = model, tokenizer = tokenizer, padding = "longest")

    trainer = Seq2SeqTrainer(model = model,
                    args = training_args,
                    train_dataset = tokenized_train,
                    #eval_dataset = tokenized_valid,
                    data_collator = collator,
                    compute_metrics = compute_metrics)

    trainer.train()

    def batch_evaluate(trainer, eval_data, batch_size = 16):
        rouge1 = []
        rouge2 = []
        rougeL = []
        counts = []
        for i in range(0, len(eval_data), batch_size):
          batch = eval_data.select(range(i, min(i+batch_size, len(eval_data))))
          metrics = trainer.evaluate(eval_dataset = batch)
          rouge1.append(metrics["eval_rouge1"])
          rouge2.append(metrics["eval_rouge2"])
          rougeL.append(metrics["eval_rougeL"])
          counts.append(len(batch))

        def sum_rouge(rouges):
          return np.sum(np.array(rouges) * np.array(counts)) / np.sum(counts)

        return {"rouge1": sum_rouge(rouge1),
                "rouge2": sum_rouge(rouge2),
                "rougeL": sum_rouge(rougeL)}

    metrics = batch_evaluate(trainer, tokenized_valid)
    print(metrics)
    return model


train = train.shuffle(seed=42).select(range(1000))
valid = valid.shuffle(seed=42).select(range(1000))
model = fine_tuning()

save_directory = './pretrained_model'
model.save_pretrained(save_directory)

