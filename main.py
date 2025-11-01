from datasets import load_dataset
import pandas as pd
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from rouge_score import rouge_scorer

#Load dataset
train_data = pd.DataFrame(load_dataset("abisee/cnn_dailymail", "3.0.0", split="train"))
valid_data = pd.DataFrame(load_dataset("abisee/cnn_dailymail", "3.0.0", split="validation"))
test_data = pd.DataFrame(load_dataset("abisee/cnn_dailymail", "3.0.0", split="test"))
print(train_data.head())

#Check data
mean_article = train_data["article"].apply(lambda str: len(str)).mean()
mean_highlights = train_data["highlights"].apply(lambda str: len(str)).mean()
compression_percent = mean_highlights/mean_article * 100

print("INFO ABOUT DATA")
print(f"Count of Null in articles: {train_data["article"].isnull().sum()}")
print(f"Count of Null in highlights: {train_data["highlights"].isnull().sum()}")
print(f"Mean article size: {mean_article.round(2)}")
print(f"Mean highlights size: {mean_highlights.round(2)}")
print(f"The article is decreasing on {compression_percent.round(2)} %")

#train_data = train_data.drop("id", axis=1)


#Extractive summarization
summarizer2 = LsaSummarizer()
lsa_highlights = []
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
keys = ["LSA highlight", "Reference highlight", "LSA rouge1", "LSA rouge2", "LSA rougeL"]
scores_tabel = {key: [] for key in keys}
for i in range(1000):
    article_parser = PlaintextParser.from_string(train_data["article"].iloc[i], Tokenizer("english"))
    highlight_parser = PlaintextParser.from_string(train_data["highlights"].iloc[i], Tokenizer("english"))

    summary_length = len(highlight_parser.document.sentences)

    lsa_highlights.append(" ".join(str(sent) for sent in summarizer2(article_parser.document, summary_length)))
    
    #ROUGE-score
    lsa_h = lsa_highlights[i]
    h = train_data["highlights"].iloc[i]
    scores_tabel["LSA highlight"].append(lsa_h)
    scores_tabel["Reference highlight"].append(h)
    lsa_sc = scorer.score(lsa_h, h)
    for key in lsa_sc:
        scores_tabel["LSA " + key].append(lsa_sc[key].fmeasure)
    
scores = pd.DataFrame(scores_tabel)
print(scores.head())
print("INFO ABOUT EXTRACTIVE SUMMARY")
print(f"Mean LSA ROUGE-1: {scores["LSA rouge1"].mean()}")
print(f"Mean LSA ROUGE-2: {scores["LSA rouge2"].mean()}") 
print(f"Mean LSA ROUGE-L: {scores["LSA rougeL"].mean()}")    

'''
from datasets import Dataset 
from transformers import (T5Tokenizer, T5Model, 
                          TrainingArguments, Trainer)

train = Dataset.from_pandas(train_data)
valid = Dataset.from_pandas(valid_data)
test = Dataset.from_pandas(test_data)

tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-small')

def tokenize_funct(text):
    return tokenizer(text, padding = 'max_length', 
                     truncation=True, return_tensors="pt")

tokenized_train = train["article"].map(tokenize_funct)
tokenized_valid = valid["artice"].map(tokenize_funct)
tokenized_test = test["article"].map(tokenize_funct)

model = T5Model.from_pretrained('google-t5/t5-small')

training_args = TrainingArguments(
	evaluation_strategy = 'epoch',
	per_device_train_batch_size = 6,
	per_device_eval_batch_size = 6,
	num_train_epochs = 5,
	report_to='none')

'''

    