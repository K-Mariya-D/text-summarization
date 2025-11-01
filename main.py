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

print("INFO ABOUT TRAIN DATA")
print(f"Count of Null in articles: {train_data["article"].isnull().sum()}")
print(f"Count of Null in highlights: {train_data["highlights"].isnull().sum()}")
print(f"Mean article size: {mean_article.round(2)}")
print(f"Mean highlights size: {mean_highlights.round(2)}")
print(f"The article is decreasing on {compression_percent.round(2)} %")

#train_data = train_data.drop("id", axis=1)

#Extractive summarization
def extractive_summarize(data, count = None):
    if count is None:
         count = data.shape[0] 

    summarizer = LsaSummarizer()
    lsa_highlights = []

    for i in range(count):
        article_parser = PlaintextParser.from_string(data["article"][i], Tokenizer("english"))
        highlight_parser = PlaintextParser.from_string(data["highlights"][i], Tokenizer("english"))

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

count = 100
lsa_h = extractive_summarize(train_data, count)
scores = f1_rouge_score(lsa_h, train_data["highlights"], count)
print("INFO ABOUT EXTRACTIVE SUMMARY")
print(f"Mean LSA ROUGE-1: {sum(scores["rouge1"])/count}")
print(f"Mean LSA ROUGE-2: {sum(scores["rouge2"])/count}") 
print(f"Mean LSA ROUGE-L: {sum(scores["rougeL"])/count}")    

'''
from datasets import Dataset 
from transformers import (T5Tokenizer, T5Model)

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
'''


    