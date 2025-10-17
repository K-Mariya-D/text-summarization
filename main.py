from datasets import load_dataset
import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
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

#Preprocessing
#clear_article = train_data["article"].str.lower()
#clear_article = clear_article.str.replace(r'[^\w\s\.\?!]', '', regex = True)
#clear_highlights = train_data["highlights"].str.lower()
#clear_highlights = clear_highlights.str.replace(r'[^\w\s\.\?!]', '', regex = True)

#article_tokens = clear_article.apply(nltk.sent_tokenize)
#highlights_tokens = clear_highlights.apply(nltk.sent_tokenize)

#Extractive summarization
summarizer1 = TextRankSummarizer()
summarizer2 = LsaSummarizer()
textRank_highlights = []
lsa_highlights = []
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
keys = ["TR highlight", "LSA highlight", "Reference highlight", "TR rouge1", "TR rouge2", "TR rougeL", "LSA rouge1", "LSA rouge2", "LSA rougeL"]
scores_tabel = {key: [] for key in keys}
for i in range(1000):
    article_parser = PlaintextParser.from_string(train_data["article"].iloc[i], Tokenizer("english"))
    highlight_parser = PlaintextParser.from_string(train_data["highlights"].iloc[i], Tokenizer("english"))

    summary_length = len(highlight_parser.document.sentences)

    textRank_highlights.append(" ".join(str(sent) for sent in summarizer1(article_parser.document, summary_length)))
    lsa_highlights.append(" ".join(str(sent) for sent in summarizer2(article_parser.document, summary_length)))
    
    #ROUGE-score
    tr_h = textRank_highlights[i]
    lsa_h = lsa_highlights[i]
    h = train_data["highlights"].iloc[i]
    scores_tabel["TR highlight"].append(tr_h)
    scores_tabel["LSA highlight"].append(lsa_h)
    scores_tabel["Reference highlight"].append(h)
    tr_sc = scorer.score(tr_h, h)
    lsa_sc = scorer.score(lsa_h, h)
    for key in tr_sc:
        scores_tabel["TR " + key].append(tr_sc[key].fmeasure)
        scores_tabel["LSA " + key].append(lsa_sc[key].fmeasure)
    
scores = pd.DataFrame(scores_tabel)
print(scores.head())
print("INFO ABOUT EXTRACTIVE SUMMARY")
print(f"Mean TR ROUGE-1: {scores["TR rouge1"].mean()}")
print(f"Mean TR ROUGE-2: {scores["TR rouge2"].mean()}")
print(f"Mean TR ROUGE-L: {scores["TR rougeL"].mean()}") 
print(f"Mean LSA ROUGE-1: {scores["LSA rouge1"].mean()}")
print(f"Mean LSA ROUGE-2: {scores["LSA rouge2"].mean()}") 
print(f"Mean LSA ROUGE-L: {scores["LSA rougeL"].mean()}")    
    