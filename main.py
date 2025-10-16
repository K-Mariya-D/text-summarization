from datasets import load_dataset
import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize

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
clear_article = train_data["article"].str.lower()
clear_article = clear_article.str.replace(r'[^\w\s\.\?!]', '', regex = True)
clear_highlights = train_data["highlights"].str.lower()
clear_highlights = clear_highlights.str.replace(r'[^\w\s\.\?!]', '', regex = True)

article_tokens = clear_article.apply(nltk.sent_tokenize)
highlights_tokens = clear_highlights.apply(nltk.sent_tokenize)
