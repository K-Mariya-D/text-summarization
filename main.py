from datasets import load_dataset
import os
from abstractive_summarize import AbstactiveSummarizer
import extractiv_summarize as extrsummy
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


#Check data
def info_about_train(train_data):

  #article_lens = train_data.map(lambda ex: {'text_len' : [len(t) for t in ex["article"]]}, batched=True)
  #highlights_lens = train_data.map(lambda ex: {'text_len': [len(t) for t in ex["highlights"]]}, batched=True)
  mean_art_len = mean_hgl_len = 0
  for i in range(0, len(train_data)):
    mean_art_len += len(train_data[i]["article"])
    mean_hgl_len += len(train_data[i]["highlights"])

  mean_art_len /= len(train_data)
  mean_hgl_len /= len(train_data)
  compression_percent = mean_hgl_len/mean_art_len * 100

  print("\nINFO ABOUT TRAIN DATA")
  print(f"Mean article size: {round(mean_art_len, 2)}")
  print(f"Mean highlights size: {round(mean_hgl_len, 2)}")
  print(f"The article is decreasing on {round(compression_percent, 2)} %")

def main(): 
  #Load dataset
  train_data = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train")
  valid_data = load_dataset("abisee/cnn_dailymail", "3.0.0", split="validation")
  #test_data = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")

  train_data = train_data.remove_columns("id")
  valid_data = valid_data.remove_columns("id")

  print(f"train size = {len(train_data)}")
  print(f"valid size = {len(valid_data)}")

  #info_about_data()

  #count = 100
  #lsa_h = extrsummy.extractive_summarize(train_data, count)
  #scores = extrsummy.f1_rouge_score(lsa_h, train_data["highlights"], count)
  #print("INFO ABOUT EXTRACTIVE SUMMARY")
  #print(f"Mean LSA ROUGE-1: {sum(scores["rouge1"])/count}")
  #print(f"Mean LSA ROUGE-2: {sum(scores["rouge2"])/count}")
  #print(f"Mean LSA ROUGE-L: {sum(scores["rougeL"])/count}")

  summator = AbstactiveSummarizer(train_data=train_data, valid_data=valid_data)
  model = summator.fine_tuning()

  save_directory = './pretrained_model'
  model.save_pretrained(save_directory)

if __name__ == '__main__':
    main() 
