from datasets import load_dataset
import os
from abstractive_summarize import AbstactiveSummarizer
import extractiv_summarize as extrsummy
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


#Check data
def info_about_train(train_data): 
  
  def compute_lengths(batch):
    return {
        "article_len_chars": [len(x) for x in batch["article"]],
        "highlights_len_chars": [len(x) for x in batch["highlights"]],
        "article_len_words": [len(x.split()) for x in batch["article"]],
        "highlights_len_words": [len(x.split()) for x in batch["highlights"]],
    }

  trains_lengths = train_data.map(compute_lengths, batched=True)

  mean_art_s = sum(trains_lengths["article_len_chars"]) / len(trains_lengths)
  mean_hgl_s = sum(trains_lengths["highlights_len_chars"]) / len(trains_lengths)
  mean_art_w = sum(trains_lengths["article_len_words"]) / len(trains_lengths)
  mean_hgl_w = sum(trains_lengths["highlights_len_words"]) / len(trains_lengths)
  compression_percent = ((mean_art_s - mean_hgl_s)/mean_art_s) * 100

  print("\nINFO ABOUT TRAIN DATA")
  print(f"Mean article size in simbols: {round(mean_art_s, 2)}")
  print(f"Mean highlights size in simbols: {round(mean_hgl_s, 2)}")
  print(f"Mean article size in words: {round(mean_art_w, 2)}")
  print(f"Mean highlights size in words: {round(mean_hgl_w, 2)}")
  print(f"The highlight smaller then article on {round(compression_percent, 2)} %")
  
def main(): 
  #Load dataset
  train_data = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train")
  valid_data = load_dataset("abisee/cnn_dailymail", "3.0.0", split="validation")
  test_data = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
  train_data = train_data.remove_columns("id")
  valid_data = valid_data.remove_columns("id")
  test_data = test_data.remove_columns("id")

  print(f"train size = {len(train_data)}")
  print(f"valid size = {len(valid_data)}")

  info_about_train(train_data)

  count = len(test_data)
  lsa_h = extrsummy.extractive_summarize(test_data, count)
  scores = extrsummy.f1_rouge_score(lsa_h, test_data["highlights"], count)
  print("INFO ABOUT EXTRACTIVE SUMMARY")
  print(f"Mean LSA ROUGE-1: {sum(scores["rouge1"])/count}")
  print(f"Mean LSA ROUGE-2: {sum(scores["rouge2"])/count}")
  print(f"Mean LSA ROUGE-L: {sum(scores["rougeL"])/count}")

  summator = AbstactiveSummarizer(train_data=train_data, valid_data=valid_data, test_data=test_data, output_dir = './logs')
  model = summator.fine_tuning()

  save_directory = './pretrained/'
  model.save_pretrained(save_directory + "pretrained_lora_adapter")
  merged_model = model.merge_and_unload()
  merged_model.save_pretrained(save_directory + 'pretrained_model')

if __name__ == '__main__':
    main() 
