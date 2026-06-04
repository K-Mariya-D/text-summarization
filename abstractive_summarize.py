import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from transformers import (T5Tokenizer, T5ForConditionalGeneration,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          DataCollatorForSeq2Seq, TrainerCallback)
from peft import LoraConfig, get_peft_model
import evaluate
import os
from transformers.trainer_utils import get_last_checkpoint

class CheckMetrics(TrainerCallback):
    """Callback для отслеживания train/val loss с отрисовкой и ранней остановкой"""

    def __init__(self, output_dir):
      self.patience = 5
      self.output_dir = output_dir

      if os.path.exists(self.output_dir + 'loss_values.csv'):
        self.df = pd.read_csv(self.output_dir + 'loss_values.csv')
        self.best_rougeL = self.df['rougeL'].max()
      else:
        self.df = pd.DataFrame(columns= ['x', 'train_loss','val_loss', 'rougeL', 'patience'])
        self.best_rougeL = -1

    def on_evaluate(self, args, state, control, metrics=None,**kwargs):
        # Сохраняем epoch и train loss
        i = len(self.df)
        train_loss = None
        for log in reversed(state.log_history):
            if "loss" in log and "eval_loss" not in log:
                train_loss = log["loss"]
                break
        self.df.loc[i] = {'x': state.epoch,
                                'train_loss': train_loss,
                                'val_loss': metrics['eval_loss'],
                                'rougeL': metrics['eval_rougeL'],
                                'patience': True} #True по умолчанию = ухудшение метрики есть

        # Очистка предыдущего вывода
        clear_output(wait=True)

        # Создаем новый график
        plt.figure(figsize=(7, 3))
        plt.plot(self.df.x, self.df.train_loss, label='train loss')
        plt.plot(self.df.x, self.df.val_loss, label='valid loss')

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training progress")
        plt.legend()
        plt.show()

        if (i > 0 and (self.df.loc[i, 'rougeL'] >= self.best_rougeL)):
          self.df.loc[i, 'patience'] = False #Ухудшений метрики нет!
          self.best_rougeL = self.df.loc[i, 'rougeL']
        self.df.to_csv(self.output_dir + 'loss_values.csv', index=False)

        # Early stopping
        if (self.df.patience.tail(self.patience).sum() == self.patience):
            control.should_training_stop = True
            plt.savefig(self.output_dir + "loss_plot.png")
            return control

class AbstactiveSummarizer(): 
    """Класс для работы с моделью Seq2Seq для абстрактивной суммаризации текста."""
    
    def __init__(self, train_data, valid_data, test_data, output_dir):
        """Должны подаваться датасеты с коллонками "article" и "highlights"."""
        self.train = train_data
        self.valid = valid_data
        self.test = test_data
        self.output_dir = output_dir

        self.tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-small')
        base_model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
        base_model = base_model.to("cuda")

        config = LoraConfig(task_type= "SEQ_2_SEQ_LM" ,
                        r= 4,
                        lora_alpha= 16,
                        target_modules=[ "q" , "v" ],
                        lora_dropout= 0.01)

        self.model = get_peft_model(base_model, config)
        self.rouge = evaluate.load("rouge")
        
    def __compute_metrics(self, eval_preds):
        """Функция для расчёта матрик ROUGE, используемая в trainer.
        Возвращает среднее метрик ROUGE для батча."""
        predictions, labels = eval_preds
        predictions = np.where(predictions < 0, self.tokenizer.pad_token_id, predictions)
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = self.rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    def __tokenize_funct(self, examples):
        """Функция для предварительной токенизации текста."""
        articles = examples['article']
        highlights = examples['highlights']

        #преобразует список строк в словарь с input_ids, attention_mask.
        inputs = self.tokenizer(articles, truncation = True, max_length=384)
        labels = self.tokenizer(text_target = highlights, truncation = True, max_length=128)

        inputs["labels"] = labels["input_ids"]

        #полученный батч: {
        #"input_ids": [...],           входные токены
        #"attention_mask": [...],      маска внимания
        #"labels": [...]               токены суммированного текста}
        return inputs

    def fine_tuning(self):
        """Функция для дообучения модели на поданном датасете."""
        tokenized_train = self.train.map(self.__tokenize_funct, batched = True)
        tokenized_valid = self.valid.map(self.__tokenize_funct, batched = True)
        tokenized_test = self.test.map(self.__tokenize_funct, batched = True)

        training_args = Seq2SeqTrainingArguments(output_dir= self.output_dir + 'trainer_logs2',
                                        save_strategy="epoch",
                                        save_total_limit=5,
                                        load_best_model_at_end=True,
                                        eval_strategy = 'epoch',
                                        metric_for_best_model = 'eval_rougeL',
                                        per_device_train_batch_size = 32,
                                        per_device_eval_batch_size= 32,
                                        generation_max_length = 128,
                                        predict_with_generate=True,
                                        num_train_epochs = 100,
                                        report_to = 'none',
                                        fp16 = True,
                                        weight_decay = 0.01,
                                        learning_rate = 1e-04)

        collator = DataCollatorForSeq2Seq(model = self.model, tokenizer = self.tokenizer, padding = "longest")

        trainer = Seq2SeqTrainer(model = self.model,
                        args = training_args,
                        train_dataset = tokenized_train,
                        eval_dataset = tokenized_valid,
                        data_collator = collator,
                        compute_metrics = self.__compute_metrics)

        trainer.add_callback(CheckMetrics(self.output_dir))

        checkpoint = get_last_checkpoint(training_args.output_dir)

        if checkpoint:
            print("Resuming from:", checkpoint)
        else:
            print("Starting training from scratch")

        trainer.train(resume_from_checkpoint=checkpoint)

        metrics = trainer.evaluate()
        #сохранение метрик
        with open(self.output_dir + 'metrics1.txt','w') as f:
          for key in metrics.keys():
            f.write(f"{key}: {metrics[key]}\n")

        print(metrics)

        return self.model