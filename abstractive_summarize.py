import numpy as np
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
from transformers import (T5Tokenizer, T5ForConditionalGeneration,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          DataCollatorForSeq2Seq, TrainerCallback)

class  CheckMetrics( TrainerCallback ):
    """Класс обратного вызова для отслеживания изменения функции потерь во время обучения и досрочного прекращения обучения."""

    def __init__(self, trainer, tokenized_valid):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 3))
        self.x = []
        self.train_loss_y = []
        self.val_loss_y = []
        self.train_graph, = self.ax.plot(self.x, self.train_loss_y, label = 'train loss')
        self.valid_graph, = self.ax.plot(self.x, self.val_loss_y, label = 'valid loss')
        self.ax.legend()
        self.trainer = trainer
        self.tokenized_valid = tokenized_valid

    def on_epoch_end(self, args, state, control, **kwargs):
        self.x.append(state.epoch)
        self.train_loss_y.append(state.log_history[-1].get('loss'))

        val_metrics = self.trainer.evaluate(self.tokenized_valid.shuffle(seed=42).select(range(64)))
        self.val_loss_y.append(val_metrics['eval_loss'])

        self.train_graph.set_data(self.x, self.train_loss_y)
        self.valid_graph.set_data(self.x, self.val_loss_y)

        self.ax.relim()
        self.ax.autoscale_view()

         # Отобразить новые данный
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)

        if (len(self.x) > 1 and (self.train_loss_y[-1] > self.train_loss_y[-2]
                                 or self.val_loss_y[-1] > self.val_loss_y[-2])):
          control.should_training_stop = True
          plt.ioff()
          plt.savefig('loss_plot.png')
          plt.show()
          return control

class AbstactiveSummarizer(): 
    """Класс для работы с моделью Seq2Seq для абстрактивной суммаризации текста."""
    
    tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-small')
    model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')

    def __init__(self, train_data, valid_data):
        """Должны подаваться датасеты с коллонками "article" и "highlights"."""
        self.train = train_data
        self.valid = valid_data

    def __compute_metrics(self, eval_preds):
        """Функция для расчёта матрик ROUGE, используемая в trainer.
        Возвращает среднее метрик ROUGE для батча."""
        predictions, labels = eval_preds

        decod_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens = True)
        #преобразование исходных текстов с учётом padding'ов
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decod_labels = self.tokenizer.batch_decode(labels, skip_special_tokens = True)

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

    def __batch_evaluate(self, trainer, eval_data, batch_size = 16):
            """Функция для проведения оценки на валидационном датасете в конце обучения.
            Возвращает среднее значение метрик ROUGE для всех примеров из датасета."""
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
    
    def __tokenize_funct(self, examples):
        """Функция для предварительной токенизации текста."""
        articles = examples['article']
        highlights = examples['highlights']

        #преобразует список строк в словарь с input_ids, attention_mask.
        inputs = self.tokenizer(articles, truncation = True, max_length=256)
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

        training_args = Seq2SeqTrainingArguments(output_dir= 'trainer_logs',
                                        logging_strategy="steps",
                                        logging_steps = 64,
                                        per_device_train_batch_size = 64, 
                                        #gradient_accumulation_steps = 4,
                                        predict_with_generate=True,
                                        num_train_epochs = 100,
                                        gradient_checkpointing = True,
                                        report_to = 'none',
                                        fp16 = True,
                                        weight_decay = 0.01,
                                        learning_rate = 5e-03)

        collator = DataCollatorForSeq2Seq(model = self.model, tokenizer = self.tokenizer, padding = "longest")

        trainer = Seq2SeqTrainer(model = self.model,
                        args = training_args,
                        train_dataset = tokenized_train,
                        data_collator = collator,
                        compute_metrics = self.__compute_metrics)

        trainer.add_callback(CheckMetrics(trainer, tokenized_valid))
        trainer.train()

        #metrics = self.__batch_evaluate(trainer, tokenized_valid)
        #print(metrics)
        return self.model