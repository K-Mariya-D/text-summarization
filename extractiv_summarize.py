from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from rouge_score import rouge_scorer

def extractive_summarize(data, count = None, sum_length = None):
    """Функция для экстрактивной суммаризации на основе LSA.
    Принимает датасеты с коллонками "article" и "highlights" (тогда summary_length расщитывается по уже существующему заголовку).
    Либо можно указать sum_length отдельно, тогда длинна готового заголовка будет игнорироваться.
    Возвращет предстказанные заголовки."""
    if count is None:
         count = len(data)

    summary_length = None
    if sum_length is not None:
         summary_length = sum_length         

    summarizer = LsaSummarizer()
    lsa_highlights = []

    for i in range(count):
        article_parser = PlaintextParser.from_string(data[i]["article"], Tokenizer("english"))
        if (sum_length is None):
            highlight_parser = PlaintextParser.from_string(data[i]["highlights"], Tokenizer("english"))

            summary_length = len(highlight_parser.document.sentences)
        
        lsa_highlights.append(" ".join(str(sent) for sent in summarizer(article_parser.document, summary_length)))

    return lsa_highlights

def f1_rouge_score(new_highlights, old_highlights, count = None):
        """Функция для рассчёта метрик ROUGE по заголовкам."""
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