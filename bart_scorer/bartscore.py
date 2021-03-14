import logging
import transformers
from bert_score import score
import bert_score

# @inproceedings{bert-score,
#   title={BERTScore: Evaluating Text Generation with BERT},
#   author={Tianyi Zhang* and Varsha Kishore* and Felix Wu* and Kilian Q. Weinberger and Yoav Artzi},
#   booktitle={International Conference on Learning Representations},
#   year={2020},
#   url={https://openreview.net/forum?id=SkeHuCVFDr}
# }

class BartScorer():
    def __init__(self):
        self.bart_scorer = bert_score.BERTScorer(model_type="facebook/bart-large-cnn")

    def calc_bart_score(self, candidates, references):
        # turn verbose=True on if we want status updates such as "preparing IDF dict"
        P, R, F1 = self.bart_scorer.score(candidates, references)
        return F1.mean()

# Test
BartScorer = BartScorer()
print(BartScorer.calc_bart_score(["hi hello"], ["hello hi"]))