import logging
import transformers
from bert_score import score

# @inproceedings{bert-score,
#   title={BERTScore: Evaluating Text Generation with BERT},
#   author={Tianyi Zhang* and Varsha Kishore* and Felix Wu* and Kilian Q. Weinberger and Yoav Artzi},
#   booktitle={International Conference on Learning Representations},
#   year={2020},
#   url={https://openreview.net/forum?id=SkeHuCVFDr}
# }


def calc_bert_score(candidate, reference):
    transformers.tokenization_utils.logger.setLevel(logging.ERROR)
    transformers.configuration_utils.logger.setLevel(logging.ERROR)
    transformers.modeling_utils.logger.setLevel(logging.ERROR)

    # turn verbose=True on if we want status updates such as "preparing IDF dict"
    P, R, F1 = score([candidate], [reference], lang='en', rescale_with_baseline=True)
    return P, R, F1


print(calc_bert_score("hi", "hello"))
