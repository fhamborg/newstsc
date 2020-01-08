import os, sys
import argparse

import torch
import torch.nn.functional as F

cur_path = os.path.dirname(os.path.realpath(__file__))
par_path = os.path.dirname(cur_path)
sys.path.append(cur_path)
sys.path.append(par_path)

from newstsc.fxlogger import get_logger
from newstsc.train import prepare_and_start_instructur, parse_arguments


class TargetSentimentClassifier:
    def __init__(self, model_name, pretrained_model_name=None, state_dict=None, device=None):
        default_opts = parse_arguments(override_args=True)
        default_opts.model_name = model_name
        default_opts.pretrained_model_name = pretrained_model_name
        default_opts.state_dict = state_dict
        default_opts.device = device

        self.logger = get_logger()

        # set training_mode to False so that we get the Instructor object
        default_opts.training_mode = False

        # prepare and initialize instructor
        instructor = prepare_and_start_instructur(default_opts)

        # get stuff that we need from the instructor
        self.model = instructor.model
        self.tokenizer = instructor.tokenizer
        self.opt = instructor.opt
        self.polarities_inv = instructor.polarity_associations_inv

        # set model to evaluation mode (disable gradient / learning)
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    @classmethod
    def create_default(cls, device=None):
        return cls('lcf_bert', 'bert_news_ccnc_10mio_3ep', 'lcf_bert_newstsc_val_recall_avg_0.5954_epoch3', device)

    def infer(self, text_left: str = None, target_mention: str = None, text_right: str = None, text: str = None,
              target_mention_from: int = None, target_mention_to: int = None):
        """
        Calculates sentiment as to target_mention in a text that is a concatenation of text_left,
        target_mention, and text_right. Note that text_left and text_right should end with a space (or comma, etc.)),
        or end with a space, respectively. Alternatively, the target can be selected via target_mention_from and
        target_mention_to in text.
        """
        assert not text_left and text or text_left and not text

        if text:
            text_left = text[:target_mention_from]
            target_mention = text[target_mention_from:target_mention_to]
            text_right = text[target_mention_from:]

        # assert text_left.endswith(' ') # we cannot handle commas, if we have this check
        assert not target_mention.startswith(' ') and not target_mention.endswith(' '), f"target_mention={target_mention}; text={text}"
        # assert text_right.startswith(' ')

        indexed_example = self.tokenizer.create_text_to_indexes(text_left, target_mention, text_right,
                                                                False)
        inputs = [torch.tensor([indexed_example[col]], dtype=torch.int64).to(self.opt.device) for col in
                  self.opt.input_columns]

        # invoke model
        outputs = self.model(inputs)
        class_probabilites = F.softmax(outputs, dim=-1).reshape((3,)).cpu().tolist()

        classification_result = []
        for class_id, class_prob in enumerate(class_probabilites):
            classification_result.append(
                {'class_id': class_id, 'class_label': self.polarities_inv[class_id], 'class_prob': class_prob})

        classification_result = sorted(classification_result, key=lambda x: x['class_prob'], reverse=True)

        return classification_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='lcf_bert', type=str)
    parser.add_argument('--pretrained_model_name', type=str, default='bert_news_ccnc_10mio_3ep',
                        help='has to be placed in folder pretrained_models')
    parser.add_argument('--state_dict', type=str, default='lcf_bert_newstsc_val_recall_avg_0.5954_epoch3')
    parser.add_argument('--device', default=None, type=str,
                        help='e.g., cuda:0; if None: any CUDA device will be used if available, else CPU')

    opt = parser.parse_args()

    tsc = TargetSentimentClassifier(opt.model_name, opt.pretrained_model_name, opt.state_dict)
    print(tsc.infer(text_left="Mr. Trump said that ", target_mention="Barack Obama", text_right=" was a lias.")[0])
    print(tsc.infer(text_left="Whatever you think of  ", target_mention="President Trump",
                    text_right=", you have to admit that he’s an astute reader of politics.")[0])
    print(tsc.infer(text=
                    "A former employee of the Senate intelligence committee, James A. Wolfe, has been arrested on charges of lying to the FBI about contacts with multiple reporters and appeared in federal court Friday in Baltimore.",
                    target_mention_from=56, target_mention_to=70)[0])
