import numpy as np
from transformers import LogitsProcessor
import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer

# src: https://huggingface.co/transformers/v4.1.1/_modules/transformers/generation_logits_process.html


# def set_scores_to_inf_for_banned_tokens(scores, banned_tokens):
#     """
#     Modifies the scores in place by setting the banned token positions to `-inf`. Banned token is expected to be a
#     list of list of banned tokens to ban in the format [[batch index, vocabulary position],...

#     Args:
#         scores: logits distribution of shape (batch size, vocabulary size)
#         banned_tokens: list of list of tokens to ban of length (batch_size)
#     """
#     banned_mask_list = []
#     for idx, batch_banned_tokens in enumerate(banned_tokens):
#         for token in batch_banned_tokens:
#             banned_mask_list.append([idx, token])
#     if not banned_mask_list:
#         return scores

#     banned_mask = torch.LongTensor(banned_mask_list)
#     indices = torch.ones(len(banned_mask))

#     banned_mask = (
#         torch.sparse.LongTensor(banned_mask.t(), indices, scores.size()).to(
#             scores.device).to_dense().bool()
#     )
#     scores = scores.masked_fill(banned_mask, -float("inf"))
#     return scores


class MinLengthLogitsProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsProcessor` enforcing a min-length by setting EOS probability to 0.

    Args:
        min_length (:obj:`int`):
            The minimum length below which the score of :obj:`eos_token_id` is set to :obj:`-float("Inf")`.
        eos_token_id (:obj:`int`):
            The id of the `end-of-sequence` token.
    """

    def __init__(self, min_length: int, eos_token_id: int):
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(
                f"`min_length` has to be a positive integer, but is {min_length}")

        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(
                f"`eos_token_id` has to be a positive integer, but is {eos_token_id}")

        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len < self.min_length:
            scores[:, self.eos_token_id] = -float("inf")
        return scores


# class EvenLogits(LogitsProcessor):
#     def __init__(self, tokenizer: AutoTokenizer):
#         super().__init__()
#         self.tokenizer = tokenizer

#     def __call__(self, input_ids, scores):

#         banned_tokens = []
#         for beam_index, (beam_input_ids, beam_scores) in enumerate(zip(input_ids, scores)):
#             elementwise_length = np.vectorize(len)
#             keys = np.array(list(self.tokenizer.vocab.keys()))
#             values = np.array(list(self.tokenizer.vocab.values()))

#             # indexes of tokens that are too long
#             indexes = np.where(elementwise_length(keys) % 2 == 0)[0]

#             banned_tokens.append(values[indexes])

#         scores = set_scores_to_inf_for_banned_tokens(scores, banned_tokens)
#         return scores


# class ABCLogits(LogitsProcessor):
#     def __init__(self, vocab):
#         """
#         vocab is a dictionary where the keys are tokens
#         and the values are the corresponding ids.
#         """
#         # create an array of tokens
#         # remove the 'Ġ' token (used to represent a blank space in the tokenizer)
#         self.keys = list(tokenizer.vocab.keys())
#         index_to_pop = self.keys.index('Ġ')
#         self.keys.pop(index_to_pop)
#         self.keys = np.array(self.keys)

#         # create an array of ids
#         # also remove the 'Ġ' token
#         self.values = list(tokenizer.vocab.values())
#         self.values.pop(index_to_pop)
#         self.values = np.array(self.values)

#         # vectorized function used to get the first character of a token
#         # ignores leading whitespaces and 'Ġ' tokens
#         def first_char(x): return x.strip('Ġ ')[0].lower()
#         self.first_char = np.vectorize(first_char)

#         # get the indexes of all IDs that do not start with the given letter
#         not_a_indexes = np.where(self.first_char(self.keys) != 'a')
#         not_b_indexes = np.where(self.first_char(self.keys) != 'b')
#         not_c_indexes = np.where(self.first_char(self.keys) != 'c')

#         # create sets of tokens that do not start with 'a', 'b' or 'c'
#         self.not_a_values = self.values[not_a_indexes]
#         self.not_b_values = self.values[not_b_indexes]
#         self.not_c_values = self.values[not_c_indexes]

#     def __call__(self, input_ids, scores):
#         banned_tokens = []
#         # for every beam (partially generated sentence)
#         for beam_index, (beam_input_ids, beam_scores) in enumerate(zip(input_ids, scores)):
#             # get the last token of this beam
#             last_word = tokenizer.decode(beam_input_ids[-1])
#             # get the first character of this last token
#             starting_char = self.first_char(last_word)
#             # if the last token starts with 'a',
#             # ban all words that do not start with 'b', etc.
#             if starting_char == 'a':
#                 banned_tokens.append(self.not_b_values)
#             elif starting_char == 'b':
#                 banned_tokens.append(self.not_c_values)
#             elif starting_char == 'c':
#                 banned_tokens.append(self.not_a_values)
#             else:
#                 banned_tokens.append(self.not_a_values)
#         # set the scores of all banned tokens over the beams to -inf
#         scores = set_scores_to_inf_for_banned_tokens(scores, banned_tokens)
#         return scores
