import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, MaxLengthCriteria, StoppingCriteriaList
from Beam_Search_Scorer import BeamSearchScorer
from LogitsProcessor import MinLengthLogitsProcessor


tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained("gpt2")

inputs = tokenizer("Hello, my dog is cute and ", return_tensors="pt")
input_ids = inputs['input_ids']


# output = model.generate(
#     **inputs, return_dict_in_generate=True, output_scores=True)


num_beams = 3
num_return_beams = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('batch size: ', input_ids.shape[0])
beam_scorer = BeamSearchScorer(
    batch_size=input_ids.shape[0],
    max_length=model.config.max_length,
    num_beams=num_beams,

    num_beam_hyps_to_keep=num_return_beams,
    device=device,
)

logits_processor = MinLengthLogitsProcessor(
    5, eos_token_id=model.config.eos_token_id
)


outputs = model.beam_search(
    input_ids=torch.cat([input_ids] * num_beams),
    beam_scorer=beam_scorer,
    logits_processor=logits_processor,
    stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=12)])
)

for index, output_tokenized in enumerate(outputs):
    output = tokenizer.decode(output_tokenized)
    print(f'beam {index}: {output}')
