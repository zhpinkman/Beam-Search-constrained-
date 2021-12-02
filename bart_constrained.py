from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import pandas as pd
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

df = pd.read_csv(
    '../dialogue_summarization/transcriptsforkeyphraseextraction/2021-07-12 14.35.38 Interscriber Wrapup.m4a.csv')
text = '; '.join(df[df['Speaker'] == 'Mark']['Utterance'].tolist())

inputs = tokenizer([text],
                   max_length=1024, return_tensors='pt')


keys = list(tokenizer.decoder.keys())
values = list(tokenizer.decoder.values())

bad_words_ids = [tokenizer(bad_word, add_prefix_space=True).input_ids for bad_word in [
    "idiot", "stupid", "shut up"]]
print(bad_words_ids)

# # Generate Summary
# summary_ids = model.generate(
#     inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
# print([tokenizer.decode(g, skip_special_tokens=True,
#       clean_up_tokenization_spaces=False) for g in summary_ids])
