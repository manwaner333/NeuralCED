import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, AutoConfig
import torch
import numpy as np


# # Load pre-trained GPT-2 model and tokenizer
# model_name = "gpt2"
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPT2LMHeadModel.from_pretrained(model_name)
#
# # Example text
# text = "The capital of France is"
#
# # Encoding: Text -> Tokens -> Token IDs
# input_ids = tokenizer.encode(text, return_tensors='pt')
#
# # Model generates predictions (here, just generating one token for simplicity)
# output = model.generate(input_ids, max_length=len(input_ids[0]) + 5)
#
# # Decoding: Token IDs -> Tokens -> Text
# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
#
# print(generated_text)
#

# device = "cuda"
#
# kwargs = {}
# model_name = 'huggyllama/llama-7b'
# config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
#     low_cpu_mem_usage=True, config=config, **kwargs)
# model.cuda()

# batch_prompts = ['This is a Wikipedia passage about John Russell Reynolds', 'This is a Wikipedia passage about Matthew Aylmer']
# batch_prompts = 'This is a Wikipedia passage about John Russell Reynolds.'
# inputs = tokenizer(batch_prompts, return_tensors="pt").to(device)
# print(inputs)

# model.eval()
# with torch.no_grad():
#     outputs = model(**inputs, output_hidden_states=True, return_dict=True)

# input1 = inputs['input_ids']
# output1 = model(input1)
# next_token1 = torch.argmax(output1['logits'][:, -1], dim=-1)
# print(next_token1)
#
# input2 = torch.cat((inputs['input_ids'], next_token1.unsqueeze(0)), 1)
# output2 = model(input2)
# next_token2 = torch.argmax(output2['logits'][:, -1], dim=-1)
# print(next_token2)

# model.eval()
# with torch.no_grad():
#     outputs = model.generate(inputs['input_ids'], max_length=50, do_sample=True, temperature=0.5, top_p=0.99, top_k=5)
#     text1 = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# print("pad_token")
# print(tokenizer.pad_token)
# print("eos_token")
# print(tokenizer.eos_token)
# question_answer = text1.replace(tokenizer.eos_token, "")
# question_answer_split = question_answer.split(tokenizer.sep_token)
# question = question_answer_split[0].strip()
# print("question")
# print(question)


# model.eval()
# with torch.no_grad():
#     outputs_test = model.generate(inputs['input_ids'], max_length=50, num_beams=5, num_return_sequences=5)
#     text2 = tokenizer.batch_decode(outputs_test, skip_special_tokens=True)
# print("text2")
# print(text2)

#
# res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# print(res)
#
# qingli = 3


# def split_data(list_ori, p):
#     list_new = []
#     list_short = []
#     for i in list_ori:
#         list_short.append(i)
#         if i == p:
#             list_new.append(list_short)
#             list_short = []
#     list_new.append(list_short)
#     return list_new
#
#
# def split_str(str, p):
#     return str.split(p)
#
# str = 'This is a Wikipedia passage about John Russell Reynolds, who was the president of R. J. Reynolds Tobacco Company from 1913 to 1948. Reynolds was born in 186'
#
# print([item + '.' for item in split_str(str, '.')])
#
# list_ori = torch.from_numpy(np.array([[ 1,   910,   338,   263, 14109, 13382,  1048,  2259, 20679, 28555,
#             3361, 29892,  1058,   471,   278,  6673,   310,   390, 29889,   435,
#             29889, 28555,  3361, 22354, 17970,  6938,   515, 29871, 29896, 29929,
#          29896, 29941,   304, 29871, 29896, 29929, 29946, 29947, 29889,    13,
#           1123,   948,  3361,   471,  6345,   297, 29871, 29896, 29947, 29953],
#             [ 1,   910,   338,   263, 14109, 13382,  1048, 22292,   319,  2904,
#           1050, 29892,   263,  4908,  5844, 19217,   713,  1058,   471,   263,
#          29323,   261,   297,   278,  1746,   310, 16712, 10693, 29889,    13,
#           1576, 13382,   338,   263,  1781,  1342,   310,   920,   304,   671,
#          14109,   304,  1284,   714,  1048,   263,  2022, 29889,    13,  1576]]))
#
# a = [split_data(a.detach().cpu().numpy().tolist(), 29889) for a in list_ori]
# qingli = 3



# df = pd.read_csv('processed_datasets/token_text_seqs_wiki_bio_info7b_rmv_period.csv')
# res = []
# for index, row in df.iterrows():
#     token_idx_seqs = eval(row['token_idx_seqs'])
#     text_seqs = eval(row['text_seqs'])
#     token_idx_len = len(token_idx_seqs)
#     text_len = len(text_seqs)
#     if token_idx_len == text_len:
#         for i in range(token_idx_len-1):
#             res.append([row['wiki_bio_text'], row['wiki_bio_test_idx'], row['name'], row['prompt'], row['text_seqs'], token_idx_seqs[i],
#                         text_seqs[i]])
#
#     df = pd.DataFrame(data=res, columns=['wiki_bio_text', 'wiki_bio_test_idx', 'name', 'prompt', "text_seqs", 'token_idx_seqs', 'text_seqs'])
#     df.to_csv('datasets/wiki_bio_sentences_level.csv', index=False)


# 将文本进行切分

# import spacy
#
# input_text = "John Russell Reynolds (1820–1876) was an English lawyer, judge, and author. He was born in London, the son of a barrister, and was educated at Eton College and Trinity College, Cambridge. He was called to the bar in 1845, and became a Queen's Counsel in 1859. He was appointed a judge of the Court of Common Pleas in 1867, and was knighted in 1871. Reynolds was a prolific author, writing on a wide range of topics. He wrote several books on legal topics, including The Law of Libel and Slander (1863), The Law of Copyright (1865), and The Law of Patents for Inventions (1868). He also wrote on a variety of other topics, including history, biography, and literature. He was a frequent contributor to the Saturday Review, and wrote several books on Shakespeare, including The Mystery of William Shakespeare (1848) and The Authorship of Shakespeare (1875). He also wrote a biography of the poet John Keats (1848)."
# nlp = spacy.load("en_core_web_sm")
# doc = nlp(input_text)
# for s in doc.sents:
#     print(s)


# d读取wiki_bio 的数据
# from datasets import load_dataset
#
# # Load the "wiki_bio" dataset
# dataset = load_dataset("wiki_bio")
#
# # Access a specific split (e.g., "train" or "test")
# split_name = "test"  # Replace with the desired split name
# split = dataset[split_name]
#
# table_data = split[62464]['input_text']['table']
#
# name = table_data['content'][table_data['column_header'].index('name')]
# print(name)


# Access individual examples in the split
# for example in split:
#     # Access the relevant fields from the example
#     title = example["title"]
#     text = example["text"]
#
#     # Print or process the data as needed
#     print("Title:", title)
#     print("Text:", text)


# without_change = np.load("processed_datasets/without_change.npy", allow_pickle=True)
# with_change = np.load("processed_datasets/with_change.npy", allow_pickle=True)
#
# qingli = 3


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b")

# Example input text
text = "The quick brown fox jumps over the lazy dog."

# Tokenize input
tokens = tokenizer(text, return_tensors="pt")
input_ids = tokens.input_ids

# Perform a forward pass to get logits
with torch.no_grad():
    outputs = model(**tokens)
    logits = outputs.logits

# Shift the logits and input_ids by one position so that we align the logits with their respective tokens
shifted_logits = logits[:, :-1, :]
shifted_input_ids = input_ids[:, 1:]

# Convert logits to probabilities
log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)

# Gather the log probabilities for the actual next tokens
gathered_log_probs = torch.gather(log_probs, 2, shifted_input_ids.unsqueeze(-1)).squeeze(-1)

# Calculate the average log probability across the sequence
average_log_prob = gathered_log_probs.sum(1) / (input_ids.size(1) - 1)

# Calculate perplexity
perplexity = torch.exp(-average_log_prob)

# Print perplexity
print("Perplexity:", perplexity.tolist())
