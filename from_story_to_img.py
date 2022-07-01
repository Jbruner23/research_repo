import os
import json
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import numpy as np

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

path_paragraph_align_alice_file = "/Users/jannabruner/Documents/Janna/MSc_IDC_Computer_Science/research/booksum/" \
                                  "alignments/paragraph-level-summary-alignments/chapter_summary_aligned_train_" \
                                  "split.jsonl.gathered.stable"

path_to_chapter_alice_dir = "/Users/jannabruner/Documents/Janna/MSc_IDC_Computer_Science/research/booksum/" \
                                "all_chapterized_books/Alice_5_chapters"

files = os.listdir(path_to_chapter_alice_dir)
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
summaries_per_chapter = []
pick_best_paragraph_chapter = []
paragraph_per_chapt_idx = {}

for file in files:

    with open(os.path.join(path_to_chapter_alice_dir, file), 'r') as file:
        chapter_content = file.read().replace('\n', '')

    article_input_ids = tokenizer.batch_encode_plus([chapter_content], return_tensors='pt', max_length=1024)['input_ids'].to(torch_device)
    summary_ids = model.generate(article_input_ids,
                                 num_beams=4,
                                 length_penalty=2.0,
                                 max_length=76,
                                 min_length=0,
                                 no_repeat_ngram_size=3)

    summary_txt = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
    summaries_per_chapter.append(summary_txt)

# choosing with roBERTa

try:
    fx = open(path_paragraph_align_alice_file, "r")
except Exception as e:
    print(e)

summary_json = json.load(fx)

with open(path_paragraph_align_alice_file) as fd:
    data = [json.loads(line) for line in fd]


summary_para_chapt = []
paragraph_per_chapt_idx = {}

for i in range(1, 6):
    # paragraph_per_chapt_idx[i].update({i: line})
    for j, line in enumerate(data):
        if ("chapter_" + str(i)) in line["title"]:
            if i in paragraph_per_chapt_idx:
                paragraph_per_chapt_idx[i].append(line)
            else:
                paragraph_per_chapt_idx[i] = [line]
    max_score = 0
    max_para_txt = []
    for k in range(len(paragraph_per_chapt_idx[i])):
        index_score = np.argmax(paragraph_per_chapt_idx[i][k]['alignment_scores'])
        max_score_k = paragraph_per_chapt_idx[i][k]['alignment_scores'][index_score]
        if float(max_score_k) > float(max_score):
            max_score = max_score_k
            max_para_txt = paragraph_per_chapt_idx[i][k]['summary'][index_score]

    summary_para_chapt.append(max_para_txt)

