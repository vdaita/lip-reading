import torch
from model import load_pretrained_bert_model
import json
from moviepy.editor import VideoFileClip
from transformers import AutoTokenizer

# This requires an older version of Pillow < 10

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

bert_tiny = load_pretrained_bert_model()
print("# of parameters: ", sum(p.numel() for p in bert_tiny.parameters()))
optimizer = torch.optim.AdamW(bert_tiny.parameters(), lr=1e-3)

batch_size = 32
word_data_file = open("word_data.json", "r")
word_data = json.loads(word_data_file.read())
word_data_file.close()

all_words = []
for segment in word_data["output"]["segments"]:
    all_words.extend(segment["words"])

train_index_end = int(9*len(all_words)/10)
train_split = all_words[:train_index_end]
test_split = all_words[train_index_end:]

print("Sample word data: ", json.dumps(train_split[0], indent=4))

video = VideoFileClip("obama-debt-cropped.mp4")
video = video.resize((64, 64)) # code hint not showing up - does this method still work?

batch_size = 4
max_block_size = 20
max_time_diff = 5 # start with 5 seconds to reduce number of input values to <1m
seq_length = 256

def get_batch(split): # returns (batch_size, seq_length, 64, 64, 3), (batch_size, seq_length), (batch_size, 1)
    split_data = train_split
    if split == "text":
        split_data = test_split
    ix = torch.randint(len(split_data), (batch_size, ))
    x_frames = []
    x_text = []
    y = []
    for i in ix:
        if len(y) > batch_size:
            break
        for j in range(max_block_size):
            if i + j >= len(split_data):
                break
            if len(y) > batch_size:
                break

            frames = torch.zeros(seq_length, 64, 64, 3) # just to fit the image embedding size
            tokens = tokenizer.encode(" ".join(split_data[k]["word"].lower() for k in range(i, i + j)))

            start_time = split_data[i]["start"]
            end_time = split_data[i + j]["end"]

            if end_time - start_time > max_time_diff:
                break

            subclip = video.subclip(start_time, end_time)
            for idx, frame in enumerate(subclip.iter_frames()):
                frames[idx] = torch.from_numpy(frame)

            x_frames.append(frames)
            expanded_input_tokens = tokens[:-1]
            expanded_input_tokens.extend([0] * (seq_length - len(tokens[:-1])))
            x_text.append(expanded_input_tokens)
            y.append(tokens[-1])

            # TODO: append case for knowing whether or not the text has finished (is there a 0 token?)

    return x_frames, x_text, y

# Trying to train the model
max_iters = 1000
for iter in range(max_iters):
    x_frames, x_text, y = get_batch("train")

    x_frames = torch.stack(x_frames)
    x_text = torch.tensor(x_text)
    y = torch.tensor(y)

    logits, loss = bert_tiny(x_frames, x_text, targets=y)
    
    if iter % 10 == 0:
        print(f"Iteration {iter}, Training loss: {loss}")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()