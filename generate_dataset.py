import torch
import json
from moviepy.editor import VideoFileClip
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import pickle
# This requires an older version of Pillow < 10

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
batch_size = 4
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
video = video.set_fps(15)
video = video.resize((64, 64)) # code hint not showing up - does this method still work?

max_block_size = 20
max_time_diff = 5 # start with 5 seconds to reduce number of input values to <1m
seq_length = 256

def process_data(split): # returns (batch_size, seq_length, 64, 64, 3), (batch_size, seq_length), (batch_size, 1)
    split_data = train_split
    if split == "test":
        split_data = test_split

    if not(os.path.exists(f"data/{split}")):
        os.makedirs(f"data/{split}", exist_ok=True)

    for i in tqdm(range(len(split_data))):
            if os.path.exists(f"data/{split}/{str(i)}.pkl"):
                continue

            x_frames = []
            x_text = []
            y = []
            for j in range(max_block_size):
                try:
                    # print("looking at words: ", i, " to ", j)
                    if i + j >= len(split_data):
                        # print(i, j, "greater than split_data")
                        break

                    frames = torch.zeros(seq_length, 64, 64, 3) # just to fit the image embedding size
                    tokens = tokenizer.encode(" ".join(split_data[k]["word"].lower() for k in range(i, i + j)))

                    start_time = split_data[i]["start"]
                    end_time = split_data[i + j]["end"]

                    if end_time - start_time > max_time_diff:
                        # print("max time diff exceeded: ", start_time, end_time)
                        break

                    subclip = video.subclip(start_time, end_time)
                    # Check the number of frames to makes sure that this works
                    idx = 0
                    too_many_frames_flag = False
                    for frame in subclip.iter_frames():
                        if idx >= seq_length:
                            # print("There were too many frames")
                            too_many_frames_flag = True
                            break
                        frames[idx] = torch.from_numpy(frame)
                        idx += 1

                    if too_many_frames_flag:
                        break
                    expanded_input_tokens = tokens[:-1]
                    expanded_input_tokens.extend([0] * (seq_length - len(tokens[:-1])))

                    # print("frames have dimension: ", frames.shape)
                    # print("text has dimension: ", expanded_input_tokens.shape)
                    
                    x_frames.append(frames)
                    x_text.append(expanded_input_tokens)
                    y.append(tokens[-1])
   
                except Exception as e:
                    pass
                    # print("Error: ", e)
                    # print(i)
                    # print(j)
                    # print(split_data[i])
                    # print(split_data[i + j])
            save_file = open(f"data/{split}/{str(i)}.pkl", "wb+")
            pickle.dump((x_frames, x_text, y), save_file)
            save_file.close()

process_data("train")
process_data("split")