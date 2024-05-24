import torch
from model import load_pretrained_bert_model
import json
from moviepy.editor import VideoFileClip
import eng_to_ipa as p
import string

# This requires an older version of Pillow < 10

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
video = video.resize((64, 64)) # code hint not showing up - does this method still work?

max_block_size = 20
max_time_diff = 5 # start with 5 seconds to reduce number of input values to <1m
seq_length = 256

phoneme_chars = ["ə", "eɪ", "ɑ", "æ", "ɔ", "ʌ", "aʊ", "aɪ", "ʧ", "ð", "ɛ", "ər", "h", "ɪ", "ʤ", "ŋ", "oʊ", "ɔɪ", "ʃ", "θ", "ʊ", "u", "ʒ", "i", "j"]
regular_chars = list(string.ascii_lowercase)
all_chars = list(set(regular_chars + phoneme_chars))

def string_to_tokens(s, mapping):
    i = 0
    tokens = []

    while i < len(s):
        if s[i:i+2] in mapping:  # Check for 2-character token
            tokens.append(mapping[s[i:i+2]])
            i += 2
        elif s[i] in mapping:    # Check for 1-character token
            tokens.append(mapping[s[i]])
            i += 1
        else:
            raise ValueError(f"Character sequence not found in mapping.")
    return tokens

ptt = {all_chars[i]: i for i in range(len(all_chars))} # phonemes to tokens
ttp = {i: all_chars[i] for i in range(len(all_chars))} # tokens to phonemes

def string_to_tokens(s, mapping):
    # Remove punctuation and spaces
    s = ''.join(char for char in s if char not in string.punctuation and char not in string.whitespace)
    
    i = 0
    tokens = []
    while i < len(s):
        if s[i:i+2] in mapping:  # Check for 2-character token
            tokens.append(mapping[s[i:i+2]])
            i += 2
        elif s[i] in mapping:    # Check for 1-character token
            tokens.append(mapping[s[i]])
            i += 1
        else:
            raise ValueError(f"Character sequence not found in mapping.")
    return tokens

def get_batch(split): # returns (batch_size, seq_length, 64, 64, 3), (batch_size, seq_length), (batch_size, 1)
    split_data = train_split
    if split == "text":
        split_data = test_split
    ix = torch.randint(len(split_data), (batch_size, ))
    x_frames = []
    x_text = []
    y = []
    for i in ix:
        if len(y) >= batch_size:
            break

        jx = torch.randint(high=min(i + max_block_size, len(split_data) - 1), size=(1, ), low=i) # pick a random length that could work
        for j in jx:
            if len(y) >= batch_size:
                break

            frames = torch.zeros(seq_length, 64, 64, 3) # just to fit the image embedding size

            phrase = " ".join(split_data[k]["word"].lower() for k in range(i, j))
            phonemes = p.convert(phrase, stress_marks=None)
            phonemes = phonemes.replace("*", "")
            phonemes = string_to_tokens(phonemes, ptt)

            start_time = split_data[i]["start"]
            end_time = split_data[j]["end"]

            if end_time - start_time > max_time_diff:
                break

            subclip = video.subclip(start_time, end_time)
            for idx, frame in enumerate(subclip.iter_frames()):
                frames[idx] = torch.from_numpy(frame)
            
            for chunk_end in range(len(phonemes)):
                if len(y) >= batch_size:
                    break

                expanded_input_tokens = phonemes[:chunk_end]
                expanded_input_tokens.extend([0] * (seq_length - len(expanded_input_tokens)))
                x_text.append(expanded_input_tokens)
                y.append(phonemes[chunk_end])
                x_frames.append(frames)
    return x_frames, x_text, y

print(get_batch("train"))