from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from tqdm import tqdm

load_dotenv(".env")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

word_data_file = open("word_data.json", "r")
word_data = json.loads(word_data_file.read())
word_data_file.close()

all_words = []
for segment in word_data["output"]["segments"]:
    all_words.extend(segment["words"])

word_list = []

for word in all_words:
    word_list.append(word["word"])

all_words = {}
phonemes_set = set()
phonemes_map = {}

def split_list_into_k_parts(lst, k):
    return [lst[i * len(lst) // k: (i + 1) * len(lst) // k] for i in range(k)]

words_lists = split_list_into_k_parts(word_list, 60)

for wl in tqdm(words_lists):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Given this list of words, output a JSON file where each word is mapped to a list of phonemes. Don't format the phonemes with slashes - just the representative characters themselves."
            },
            {
                "role": "user",
                "content": json.dumps(wl)
            }
        ],
        response_format={"type": "json_object"},
    )

    content = json.loads(response.choices[0].message.content)
    phonemes_map.update(content)

for i in range(len(all_words)):
    all_words[i]["phonemes"] = content[all_words[i]["word"]]
    phonemes_set.update(content[all_words[i]["word"]])

print(list(phonemes_set))

output_file = open("phoneme-data.json", "w+")
output_file.write(json.dumps(all_words, indent=4))
output_file.close()