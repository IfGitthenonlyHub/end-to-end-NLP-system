import json
import os

# Load your data
with open("data/annotated/all_qa.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Define paths (as per PDF page 6)
paths = ["data/test", "data/train"]
for p in paths: os.makedirs(p, exist_ok=True)

# Let's put 25 in test, the rest in train
test_data = data[:25]
train_data = data[25:]

def save_to_txt(dataset, folder):
    with open(f"{folder}/questions.txt", "w", encoding="utf-8") as q_file, \
         open(f"{folder}/reference_answers.txt", "w", encoding="utf-8") as a_file:
        for item in dataset:
            q_file.write(item['question'] + "\n")
            a_file.write(item['answer'] + "\n")

save_to_txt(test_data, "data/test")
save_to_txt(train_data, "data/train")

print("Files converted to .txt and split into folders!")