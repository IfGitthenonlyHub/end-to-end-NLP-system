import json
import os
import random

# 1. Configuration
JSON_FILE = "data/annotated/all_qa.json"
TEST_RATIO = 0.20  # 20% for Test, 80% for Train
RANDOM_SEED = 42   # Ensures you get the same result every time you run it

# 2. Language Detection Helper
def is_vietnamese(text):
    # Checks for characters unique to Vietnamese
    vn_chars = "đĐáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ"
    return any(char in vn_chars for char in text)

# 3. Load Data
if not os.path.exists(JSON_FILE):
    print(f"Error: {JSON_FILE} not found!")
    exit()

with open(JSON_FILE, "r", encoding="utf-8") as f:
    all_data = json.load(f)

# 4. Stratify by Language
vn_pairs = [item for item in all_data if is_vietnamese(item['question'])]
en_pairs = [item for item in all_data if not is_vietnamese(item['question'])]

print(f"Total pairs found: {len(all_data)}")
print(f"Detected: {len(vn_pairs)} Vietnamese and {len(en_pairs)} English pairs.")

# 5. Shuffle and Split
random.seed(RANDOM_SEED)
random.shuffle(vn_pairs)
random.shuffle(en_pairs)

vn_split_idx = int(len(vn_pairs) * TEST_RATIO)
en_split_idx = int(len(en_pairs) * TEST_RATIO)

# Create subsets
test_subset = vn_pairs[:vn_split_idx] + en_pairs[:en_split_idx]
train_subset = vn_pairs[vn_split_idx:] + en_pairs[en_split_idx:]

# Final shuffle of the mixed subsets so the files aren't grouped by language
random.shuffle(test_subset)
random.shuffle(train_subset)

# 6. Save to Folders
def save_to_folders(dataset, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/questions.txt", "w", encoding="utf-8") as q_f, \
         open(f"{folder_path}/reference_answers.txt", "w", encoding="utf-8") as a_f:
        for item in dataset:
            # Clean up newlines just in case
            q = item['question'].strip().replace('\n', ' ')
            a = item['answer'].strip().replace('\n', ' ')
            q_f.write(f"{q}\n")
            a_f.write(f"{a}\n")

save_to_folders(test_subset, "data/test")
save_to_folders(train_subset, "data/train")

print("-" * 30)
print(f"SUCCESS!")
print(f"Test set: {len(test_subset)} pairs (Randomly balanced)")
print(f"Train set: {len(train_subset)} pairs (Randomly balanced)")
print("Files are ready in data/test/ and data/train/")