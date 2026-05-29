import os
import json
import re
import sys
from pathlib import Path
from tqdm import tqdm
from groq import Groq  

# 1. Configauration
KNOWLEDGE_DIR = "data\\knowledgebase"
OUTPUT_DIR = "data\\annotated"
GROQ_API_KEY = "" # Your acquired Groq API key 
MODEL_NAME = "llama-3.3-70b-versatile" 
NUM_QA_PER_DOC = 25
MAX_CHARS = 10000 

client = Groq(api_key=GROQ_API_KEY)
os.makedirs(OUTPUT_DIR, exist_ok=True)

#2. Functions
def clean_document(text: str) -> str:
    text = re.sub(r'SOURCE:.*?\nTITLE:.*?\n-+', '', text, flags=re.DOTALL)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()[:MAX_CHARS]

PROMPT_TEMPLATE = """You are a Data Annotation Expert. 

TASK: Create {num} Question-Answer pairs based on the text provided (you can be flexible with the number within {num} +- 10 question-answer pairs depending on how long the given document is).

STRICT LANGUAGE RULES:
1. If the input text is in ENGLISH -> The Questions and Answers MUST be in ENGLISH.
2. If the input text is in VIETNAMESE -> The Questions and Answers MUST be in VIETNAMESE.
3. DO NOT translate. Use the language of the source document.

REQUIREMENTS:
- Questions don't need to be short like answers, but answers must be concise (1-20 words).
- Format: Q: [Question] | A: [Answer]
- No numbering, no introductory text.

DOCUMENT:
{document}
"""

def call_groq_api(prompt: str):
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1000,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"   [!] Error: {e}")
        return None

def parse_robust(text):
    qa_list = []
    # Regex bắt cặp Q: A: linh hoạt
    pattern = re.compile(r"[Qq]:\s*(.*?)\s*[|:-]\s*[Aa]:\s*(.*)")
    for line in text.split('\n'):
        line = re.sub(r'^\d+[\.\s\-]*', '', line.strip()) # Xóa số thứ tự nếu AI tự thêm
        match = pattern.search(line)
        if match:
            qa_list.append({"question": match.group(1).strip(), "answer": match.group(2).strip()})
    return qa_list

def generate_qa_for_file(file_path: Path):
    print(f"Processing: {file_path.name}")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    clean_content = clean_document(content)
    prompt = PROMPT_TEMPLATE.format(num=NUM_QA_PER_DOC, document=clean_content)

    response = call_groq_api(prompt)
    if not response: return []

    qa_pairs = parse_robust(response)
    
    if qa_pairs:
        output_file = Path(OUTPUT_DIR) / f"{file_path.stem}_qa.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        print(f"[OK] Generated {len(qa_pairs)} pairs")
    else:
        print(f"[!] Failed to parse response for {file_path.name}")
        
    return qa_pairs

# 3. Main Execution
if __name__ == "__main__":
    all_qa = []
    txt_files = list(Path(KNOWLEDGE_DIR).glob("*.txt"))
    
    for txt_file in tqdm(txt_files):
        qa = generate_qa_for_file(txt_file)
        all_qa.extend(qa)

    if all_qa:
        with open(Path(OUTPUT_DIR) / "all_qa.json", "w", encoding="utf-8") as f:
            json.dump(all_qa, f, ensure_ascii=False, indent=2)
        print(f"\nCreated {len(all_qa)} question-answer pairs.")