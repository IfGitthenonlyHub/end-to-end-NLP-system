import os
import re

def final_clean(text):
    # 1. Remove Wikipedia-style citations: [1], [12], [citation needed]
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[citation\sneeded\]', '', text)
    
    # 2. Remove UI common fragments (Add to this list as you find more)
    ui_residues = [
        "Scroll to Top", "Bài tiếp theo", "Thẻ:tuyển sinh", 
        "Đăng bởi:", "Ngày đăng:", "Xem thêm"
    ]
    for junk in ui_residues:
        text = text.replace(junk, "")

    # 3. Clean up Table artifacts
    text = text.replace("[TABLE]", "\n[Data Table]:\n") # Make it more semantic for the LLM
    text = re.sub(r'\|\s+\|', '|', text) # Remove empty table cells
    
    # 4. Remove empty lines or lines with only special characters
    lines = text.split('\n')
    clean_lines = [line.strip() for line in lines if len(line.strip()) > 2]
    
    # 5. Fix whitespace (Collapse 3+ newlines into 2)
    result = "\n\n".join(clean_lines)
    return result

def process_knowledge_base(folder):
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            path = os.path.join(folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Split metadata from content to avoid cleaning the SOURCE/TITLE lines
            parts = content.split("-" * 30)
            if len(parts) > 1:
                metadata = parts[0]
                body = parts[1]
                cleaned_body = final_clean(body)
                
                with open(path, "w", encoding="utf-8") as f:
                    f.write(metadata + "-" * 30 + "\n\n" + cleaned_body)
                print(f"Cleaned: {filename}")

if __name__ == "__main__":
    process_knowledge_base("data/knowledgebase")