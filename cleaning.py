import os
import re

# 1. Functions
def final_clean(text):
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[citation\sneeded\]', '', text)
    
    ui_residues = [
        "Scroll to Top", "Bài tiếp theo", "Thẻ:tuyển sinh", 
        "Đăng bởi:", "Ngày đăng:", "Xem thêm"
    ]
    for junk in ui_residues:
        text = text.replace(junk, "")

    text = text.replace("[TABLE]", "\n[Data Table]:\n") 
    text = re.sub(r'\|\s+\|', '|', text) 
    
    lines = text.split('\n')
    clean_lines = [line.strip() for line in lines if len(line.strip()) > 2]
    
    result = "\n\n".join(clean_lines)
    return result

def process_knowledge_base(folder):
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            path = os.path.join(folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            parts = content.split("-" * 30)
            if len(parts) > 1:
                metadata = parts[0]
                body = parts[1]
                cleaned_body = final_clean(body)
                
                with open(path, "w", encoding="utf-8") as f:
                    f.write(metadata + "-" * 30 + "\n\n" + cleaned_body)
                print(f"Cleaned: {filename}")

# 2. Execution
if __name__ == "__main__":
    process_knowledge_base("data/knowledgebase")