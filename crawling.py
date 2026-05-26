import os
import re
import requests
import urllib3
import pdfplumber
from bs4 import BeautifulSoup

# --- 1. CONFIGURATION ---
# Destination folder for cleaned text files 
DEST_FOLDER = "data/knowledgebase"

# Added Pittsburgh/CMU English URLs to test bilingual capability
PDF_FILES = [
    r"pdf\Tuyen-sinh-hus.pdf",
    r"pdf\Tuyen-sinh-vnu.pdf",
]

# List of URLs to crawl
URLS = [
    "https://vnu.edu.vn/gioi-thieu",
    "https://vnu.edu.vn/gioi-thieu/tong-quan/su-mang-tam-nhin",
    "https://vnu.edu.vn/dao-tao/ke-hoach-hoc-tap-va-giang-day",
    "https://vi.wikipedia.org/wiki/%C4%90%E1%BA%A1i_h%E1%BB%8Dc_Qu%E1%BB%91c_gia_H%C3%A0_N%E1%BB%99i",
    "https://uet.vnu.edu.vn/truong-dai-hoc-cong-nghe-dhqghn-ma-truong-qhi-tuyen-sinh-nam-2026-bac-dai-hoc-2/",
    "https://fied.ulis.vnu.edu.vn/tuyen-sinh-cu-nhan/",
    "https://tuyensinh.ussh.edu.vn/mot-so-thong-tin-thi-sinh-can-biet-ve-tuyen-sinh-dai-hoc-he-chinh-quy-nam-2026.html",
    "https://vnu.edu.vn/dao-tao/gioi-thieu-chung",
    "https://vnu.edu.vn/cac-chuong-trinh-lien-ket-dao-tao-quoc-te-o-dai-hoc-quoc-gia-ha-noi-post15064.html",
    "https://en.wikipedia.org/wiki/Pittsburgh",
    "https://en.wikipedia.org/wiki/International_Conference_on_Machine_Learning",
    "https://en.wikipedia.org/wiki/Carnegie_Mellon_University",
    "https://www.cmu.edu/about/history",
    "https://www.cmu.edu/about/vision-mission-values",
]

# Phrases to filter out (Bilingual)
NOISE_PHRASES = [
    r"có thể bạn quan tâm", r"xem thêm", r"tin liên quan", r"bài viết liên quan",
    r"you might also like", r"read more", r"related articles", r"latest news",
    r"trang chủ", r"home page", r"đăng nhập", r"login", r"facebook", r"youtube", r"tiktok"
]

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 2. CLEANING FUNCTIONS ---

def clean_text(text):
    """Deep cleans text for RAG processing."""
    if not text: return ""
    
    # Remove noise phrases line by line
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        is_junk = any(re.search(pattern, line, re.IGNORECASE) for pattern in NOISE_PHRASES)
        if not is_junk and len(line.strip()) > 5:
            clean_lines.append(line.strip())
    
    text = "\n".join(clean_lines)

    # Basic cleanup of multiple spaces/newlines
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

def table_to_markdown(table):
    """Converts pdfplumber table list to Markdown format."""
    if not table: return ""
    md_table = ""
    for i, row in enumerate(table):
        clean_row = [str(cell).replace('\n', ' ') if cell else "" for cell in row]
        if all(cell == "" for cell in clean_row): continue
        md_table += "| " + " | ".join(clean_row) + " |\n"
        if i == 0:
            md_table += "| " + " | ".join(["---"] * len(row)) + " |\n"
    return md_table

# --- 3. WORKER FUNCTIONS ---

def crawl_web(url):
    """Deeply robust crawler with Waterfall targeting for VNU, CMU, and Wikipedia."""
    print(f"--- Crawling: {url}")
    try:
        # 1. Setup Request
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=20, verify=False) 
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 2. Pre-Clean: Remove absolutely useless tags
        for junk in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'button', 'noscript', 'iframe']):
            junk.decompose()

        # 3. Waterfall Targeting: Find the main content area
        # We try Wikipedia specific, then general article, then specific IDs, then fallback to body
        main_content = soup.find('div', {'class': 'mw-parser-output'}) or \
                       soup.find('div', {'id': 'mw-content-text'}) or \
                       soup.find('article') or \
                       soup.find('main') or \
                       soup.find('div', {'id': 'content'}) or \
                       soup.find('div', {'class': 'content'}) or \
                       soup.body

        if not main_content:
            print(f"   [!] Error: No content container found for {url}")
            return

        # 4. Content Extraction
        stop_sections = [
            "references", "external links", "see also", "further reading", "notes", "bibliography",
            "tài liệu tham khảo", "liên kết ngoài", "xem thêm", "ghi chú", "thư mục", "bài viết liên quan"
        ]

        title = soup.title.string if soup.title else "No Title"
        # Search for text-heavy tags
        elements = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li', 'table', 'span'])
        
        content_pieces = []
        for el in elements:
            # STOP logic: triggers only on H1 or H2 that match stop words exactly
            if el.name in ['h1', 'h2']:
                header_text = el.get_text().strip().lower().replace('[edit]', '').strip()
                if header_text in stop_sections:
                    print(f"   [Stop] Reached end section: {header_text}")
                    break 
            
            if el.name == 'table':
                table_text = el.get_text(separator=" | ").strip()
                if len(table_text) > 30: 
                    content_pieces.append(f"\n[TABLE]\n{table_text}\n")
            else:
                # Get text but filter out very short UI fragments
                text = el.get_text().strip()
                if len(text) > 10: 
                    # Prevent duplicate lines (common with span inside p)
                    if not content_pieces or text not in content_pieces[-1]:
                        content_pieces.append(text)

        # 5. Build and Clean
        full_text = "\n\n".join(content_pieces)
        cleaned_content = clean_text(full_text)
        
        # Final sanity check: if too small, something is wrong
        if len(cleaned_content) < 100:
            # Emergency fallback: Just grab all paragraphs in the whole body
            print(f"   [!] Targeted extraction failed. Using Emergency Fallback for {url}")
            paras = soup.find_all('p')
            cleaned_content = clean_text("\n\n".join([p.get_text() for p in paras]))

        if not cleaned_content:
            print(f"   [X] Failed: Content is still empty for {url}")
            return

        # 6. Save to File
        # Shorten filenames to avoid OS errors
        domain = url.split('//')[-1].split('/')[0].replace('.', '')
        page_path = url.split('/')[-1] or "home"
        safe_name = re.sub(r'[^\w\s-]', '', f"{domain}_{page_path}")[:80] + ".txt"
        filepath = os.path.join(DEST_FOLDER, safe_name)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"SOURCE: {url}\n")
            f.write(f"TITLE: {title}\n")
            f.write("-" * 30 + "\n\n")
            f.write(cleaned_content)
            
        print(f"   [OK] Saved: {safe_name} ({len(cleaned_content)} chars)")
        
    except Exception as e:
        print(f"   [!] Error crawling {url}: {e}")

def process_digital_pdf(pdf_path):
    """Extracts text and tables from Digital PDFs using pdfplumber."""
    if not os.path.exists(pdf_path):
        print(f"   [!] File not found: {pdf_path}")
        return

    base_name = os.path.basename(pdf_path).replace(".pdf", ".txt")
    output_path = os.path.join(DEST_FOLDER, base_name)
    print(f"--- Processing Digital PDF: {pdf_path}")
    
    full_content = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Extract text
                page_text = page.extract_text()
                if page_text:
                    full_content.append(f"--- PAGE {i+1} ---\n{page_text}")
                
                # Extract tables
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        full_content.append(f"\n[TABLE PAGE {i+1}]\n{table_to_markdown(table)}\n")

        with open(output_path, "w", encoding="utf-8") as f:
            final_text = clean_text("\n\n".join(full_content))
            f.write(f"SOURCE: {pdf_path}\n\n" + final_text)
        print(f"   [DONE] Saved: {base_name}")

    except Exception as e:
        print(f"   [!] Error processing {pdf_path}: {e}")

# --- 4. EXECUTION ---

if __name__ == "__main__":
    if not os.path.exists(DEST_FOLDER):
        os.makedirs(DEST_FOLDER)

    print("\n>>> STEP 1: WEB CRAWLING (Bilingual)")
    for url in URLS:
        crawl_web(url)
        
    print("\n>>> STEP 2: DIGITAL PDF PROCESSING")
    for pdf in PDF_FILES:
        process_digital_pdf(pdf)
    
    print(f"\nCOMPLETION! All text data is cleaned and ready in: {DEST_FOLDER}")