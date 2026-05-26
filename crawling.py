import os
import re
import requests
import urllib3
import pytesseract
import pdfplumber
import io
from bs4 import BeautifulSoup
from pdf2image import convert_from_path
from PIL import Image

# --- 1. CONFIGURATION ---

# Processing paths to external tools (Tesseract OCR and Poppler)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
POPPLER_PATH = r'C:\poppler\Library\bin' 
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

# Destination folder for cleaned text files 
DEST_FOLDER = "data/knowledgebase"

# List of PDF files to process
PDF_FILES = [
    r"pdf\Tuyen-sinh-hus.pdf",
    r"pdf\Tuyen-sinh-vnu.pdf",
    r"pdf\Quy-che-dao-tao-dai-hoc.pdf",
    r"pdf\Quy-che-dao-tao-thac-si.pdf",
    r"pdf\Quy-che-dao-tao-tien-si.pdf",
]

# List of URLs to crawl
URLS = [
    "https://vnu.edu.vn/gioi-thieu",
    "https://vnu.edu.vn/gioi-thieu/tong-quan/su-mang-tam-nhin",
    "https://vi.wikipedia.org/wiki/%C4%90%E1%BA%A1i_h%E1%BB%8Dc_Qu%E1%BB%91c_gia_H%C3%A0_N%E1%BB%99i",
    "https://uet.vnu.edu.vn/truong-dai-hoc-cong-nghe-dhqghn-ma-truong-qhi-tuyen-sinh-nam-2026-bac-dai-hoc-2/",
    "https://fied.ulis.vnu.edu.vn/tuyen-sinh-cu-nhan/",
    "https://tuyensinh.ussh.edu.vn/mot-so-thong-tin-thi-sinh-can-biet-ve-tuyen-sinh-dai-hoc-he-chinh-quy-nam-2026.html",
    "https://vnu.edu.vn/dao-tao/gioi-thieu-chung",
    "https://vnu.edu.vn/cac-chuong-trinh-lien-ket-dao-tao-quoc-te-o-dai-hoc-quoc-gia-ha-noi-post15064.html",
]

# Turn off warnings about insecure requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 2. UTILITY FUNCTIONS ---

def clean_text(text):
    """Làm sạch văn bản: xóa dòng trống thừa và khoảng trắng dư."""
    if not text: return ""
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def table_to_markdown(table):
    """Chuyển đổi dữ liệu bảng từ pdfplumber sang định dạng Markdown."""
    if not table: return ""
    md_table = ""
    for i, row in enumerate(table):
        clean_row = [str(cell).replace('\n', ' ') if cell else "" for cell in row]
        md_table += "| " + " | ".join(clean_row) + " |\n"
        if i == 0:
            md_table += "| " + " | ".join(["---"] * len(row)) + " |\n"
    return md_table

# --- 3. WEBSITE CRAWLING FUNCTION ---

def crawl_web(url):
    """Tải nội dung từ trang web và lưu thành file .txt."""
    print(f"--- Crawling: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15, verify=False) 
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for junk in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            junk.decompose()

        title = soup.title.string if soup.title else "No Title"
        elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'li', 'table'])
        content = "\n".join([el.get_text() for el in elements])
        
        safe_name = re.sub(r'[^\w\s-]', '', url.split('//')[-1].replace('/', '_')) + ".txt"
        filepath = os.path.join(DEST_FOLDER, safe_name)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"SOURCE: {url}\nTITLE: {title}\n\n{clean_text(content)}")
        print(f"   [OK] Saved to: {safe_name}")
    except Exception as e:
        print(f"   [!] Error crawling {url}: {e}")

# --- 4. SMART PDF PROCESSING FUNCTION ---

def process_smart_pdf(pdf_path):
    """Xử lý PDF: Trích xuất text máy, bảng biểu và dùng OCR nếu là trang quét."""
    if not os.path.exists(pdf_path):
        print(f"   [!] File not found: {pdf_path}")
        return

    base_name = os.path.basename(pdf_path).replace(".pdf", ".txt")
    output_path = os.path.join(DEST_FOLDER, base_name)
    print(f"--- Processing PDF: {pdf_path}")
    
    full_content = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                
                # A.TEXT PDF
                if len(page_text.strip()) > 50:
                    print(f"   Page {i+1}: Digital text found")
                    full_content.append(f"--- TRANG {i+1} (Text) ---\n{page_text}")
                    
                    tables = page.extract_tables()
                    if tables:
                        print(f"      - Found {len(tables)} table(s)")
                        for table in tables:
                            full_content.append(f"\n[BẢNG TRANG {i+1}]\n{table_to_markdown(table)}\n")
                
                # B. SCANNED PDF
                else:
                    print(f"   Page {i+1}: Scanned image detected, using OCR...")
                    # Chuyển trang PDF thành ảnh (300 DPI)
                    images = convert_from_path(
                        pdf_path, 300, 
                        first_page=i+1, last_page=i+1, 
                        poppler_path=POPPLER_PATH
                    )
                    ocr_text = pytesseract.image_to_string(images[0], lang='vie', config='--psm 6')
                    full_content.append(f"--- TRANG {i+1} (OCR) ---\n{ocr_text}")

        # Save the combined content to a text file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"SOURCE: {pdf_path}\n\n" + "\n\n".join(full_content))
        print(f"   [DONE] Full processing finished for: {base_name}")

    except Exception as e:
        print(f"   [!] Critical error processing {pdf_path}: {e}")

# --- 5. MAIN PROGRAM ---

if __name__ == "__main__":
    if not os.path.exists(DEST_FOLDER):
        os.makedirs(DEST_FOLDER)

    print("\nStage 1: CRAWLING WEB PAGES")
    for url in URLS:
        crawl_web(url)
        
    print("\nStage 2: PROCESSING PDF FILES (DIGITAL & SCAN)")
    for pdf in PDF_FILES:
        process_smart_pdf(pdf)
    
    print(f"\nCOMPLETION! All cleaned data is ready at: {DEST_FOLDER}")