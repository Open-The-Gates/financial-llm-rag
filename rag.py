import os
import pandas as pd
import pdfplumber
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()

PDF_DIR = "./data"

def extract_text_and_tables(pdf_path):
    docs = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                # Try using first non-empty line as a section heading
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                if lines:
                    section_title = lines[0]
                else:
                    section_title = "Untitled Section"

                doc = Document(
                    page_content=f"SECTION TITLE: {section_title}\n{text}",
                    metadata={
                        "page": page_num + 1,
                        "type": "text",
                        "section_title": section_title
                    }
                )
                docs.append(doc)

            # Tables
            tables = page.extract_tables()
            for table in tables:
                # Convert table to readable format
                table_str = "\n".join(
                    [" | ".join(cell if cell is not None else "" for cell in row) for row in table if any((cell or "").strip() for cell in row)]
                )
                table_doc = Document(
                    page_content=f"TABLE (Page {page_num+1}):\n{table_str}",
                    metadata={
                        "page": page_num + 1,
                        "type": "table"
                    }
                )
                docs.append(table_doc)
    return docs
    
def extract_excel_chunks(excel_path):
    all_docs = []
    xls = pd.ExcelFile(excel_path, engine="xlrd")
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        if df.empty:
            continue
        table_str = df.fillna("").astype(str).to_markdown(index=False)
        doc = Document(
            page_content=f"EXCEL SHEET: {sheet_name}\n{table_str}",
            metadata={"source": os.path.basename(excel_path), "sheet": sheet_name}
        )
        all_docs.append(doc)
    return all_docs

def load_all_docs(directory):
    all_docs = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.lower().endswith(".pdf"):
            print(f"📄 Parsing PDF: {filename}")
            all_docs.extend(extract_text_and_tables(filepath))
        elif filename.lower().endswith((".xls", ".xlsx")):
            print(f"📊 Parsing Excel: {filename}")
            all_docs.extend(extract_excel_chunks(filepath))
    return all_docs

def smart_chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(docs)
    

def main():
    print("📄 Loading PDFs with tables...")
    documents = load_all_docs(PDF_DIR)
    print(f"Loaded {len(documents)} raw text/table segments")

    print("🔪 Splitting into smart chunks...")
    chunks = smart_chunk_docs(documents)
    print(f"✅ Created {len(chunks)} chunks")

    # Show samples
    for i, chunk in enumerate(chunks[:20]):
        print(f"\n--- Chunk {i+1} ---\n{chunk.page_content}\n --- Metadata ---\n{chunk.metadata}")
    # Create embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load chunks
    chunks = smart_chunk_docs(documents)

    # Save into FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local("faiss_index")

if __name__ == "__main__":
    main()
