import os
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_chunks(file_path: str):
    """Extract and chunk text from a .txt, .md, or .pdf file."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".txt", ".md"]:
        loader = TextLoader(file_path)
    elif ext == ".pdf":
        loader = UnstructuredPDFLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)


def load_docs_from_dir(directory: str):
    """Walk a directory, extract and chunk all supported docs."""
    chunks = []
    for root, _, files in os.walk(directory):
        for fname in files:
            path = os.path.join(root, fname)
            if os.path.splitext(fname)[1].lower() in ['.txt', '.md', '.pdf']:
                chunks.extend(extract_chunks(path))
    return chunks