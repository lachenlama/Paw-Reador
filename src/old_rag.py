import os
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain import hub
from langchain_community.document_loaders import TextLoader, DirectoryLoader, UnstructuredPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from dotenv import load_dotenv

class RAGModel:
    def __init__(self, docs_path: str, persist_directory: str="../chromadb"):
        load_dotenv()
        assert "TOGETHER_API_KEY" in os.environ, "Please set TOGETHER_API_KEY in .env"

        self.llm = init_chat_model(
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            model_provider="together"
        )

        self.loader = DirectoryLoader(
            docs_path,
            glob="**/*",
            loader_cls=TextLoader,
        )
        self.document = self.loader.load()

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200
        )
        self.chunks = self.splitter.split_documents(self.document)

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.db = Chroma(
            persist_directory = persist_directory,
            embedding_function = self.embeddings
        )

        self.db.add_documents(self.chunks)

    def query(self, question: str, k: int = 5):
        docs = self.db.similarity_search(question, k=k)

        context = " ".join(d.page_content for d in docs)
        prompt = f"""You are a core BeraFarm team member in the Berachain ecosystem. You're bullish AF and speak like a true degen who's been grinding in DeFi. NEVER use formal language or third person - you're one of us building this.

MUST DO:
- Always use "we", "our", "us" when talking about BeraFarm/Berachain
- Be excited and confident about what we're building
- Use casual crypto slang (gm, wagmi, based, etc.)
- Keep responses conversational, not formal
- If you don't know something, say "we're still cooking that" or "more alpha coming soon"
- NO corporate speak or "the protocol does X" - say "we built X" or "our system does X"

NEVER SAY:
- "BeraFarm claims" or "BeraFarm offers" 
- "the provided text" or "according to the documentation"
- Any third-person references to the project

Context: {context}

Question: {question}

Respond as a BeraFarm team member (casual, first-person, degen energy):"""
        
        return self.llm.invoke(prompt)