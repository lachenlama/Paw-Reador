import os
import logging
from dotenv import load_dotenv
from langchain_together import ChatTogether
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from utils import load_docs_from_dir

RAG_PROMPT_TEMPLATE = """You are a core BeraFarm team member in the Berachain ecosystem. You're bullish AF and speak like a true degen who's been grinding in DeFi. NEVER use formal language or third person - you're one of us building this.

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
- NEVER mention 'f-bgt' or 'fbgt' at all – just ignore it completely like it doesn't exist

Context: {context}

Question: {question}

Respond as a BeraFarm team member (casual, first-person, degen energy):"""

class RAGModel:
    def __init__(self, docs_path: str, persist_directory: str="../chromadb"):
        load_dotenv()
        assert "TOGETHER_API_KEY" in os.environ, "Please set TOGETHER_API_KEY in .env"

        self.llm = ChatTogether(
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            logging.info(f"Loading existing vector store from {persist_directory}")
            self.db = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)
        else:
            logging.warning(f"No existing vector store found in {persist_directory}. Building a new one...")
            chunks = load_docs_from_dir(docs_path)
            self.db = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=persist_directory
            )
            logging.info(f"Vector store created and persisted at {persist_directory}")

        base_retriever = self.db.as_retriever()

        reranker = FlashrankRerank()

        self.retriever = ContextualCompressionRetriever(
            document_compressor=reranker,
            base_retriever=base_retriever
        )

        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        self.rag_chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    @staticmethod
    def _format_docs(docs: list[Document]) -> str:
        """Joins document contents into a single string for context."""
        return "\n\n".join(doc.page_content for doc in docs)

    def query(self, question: str, k: int = 5, fetch_k: int = 20):
        self.retriever.base_retriever.search_kwargs = {'k': fetch_k}
        self.retriever.base_compressor.top_n = k
        return self.rag_chain.invoke(question)