import os
import logging
import asyncio
from typing import List, Dict, Optional
from datetime import datetime, timedelta
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
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
import json

from utils import load_docs_from_dir

RAG_PROMPT_TEMPLATE = """You are a core BeraFarm team member in the Berachain ecosystem. You're bullish AF and speak like a true degen who's been grinding in DeFi. NEVER use formal language or third person - you're one of us building this.

MUST DO:
- Always use "we", "our", "us" when talking about BeraFarm/Berachain
- Be excited and confident about what we're building
- Use casual crypto slang (gm, wagmi, based, etc.)
- Keep responses conversational, not formal
- If you don't know something, say "we're still cooking that" or "more alpha coming soon"
- NO corporate speak or "the protocol does X" - say "we built X" or "our system does X"
- Keep Discord responses under 2000 charaters when possible
- Use emojis sparingly but effectively

NEVER SAY:
- "BeraFarm claims" or "BeraFarm offers" 
- "the provided text" or "according to the documentation"
- Any third-person references to the project
- Anything about f-BGT. If asked about it, just say we're not ready to share the full alpha on that yet.

Previous context (if any): {chat_history}

Context: {context}

Question: {question}

Respond as a BeraFarm team member (casual, first-person, degen energy):"""

class RAGModel:
    def __init__(self, docs_path: str, persist_directory: str="../chromadb",
                 temperature: float=0.7, max_tokens: int=1024, cache_size: int=100, context_window: int=5):
        load_dotenv()
        assert "TOGETHER_API_KEY" in os.environ, "Please set TOGETHER_API_KEY in .env"

        self.llm = ChatTogether(
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            temperature=temperature,
            max_tokens=max_tokens
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5"
        )

        #Enhanced Memory and caching
        self.chat_histores: Dict[str, List[Dict]] = {} #user_id --> chat history
        self.response_cache: Dict[str, str] = {} #question hash --> response
        self.cache_size: cache_size
        self.context_window = context_window

        #Rate limiting
        self.user_requests: Dict[str, List[datetime]] = {} #user_id --> list of request timestamps
        self.rate_limit_window = timedelta(minutes=1)
        self.max_requests_per_window = 10

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

        base_retriever = self.db.as_retriever(search_kwargs={"k": 5})

        reranker = FlashrankRerank()

        self.retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever
        )

        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        self.rag_chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough(), "chat_history": lambda x: ""} #will be populated dynamically
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _load_and_process_docs(self, docs_path: str) -> List[Document]:
        chunks = load_docs_from_dir(docs_path)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". "," ",""]
        )

        final_chunks = []
        for chunk in chunks:
            if len(chunk.page_content) > 1000:
                sub_chunk = text_splitter.split_documents([chunk])
                final_chunks.extend(sub_chunk)
            else:
                final_chunks.append(chunk)

        return final_chunks

    @staticmethod
    def _format_docs(docs: list[Document]) -> str:
        """Joins document contents into a single string for context."""
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def _get_question_hash(self, question: str) -> str:
        return hashlib.sha256(question.lower().strip().encode()).hexdigest()

    def _check_rate_limit(self, user_id: str) -> bool:
        now = datetime.now()
        if user_id not in self.user_requests:
            self.user_requests[user_id] = []

        self.user_requests[user_id] = [
            req_time for req_time in self.user_requests[user_id]
            if now - req_time < self.rate_limit_window
        ]

        return len(self.user_requests[user_id]) < self.max_requests_per_window
    
    def _add_request(self, user_id: str):
        if user_id not in self.user_requests:
            self.user_requests[user_id] = []
        self.user_requests[user_id].append(datetime.now())

    def _get_chat_history(self, user_id: str) -> str:
        if user_id not in self.chat_histores:
            return ""
        
        history = self.chat_histores[user_id][-self.context_window:]
        formatted = []
        for entry in history:
            formatted.append(f"User: {entry['question']}")
            formatted.append(f"You: {entry['response'][:200]}...")

        return "\n".join(formatted[-4:])

    def _update_chat_history(self, user_id: str, question: str, response: str):
        if user_id not in self.chat_histories:
            self.chat_histores[user_id] = []

        self.chat_histories[user_id].append({
            "question": question,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })

        if len(self.chat_histores[user_id]) > self.context_window * 2:
            self.chat_histores[user_id] = self.chat_histores[user_id][-self.context_window:]

    def _clean_response_for_discord(self, response: str) -> str:
        response = "\n".join(line.strip() for line in response.splitlines("\n") if line.strip())

        if len(response) > 1900:
            response = response[:1900] + "\n\n...ask for more details! 😉"

        return response

    async def query_async(self, question: str, user_id: str = None, k: int = 5) -> Dict:
        """Async query with enhanced features"""
        # Rate limiting check
        if user_id and not self._check_rate_limit(user_id):
            return {
                "response": "Whoa there anon! You're asking too fast 🚀 Slow down a bit and try again in a minute.",
                "cached": False,
                "rate_limited": True
            }

        # Check cache first
        question_hash = self._get_question_hash(question)
        if question_hash in self.response_cache:
            return {
                "response": self.response_cache[question_hash],
                "cached": True,
                "rate_limited": False
            }

        try:
            # Update rate limit
            if user_id:
                self._add_request(user_id)

            # Get chat history context
            chat_history = self._get_chat_history(user_id) if user_id else ""
            
            # Update retriever
            # self.retriever.search_kwargs = {'k': k} if hasattr(self.retriever, 'search_kwargs') else {'search_type': 'similarity', 'k': k}
            
            # Get context docs
            docs = await asyncio.get_event_loop().run_in_executor(
                None, self.retriever.get_relevant_documents, question
            )
            context = self._format_docs(docs)
            
            # Generate response
            prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
            formatted_prompt = prompt.format(
                context=context,
                question=question,
                chat_history=chat_history
            )
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.llm.invoke, formatted_prompt
            )
            
            clean_response = self._clean_response_for_discord(response.content)
            
            # Cache response
            if len(self.response_cache) >= self.cache_size:
                # Remove oldest cache entry (simple FIFO)
                oldest_key = next(iter(self.response_cache))
                del self.response_cache[oldest_key]
            
            self.response_cache[question_hash] = clean_response
            
            # Update chat history
            if user_id:
                self._update_chat_history(user_id, question, clean_response)
            
            return {
                "response": clean_response,
                "cached": False,
                "rate_limited": False
            }

        except Exception as e:
            logging.error(f"Error in query: {e}")
            return {
                "response": "Oof, something went wrong on our end! We're fixing it rn 🔧",
                "cached": False,
                "rate_limited": False,
                "error": str(e)
            }

    def query(self, question: str, user_id: str = None, k: int = 5, stream: bool = False):
        """Synchronous query with enhancements"""
        if stream:
            # For streaming, use original implementation
            self.retriever.search_kwargs = {'k': k}
            return self.rag_chain.stream(question)
        
        # Run async query in sync context
        return asyncio.run(self.query_async(question, user_id, k))

    def add_documents(self, docs: List[Document]):
        """Add new documents to the vector store"""
        self.db.add_documents(docs)
        logging.info(f"Added {len(docs)} new documents to vector store")

    def get_stats(self) -> Dict:
        """Get model statistics"""
        return {
            "total_users": len(self.chat_histories),
            "cache_size": len(self.response_cache),
            "total_documents": self.db._collection.count(),
            "active_users": len([uid for uid, reqs in self.user_requests.items() 
                                if any(datetime.now() - req < self.rate_limit_window for req in reqs)])
        }

    def clear_user_history(self, user_id: str):
        """Clear chat history for a user"""
        if user_id in self.chat_histories:
            del self.chat_histories[user_id]
        if user_id in self.user_requests:
            del self.user_requests[user_id]

    def update_system_prompt(self, new_template: str):
        """Update the system prompt template"""
        prompt = ChatPromptTemplate.from_template(new_template)
        self.rag_chain = (
            {"context": self.retriever | self._format_docs, 
             "question": RunnablePassthrough(),
             "chat_history": lambda x: ""}
            | prompt
            | self.llm
            | StrOutputParser()
        )
