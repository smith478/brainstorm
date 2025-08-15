from typing import List, Dict
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.memory import ConversationBufferMemory

from config import Config

class DiscussionAgent:
    """Handles discussions about papers and AI topics"""
    
    def __init__(self, config: Config, llm):
        self.config = config
        self.llm = llm
        self.embeddings = OllamaEmbeddings(
            model=config.ollama_model,
            base_url=config.ollama_base_url
        )
        self.vector_store = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def index_papers(self, papers: List[Dict]):
        """Create vector store from papers for RAG"""
        texts = []
        metadatas = []
        
        for paper in papers:
            text = f"Title: {paper['title']}\n\nSummary: {paper['summary']}"
            texts.append(text)
            metadatas.append({
                'title': paper['title'],
                'url': paper['url'],
                'authors': ', '.join(paper['authors'][:3])
            })
        
        self.vector_store = Chroma.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=self.embeddings,
            persist_directory=f"{self.config.db_path}/papers_vectors"
        )
    
    def discuss(self, query: str) -> str:
        """Discuss AI topics with context from papers"""
        context = ""
        if self.vector_store:
            relevant_docs = self.vector_store.similarity_search(query, k=3)
            context = "\n\n".join([
                f"Paper: {doc.metadata['title']}\n{doc.page_content}"
                for doc in relevant_docs
            ])
        
        prompt = f"""You are an AI research assistant helping discuss AI concepts and recent developments.

Relevant context from recent papers:
{context}

User question: {query}

Provide an informative response that:
1. Answers the question thoroughly
2. References relevant papers when applicable
3. Explains concepts clearly
4. Suggests related topics to explore

Response:"""

        return self.llm.invoke(prompt)
