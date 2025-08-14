# Brainstorm - AI Knowledge Building Agent
# A modular AI learning assistant using local LLMs via Ollama

import os
import json
import random
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pickle
from pathlib import Path

# External dependencies (add to requirements.txt)
# pip install langchain langchain-community arxiv requests beautifulsoup4 chromadb

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import arxiv
import requests
from bs4 import BeautifulSoup


# ============= Configuration Module =============
@dataclass
class Config:
    """Central configuration for the system"""
    ollama_model: str = "llama3.2"  # Change to any Ollama model
    ollama_base_url: str = "http://localhost:11434"
    arxiv_max_results: int = 20
    weeks_lookback: int = 2
    db_path: str = "./brainstorm_db"
    challenges_path: str = "./challenges"
    papers_cache_path: str = "./papers_cache"
    
    def save(self, path: str = "./config.json"):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path: str = "./config.json"):
        if os.path.exists(path):
            with open(path, 'r') as f:
                return cls(**json.load(f))
        return cls()


# ============= Paper Discovery Module =============
class PaperDiscovery:
    """Handles arXiv paper discovery and recommendations"""
    
    def __init__(self, config: Config):
        self.config = config
        self.cache_path = Path(config.papers_cache_path)
        self.cache_path.mkdir(exist_ok=True)
        
    def search_recent_papers(self, weeks: Optional[int] = None) -> List[Dict]:
        """Search for recent AI papers from arXiv"""
        weeks = weeks or self.config.weeks_lookback
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=weeks)
        
        # Search categories: cs.AI, cs.LG, cs.CL, cs.CV
        search_query = (
            f"(cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV OR cat:cs.NE) "
            f"AND submittedDate:[{start_date.strftime('%Y%m%d')}0000 TO {end_date.strftime('%Y%m%d')}2359]"
        )
        
        search = arxiv.Search(
            query=search_query,
            max_results=self.config.arxiv_max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        for result in search.results():
            papers.append({
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'summary': result.summary,
                'url': result.entry_id,
                'pdf_url': result.pdf_url,
                'published': result.published.isoformat(),
                'categories': result.categories,
                'primary_category': result.primary_category
            })
        
        return papers
    
    def rank_papers(self, papers: List[Dict], interests: List[str] = None) -> List[Dict]:
        """Rank papers by relevance and impact"""
        interests = interests or ["transformers", "llm", "rlhf", "fine-tuning", "multimodal", 
                                  "efficiency", "architecture", "training", "infrastructure"]
        
        for paper in papers:
            score = 0
            title_summary = (paper['title'] + " " + paper['summary']).lower()
            
            # Check for interest keywords
            for interest in interests:
                if interest.lower() in title_summary:
                    score += 2
            
            # Boost for certain categories
            if 'cs.LG' in paper['categories']:
                score += 1
            if 'cs.AI' in paper['categories']:
                score += 1
                
            # Check for important keywords
            important_keywords = ['sota', 'state-of-the-art', 'novel', 'efficient', 
                                 'breakthrough', 'significantly', 'outperforms']
            for keyword in important_keywords:
                if keyword in title_summary:
                    score += 1
            
            paper['relevance_score'] = score
        
        return sorted(papers, key=lambda x: x['relevance_score'], reverse=True)


# ============= Coding Challenge Module =============
class CodingChallenge:
    """Manages coding challenges and feedback"""
    
    def __init__(self, config: Config, llm):
        self.config = config
        self.llm = llm
        self.challenges_path = Path(config.challenges_path)
        self.challenges_path.mkdir(exist_ok=True)
        self.challenges_db = self._load_challenges()
    
    def _load_challenges(self) -> List[Dict]:
        """Load predefined challenges"""
        # These are template challenges - in production, load from file
        return [
            {
                'id': 'attention_impl',
                'title': 'Implement Scaled Dot-Product Attention',
                'difficulty': 'medium',
                'category': 'architecture',
                'description': '''Implement the scaled dot-product attention mechanism from scratch using NumPy.
                
Your function should:
1. Take Q, K, V matrices as input
2. Compute attention scores
3. Apply scaling by sqrt(d_k)
4. Apply softmax
5. Return attention output and attention weights

Bonus: Add support for attention mask''',
                'hints': [
                    "Remember to scale by 1/sqrt(d_k) where d_k is the dimension of keys",
                    "Softmax should be applied along the last axis",
                    "For masked attention, set masked positions to -inf before softmax"
                ],
                'test_cases': '''
import numpy as np

# Test your implementation
Q = np.random.randn(2, 4, 8)  # (batch, seq_len, d_k)
K = np.random.randn(2, 4, 8)
V = np.random.randn(2, 4, 8)

output, weights = scaled_dot_product_attention(Q, K, V)
assert output.shape == (2, 4, 8)
assert weights.shape == (2, 4, 4)
assert np.allclose(weights.sum(axis=-1), 1.0)  # Weights sum to 1
'''
            },
            {
                'id': 'tokenizer_bpe',
                'title': 'Build a Simple BPE Tokenizer',
                'difficulty': 'hard',
                'category': 'nlp',
                'description': '''Implement a basic Byte Pair Encoding (BPE) tokenizer.

Your implementation should:
1. Learn BPE merge rules from a corpus
2. Tokenize new text using learned rules
3. Support encoding and decoding

Start with a simple version that works on characters, then extend to bytes if you want an extra challenge.''',
                'hints': [
                    "Start by counting all bigram frequencies in your corpus",
                    "Merge the most frequent bigram and update your vocabulary",
                    "Keep track of merge rules in order for decoding"
                ]
            },
            {
                'id': 'gradient_accumulation',
                'title': 'Implement Gradient Accumulation',
                'difficulty': 'easy',
                'category': 'training',
                'description': '''Implement gradient accumulation to simulate larger batch sizes.

Create a PyTorch training loop that:
1. Accumulates gradients over N steps
2. Updates weights only after N accumulation steps
3. Properly handles loss scaling

This is essential for training large models with limited GPU memory.''',
                'hints': [
                    "Remember to divide loss by accumulation steps",
                    "Only call optimizer.step() every N iterations",
                    "Don't forget to call optimizer.zero_grad() after updating"
                ]
            },
            {
                'id': 'lora_implementation',
                'title': 'Implement LoRA (Low-Rank Adaptation)',
                'difficulty': 'hard',
                'category': 'fine-tuning',
                'description': '''Implement LoRA for efficient fine-tuning of large models.

Create a LoRA layer that:
1. Adds low-rank decomposition to existing linear layers
2. Keeps original weights frozen
3. Only trains the low-rank matrices A and B
4. Supports different ranks and alpha scaling''',
                'hints': [
                    "LoRA adds BA to the original weight matrix where B and A are low-rank",
                    "Initialize A with random Gaussian and B with zeros",
                    "The effective update is (BA * alpha / rank) * x"
                ]
            },
            {
                'id': 'kv_cache',
                'title': 'Implement KV-Cache for Efficient Inference',
                'difficulty': 'medium',
                'category': 'optimization',
                'description': '''Implement key-value caching for transformer inference.

Your implementation should:
1. Cache key and value projections from previous tokens
2. Only compute new KV for the current token
3. Concatenate with cached values for attention
4. Show the speedup compared to recomputing everything''',
                'hints': [
                    "Store K and V matrices from all previous positions",
                    "Only compute Q, K, V for the new token position",
                    "Append new K, V to cache before attention computation"
                ]
            }
        ]
    
    def get_challenge(self, difficulty: Optional[str] = None, 
                      category: Optional[str] = None) -> Dict:
        """Get a coding challenge based on criteria"""
        challenges = self.challenges_db
        
        if difficulty:
            challenges = [c for c in challenges if c['difficulty'] == difficulty]
        if category:
            challenges = [c for c in challenges if c['category'] == category]
        
        if not challenges:
            challenges = self.challenges_db
            
        return random.choice(challenges)
    
    def get_hint(self, challenge_id: str, hint_number: int = 1) -> str:
        """Get a hint for a specific challenge"""
        challenge = next((c for c in self.challenges_db if c['id'] == challenge_id), None)
        if challenge and 'hints' in challenge:
            hints = challenge['hints']
            if 0 < hint_number <= len(hints):
                return hints[hint_number - 1]
        return "No hint available for this challenge."
    
    def evaluate_solution(self, challenge_id: str, solution_code: str) -> str:
        """Use LLM to evaluate a solution"""
        challenge = next((c for c in self.challenges_db if c['id'] == challenge_id), None)
        if not challenge:
            return "Challenge not found."
        
        prompt = f"""Evaluate this solution for the coding challenge: {challenge['title']}

Challenge Description:
{challenge['description']}

Submitted Solution:
```python
{solution_code}
```

Provide feedback on:
1. Correctness of the implementation
2. Code quality and efficiency
3. Edge cases handling
4. Suggestions for improvement
5. What the solution does well

Be constructive and educational in your feedback."""

        return self.llm.invoke(prompt)


# ============= Discussion Module =============
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


# ============= Main Agent Orchestrator =============
class BrainstormAgent:
    """Main agent that orchestrates all modules"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
        # Initialize LLM
        self.llm = Ollama(
            model=self.config.ollama_model,
            base_url=self.config.ollama_base_url
        )
        
        # Initialize modules
        self.paper_discovery = PaperDiscovery(self.config)
        self.coding_challenges = CodingChallenge(self.config, self.llm)
        self.discussion = DiscussionAgent(self.config, self.llm)
        
        # Setup agent
        self._setup_agent()
    
    def _setup_agent(self):
        """Setup LangChain agent with tools"""
        tools = [
            Tool(
                name="search_papers",
                func=lambda x: self._search_papers_tool(x),
                description="Search for recent AI papers. Input: number of weeks to look back"
            ),
            Tool(
                name="get_challenge",
                func=lambda x: self._get_challenge_tool(x),
                description="Get a coding challenge. Input: 'easy', 'medium', or 'hard'"
            ),
            Tool(
                name="get_hint",
                func=lambda x: self._get_hint_tool(x),
                description="Get hint for a challenge. Input: 'challenge_id,hint_number'"
            ),
            Tool(
                name="evaluate_code",
                func=lambda x: self._evaluate_code_tool(x),
                description="Evaluate solution code. Input: 'challenge_id|solution_code'"
            ),
            Tool(
                name="discuss",
                func=lambda x: self.discussion.discuss(x),
                description="Discuss AI topics and papers. Input: your question"
            )
        ]
        
        prompt = PromptTemplate(
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
            template="""You are Brainstorm, an AI learning assistant that helps users stay current with AI research and maintain their technical skills.

You have access to these tools:
{tools}

Tool Names: {tool_names}

To use a tool, respond with:
Thought: [your reasoning]
Action: [tool_name]
Action Input: [input for the tool]
Observation: [tool's response will appear here]

Then continue with:
Thought: [analyze the observation]
Final Answer: [your complete response to the user]

Current conversation:
Human: {input}

{agent_scratchpad}"""
        )
        
        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=5
        )
    
    def _search_papers_tool(self, weeks: str) -> str:
        """Tool wrapper for paper search"""
        try:
            weeks = int(weeks) if weeks.isdigit() else self.config.weeks_lookback
            papers = self.paper_discovery.search_recent_papers(weeks)
            papers = self.paper_discovery.rank_papers(papers)[:5]
            
            # Index papers for discussion
            self.discussion.index_papers(papers)
            
            result = "Top recent papers:\n\n"
            for i, paper in enumerate(papers, 1):
                result += f"{i}. {paper['title']}\n"
                result += f"   Authors: {', '.join(paper['authors'][:3])}\n"
                result += f"   Score: {paper['relevance_score']}\n"
                result += f"   URL: {paper['url']}\n\n"
            
            return result
        except Exception as e:
            return f"Error searching papers: {str(e)}"
    
    def _get_challenge_tool(self, difficulty: str) -> str:
        """Tool wrapper for getting challenges"""
        difficulty = difficulty.lower() if difficulty in ['easy', 'medium', 'hard'] else None
        challenge = self.coding_challenges.get_challenge(difficulty=difficulty)
        
        return f"""Challenge: {challenge['title']}
ID: {challenge['id']}
Difficulty: {challenge['difficulty']}
Category: {challenge['category']}

{challenge['description']}

To get hints, use: get_hint with '{challenge['id']},1' (for hint 1)
To submit solution, use: evaluate_code with '{challenge['id']}|your_code'"""
    
    def _get_hint_tool(self, input_str: str) -> str:
        """Tool wrapper for getting hints"""
        parts = input_str.split(',')
        if len(parts) != 2:
            return "Please provide input as: 'challenge_id,hint_number'"
        
        challenge_id = parts[0].strip()
        try:
            hint_num = int(parts[1].strip())
        except:
            return "Hint number must be an integer"
        
        return self.coding_challenges.get_hint(challenge_id, hint_num)
    
    def _evaluate_code_tool(self, input_str: str) -> str:
        """Tool wrapper for code evaluation"""
        parts = input_str.split('|', 1)
        if len(parts) != 2:
            return "Please provide input as: 'challenge_id|solution_code'"
        
        challenge_id = parts[0].strip()
        solution_code = parts[1].strip()
        
        return self.coding_challenges.evaluate_solution(challenge_id, solution_code)
    
    def run(self, query: str) -> str:
        """Main entry point for the agent"""
        return self.agent_executor.run(query)


# ============= CLI Interface =============
def main():
    """Command-line interface for Brainstorm"""
    print("""
    ╔══════════════════════════════════════╗
    ║       BRAINSTORM AI ASSISTANT        ║
    ║   Your AI Learning & Research Tool   ║
    ╚══════════════════════════════════════╝
    
    Commands:
    - 'papers [weeks]' - Get recent papers
    - 'challenge [difficulty]' - Get coding challenge  
    - 'discuss [topic]' - Discuss AI topics
    - 'config' - Show configuration
    - 'quit' - Exit
    """)
    
    # Load or create config
    config = Config.load()
    agent = BrainstormAgent(config)
    
    while True:
        try:
            user_input = input("\n[Brainstorm] > ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye! Keep learning! 🧙‍♂️")
                break
            elif user_input.lower() == 'config':
                print(f"Current config: {json.dumps(config.__dict__, indent=2)}")
            else:
                response = agent.run(user_input)
                print(f"\n{response}")
                
        except KeyboardInterrupt:
            print("\nGoodbye! Keep learning! 🧙‍♂️")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()