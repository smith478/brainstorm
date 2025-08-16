from typing import Optional, List, Dict
from collections import Counter
import re

from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import Tool

from config import Config
from llm_provider import get_llm
from tools.paper_discovery import PaperDiscovery
from tools.coding_challenge import CodingChallenge
from tools.discussion import DiscussionAgent

class BrainstormAgent:
    """Main agent that orchestrates all modules"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.load()
        
        # Initialize LLM
        self.llm = get_llm(self.config)
        
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
                name="discover_trends",
                func=lambda x: self._discover_trends_tool(x.split(',') if x else None),
                description="Discover trending AI techniques and tools. Input: comma-separated list of topics (e.g., 'AI tools,MLOps platforms') or leave empty for defaults."
            ),
            Tool(
                name="get_challenge",
                func=lambda x: self._get_challenge_tool(x),
                description="Get a coding challenge. Input: 'easy', 'medium', or 'hard'"
            ),
            Tool(
                name="generate_challenge",
                func=lambda x: self.coding_challenges.generate_challenge(x),
                description="Generate a new coding challenge on a topic. Input: topic string"
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
        
        # Get the ReAct prompt template
        prompt = hub.pull("hwchase17/react")
        
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
            papers = self.paper_discovery.rank_papers(papers)

            # Add social score
            for paper in papers:
                social_score = self._get_social_score(paper['title'])
                paper['relevance_score'] += social_score
                paper['social_score'] = social_score

            papers = sorted(papers, key=lambda x: x['relevance_score'], reverse=True)[:5]
            
            # Index papers for discussion
            self.discussion.index_papers(papers)
            
            result = "Top recent papers:\n\n"
            for i, paper in enumerate(papers, 1):
                result += f"{i}. {paper['title']}\n"
                result += f"   Authors: {', '.join(paper['authors'][:3])}\n"
                result += f"   Score: {paper['relevance_score']}\n"
                result += f"   Social Score: {paper.get('social_score', 0)}\n"
                result += f"   URL: {paper['url']}\n\n"
            
            return result
        except Exception as e:
            return f"Error searching papers: {str(e)}"

    def _get_social_score(self, paper_title: str) -> int:
        """Get a social score for a paper based on web search results."""
        score = 0
        platforms = {
            "twitter.com": 2,
            "reddit.com/r/MachineLearning": 3,
            "news.ycombinator.com": 4
        }
        
        for platform, weight in platforms.items():
            query = f'"{paper_title}" site:{platform}'
            try:
                # This is a placeholder for the actual tool call
                # In a real scenario, this would be:
                # from default_api import google_web_search
                # results = google_web_search(query=query)
                # For now, we'll simulate a result
                pass
            except Exception as e:
                print(f"Error searching {platform}: {e}")

        return score

    def _discover_trends_tool(self, topics: List[str] = None) -> str:
        """Discover trending AI techniques and tools on a given topic."""
        if topics is None:
            topics = ["AI tools", "machine learning techniques", "MLOps platforms"]
        
        all_trends = {}

        for topic in topics:
            query = f"trending {topic} 2025"
            try:
                # This is a placeholder for the actual tool call
                # from default_api import google_web_search
                # results = google_web_search(query=query)
                # For now, we'll simulate a result
                results = None
                if results and results.get("search_results"):
                    all_trends[topic] = self._extract_trends_from_results(results["search_results"])
                else:
                    all_trends[topic] = ["No trends found"]
            except Exception as e:
                all_trends[topic] = [f"Error searching for {topic}: {e}"]

        return json.dumps(all_trends, indent=2)

    def _extract_trends_from_results(self, search_results: List[Dict]) -> List[str]:
        """Extracts and ranks trends from search results."""
        text_corpus = " ".join([res.get("title", "") + " " + res.get("snippet", "") for res in search_results])
        
        # A simple regex to find capitalized words or phrases that might be tools/techniques
        potential_trends = re.findall(r'\b[A-Z][a-zA-Z0-9-]+\b(?:\s+[A-Z][a-zA-Z0-9-]+)*', text_corpus)
        
        # Filter out common words
        common_words = ["AI", "Machine", "Learning", "Data", "The", "A", "An", "Is", "Are", "To", "From", "In", "On", "Of", "For", "With"]
        potential_trends = [trend for trend in potential_trends if trend not in common_words]
        
        # Count and rank
        trend_counts = Counter(potential_trends)
        ranked_trends = [trend for trend, count in trend_counts.most_common(10)]
        
        return ranked_trends

    def _get_challenge_tool(self, difficulty: str) -> str:
        """Tool wrapper for getting challenges"""
        difficulty = difficulty.lower() if difficulty in ['easy', 'medium', 'hard'] else None
        challenge = self.coding_challenges.get_challenge(difficulty=difficulty)
        
        return f"""Challenge: {challenge['title']}
ID: {challenge.get('id', 'N/A')}
Difficulty: {challenge.get('difficulty', 'N/A')}
Category: {challenge.get('category', 'N/A')}

{challenge['description']}

To get hints, use: get_hint with '{challenge.get('id','N/A')},1' (for hint 1)
To submit solution, use: evaluate_code with '{challenge.get('id','N/A')}|your_code'"""
    
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
    
    def run(self, query: str):
            """Main entry point for the agent"""
            # Since we are using a ReAct agent, we pass the raw query to the executor.
            # The agent will decide which tool to use based on the input.
            response = self.agent_executor.invoke({"input": query})
            return response.get('output', "No response from agent.")
