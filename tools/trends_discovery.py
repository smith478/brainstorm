from typing import List, Dict
from default_api import google_web_search
from collections import Counter
import re

class TrendDiscovery:
    """Discovers trending AI techniques and tools."""

    def discover_trends(self, topics: List[str] = None) -> Dict[str, List[str]]:
        """Discover trending AI techniques and tools on a given topic."""
        if topics is None:
            topics = ["AI tools", "machine learning techniques", "MLOps platforms"]
        
        all_trends = {}

        for topic in topics:
            query = f"trending {topic} 2025"
            try:
                results = google_web_search(query=query)
                if results and results.get("search_results"):
                    all_trends[topic] = self._extract_trends_from_results(results["search_results"])
            except Exception as e:
                print(f"Error searching for {topic}: {e}")

        return all_trends

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
