from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
import arxiv
from config import Config

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
