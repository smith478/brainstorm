import os
import json
import random
from pathlib import Path
from typing import List, Dict, Optional

from config import Config
from llm_provider import get_llm

class CodingChallenge:
    """Manages coding challenges and feedback"""
    
    def __init__(self, config: Config, llm):
        self.config = config
        self.llm = llm
        self.challenges_path = Path(config.challenges_path)
        self.challenges_path.mkdir(exist_ok=True)
        self.challenges_db = self._load_challenges()
    
    def _load_challenges(self) -> List[Dict]:
        """Load challenges from the challenges directory"""
        challenges = []
        for challenge_file in self.challenges_path.glob("*.json"):
            with open(challenge_file, 'r') as f:
                challenges.append(json.load(f))
        return challenges

    def get_challenge(self, difficulty: Optional[str] = None, 
                      category: Optional[str] = None) -> Dict:
        """Get a coding challenge based on criteria"""
        challenges = self.challenges_db
        
        if difficulty:
            challenges = [c for c in challenges if c['difficulty'] == difficulty]
        if category:
            challenges = [c for c in challenges if c['category'] == category]
        
        if not challenges:
            if not self.challenges_db:
                return {"title": "No challenges found", "description": "Please generate a challenge first."}
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

    def generate_challenge(self, topic: str, difficulty: str = 'medium') -> str:
        """Generate a new coding challenge using an LLM."""
        prompt = f"""Generate a new coding challenge on the topic of '{topic}' with difficulty '{difficulty}'.

The challenge should be self-contained and include:
1.  A unique `id` (e.g., a short, descriptive slug like 'async-web-scraper').
2.  A clear `title`.
3.  A `difficulty` level ('easy', 'medium', or 'hard').
4.  A `category` (e.g., 'web', 'data-structures', 'mlops').
5.  A detailed `description` of the task.
6.  A list of `hints`.
7.  A `test_cases` string containing python code to verify the solution.

Return the challenge as a single JSON object.
"""
        
        challenge_json_str = self.llm.invoke(prompt)
        
        try:
            challenge_data = json.loads(challenge_json_str)
            challenge_id = challenge_data.get('id', f"challenge_{random.randint(1000, 9999)}")
            
            # Save the new challenge
            with open(self.challenges_path / f"{challenge_id}.json", 'w') as f:
                json.dump(challenge_data, f, indent=2)
            
            # Reload challenges
            self.challenges_db = self._load_challenges()
            
            return f"Successfully generated and saved new challenge: {challenge_data.get('title')}"
        except json.JSONDecodeError:
            return "Failed to generate a valid challenge. The LLM did not return valid JSON."
        except Exception as e:
            return f"An error occurred: {e}"
