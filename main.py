import json
from config import Config
from agent import BrainstormAgent

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
    - 'generate challenge [topic]' - Generate a new coding challenge
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
            elif user_input.lower().startswith('generate challenge'):
                topic = user_input[len('generate challenge'):].strip()
                if topic:
                    response = agent.coding_challenges.generate_challenge(topic)
                    print(f"\n{response}")
                else:
                    print("\nPlease provide a topic for the challenge.")
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
