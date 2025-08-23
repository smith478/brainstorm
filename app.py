import streamlit as st
import os
import json
import subprocess
import sys
from pathlib import Path

from config import Config
from llm_provider import get_llm
from tools.coding_challenge import CodingChallenge

# Initialize components
config = Config.load()
llm = get_llm(config)
coding_challenges = CodingChallenge(config, llm)

st.set_page_config(layout="wide")
st.title("Brainstorm AI Learning Assistant")

# Sidebar for navigation/features
st.sidebar.header("Features")
selected_feature = st.sidebar.radio("Go to", ["Coding Challenges", "Paper Discovery", "AI Discussion"])

if selected_feature == "Coding Challenges":
    st.header("Coding Challenges")

    # Challenge generation/selection
    if st.button("Generate New Challenge"):
        with st.spinner("Generating a new challenge..."):
            # This will call the LLM to generate a new challenge and save it
            # The generate_challenge method returns a string message
            generation_message = coding_challenges.generate_challenge(topic="Python programming", difficulty="medium")
            st.success(generation_message)
            st.session_state.current_challenge = None # Clear current challenge to force reload

    if "current_challenge" not in st.session_state or st.session_state.current_challenge is None:
        # Load a random challenge if none is selected or after generation
        st.session_state.current_challenge = coding_challenges.get_challenge()
        if "No challenges found" in st.session_state.current_challenge.get("title", ""):
            st.warning("No challenges available. Please generate one first.")
            st.stop()

    challenge = st.session_state.current_challenge
    st.subheader(challenge.get("title", "No Title"))
    st.write(f"**Difficulty:** {challenge.get('difficulty', 'N/A')} | **Category:** {challenge.get('category', 'N/A')}")
    st.markdown("---")
    st.markdown(challenge.get("description", "No description provided."))
    st.markdown("---")

    # Code editor
    user_code = st.text_area("Write your solution here:", height=300, value="def solve():\n    # Your code here\n    pass")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run Tests"):
            if user_code.strip() == "":
                st.warning("Please enter your code before running tests.")
            else:
                challenge_id = challenge.get("id")
                test_cases = challenge.get("test_cases")

                if not challenge_id or not test_cases:
                    st.error("Challenge ID or test cases missing. Cannot run tests.")
                else:
                    # Create a temporary file to run the user's code and tests
                    temp_file_path = Path("temp_challenge_solution.py")
                    try:
                        with open(temp_file_path, "w") as f:
                            f.write(user_code)
                            f.write("\n\n")
                            f.write(test_cases) # Append the test cases

                        st.info("Running tests...")
                        # Execute the temporary file
                        # WARNING: Running user-provided code directly can be a security risk.
                        # For a production system, consider a more robust sandboxing solution (e.g., Docker, isolated environments).
                        process = subprocess.run(
                            [sys.executable, str(temp_file_path)],
                            capture_output=True,
                            text=True,
                            timeout=10 # Timeout to prevent infinite loops
                        )

                        st.subheader("Test Results:")
                        if process.returncode == 0:
                            st.success("Tests completed successfully!")
                            st.code(process.stdout)
                        else:
                            st.error("Tests failed or encountered an error.")
                            st.code(process.stdout + process.stderr)

                    except subprocess.TimeoutExpired:
                        st.error("Code execution timed out. Your code might be in an infinite loop or too slow.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during test execution: {e}")
                    finally:
                        if temp_file_path.exists():
                            os.remove(temp_file_path)

    with col2:
        if st.button("Get LLM Feedback"):
            if user_code.strip() == "":
                st.warning("Please enter your code before getting feedback.")
            else:
                challenge_id = challenge.get("id")
                if not challenge_id:
                    st.error("Challenge ID missing. Cannot get feedback.")
                else:
                    with st.spinner("Getting feedback from LLM..."):
                        feedback = coding_challenges.evaluate_solution(challenge_id, user_code)
                        st.subheader("LLM Feedback:")
                        st.markdown(feedback)

elif selected_feature == "Paper Discovery":
    st.header("Paper Discovery (Coming Soon!)")
    st.info("This feature will allow you to discover recent AI research papers.")

elif selected_feature == "AI Discussion":
    st.header("AI Discussion (Coming Soon!)")
    st.info("This feature will allow you to discuss AI topics using a RAG pipeline.")
