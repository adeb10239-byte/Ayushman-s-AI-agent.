import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from duckduckgo_search import DDGS

# 1. Web Page Configuration
st.set_page_config(page_title="Ayushman's AI AGENT", layout="centered")
st.title("🤖 AI Agent For Extensive Research on any Topic (By Ayushman Deb)")
st.write("Type What's on your mind below to get extensive details about it!")

# 2. Sidebar Configuration (The "No-Code" Control Panel)
st.sidebar.header("🔧 Configure Your Agent")

# Input for the free Gemini API Key
api_key = st.sidebar.text_input("Enter Gemini API Key:", type="password")
agent_role = st.sidebar.text_input("Agent Role:", value="Expert Research Assistant")
agent_goal = st.sidebar.text_area("Agent Goal:", value="Find the latest trends on a given topic.")
agent_backstory = st.sidebar.text_area("Agent Backstory:", value="You are a fast, high-efficiency assistant.")

# Toggle capability
enable_web_search = st.sidebar.checkbox("Give Agent Web Search Capabilities?", value=True)

# OPTIMIZATION 1: Reduced max_results to 2 to minimize web scraping latency
@tool("Web Search Tool")
def web_search(query: str) -> str:
    """Searches the live internet using DuckDuckGo to find information on a given topic query."""
    try:
        with DDGS() as ddgs:
            results = [r['body'] for r in ddgs.text(query, max_results=2)]
            return "\n\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search failed: {str(e)}"

# 3. Main Dashboard Input
user_prompt = st.text_input("What do you want your AI Agent to do?", placeholder="e.g., Research the current state of electric vehicles")

# 4. Trigger Execution
if st.button("🚀 Run AI Agent"):
    if not api_key:
        st.error("Please enter your Gemini API Key in the sidebar to proceed.")
    elif not user_prompt:
        st.warning("Please give the agent a task query!")
    else:
        with st.spinner("Agent is thinking, planning, and executing..."):
            try:
                # LLM Configuration using the super-fast gemini-2.0-flash
                gemini_llm = LLM(
                    model="gemini/gemini-3.8-flash", 
                    api_key=api_key
                )

                tools_list = []
                if enable_web_search:
                    tools_list.append(web_search)

                # OPTIMIZATION 2: Set max_iter=1 and max_rpm=10 to force the agent 
                # to generate the answer immediately instead of looping indefinitely.
                my_agent = Agent(
                    role=agent_role,
                    goal=agent_goal,
                    backstory=agent_backstory,
                    tools=tools_list,
                    llm=gemini_llm,
                    verbose=True,
                    max_iter=1,  # Stops the "thought loop" after 1 try
                    max_rpm=10   # Limits requests to prevent API throttling
                )

                # Define the Task using UI query
                my_task = Task(
                    description=user_prompt,
                    expected_output="A concise, high-level summary report addressing the query.",
                    agent=my_agent
                )

                # Assemble the Crew
                crew = Crew(agents=[my_agent], tasks=[my_task])

                # Execute
                result = crew.kickoff()

                # Display the Result
                st.success("✨ Task Completed!")
                st.markdown("### 📋 Agent Output")
                st.write(str(result))

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
