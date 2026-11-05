from dotenv import load_dotenv
load_dotenv()
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
import streamlit as st 
from langgraph.checkpoint.memory import MemorySaver

st.set_page_config(
    page_title="AI SMART STUDY BUDDY",
    page_icon="🎯",
    layout="centered"
)

# ---------------- Header ---------------- #

st.title("🎯AI SMART STUDY BUDDY")
st.caption("STUDY AND LEARN USING SMART STUDY BUDDY AI")

st.divider()

# ---------------- Sidebar ---------------- #

with st.sidebar:
    st.header("Interview Settings")

    st.info(
        """
This AI helps you prepare for interviews and exams.

It can generate:
- explanation of topic you want 
- Technical Questions
- Behavioral Questions
- Coding Questions
"""
    )

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


llm=ChatGroq(model="openai/gpt-oss-20b")
search= GoogleSerperAPIWrapper()
memory=MemorySaver()
@st.cache_resource
def agentss():
    agnt=create_agent(model=llm,
                 tools=[search.run],
                 system_prompt="your are a ai sudy buddy ask the user the topic he or she wants to learn. you ask user the topic input diffulty selection (easy,medium,hard) and then give simple but effective explanation with proper exmple and real life example and then at last give him 5-10 mcq and 5 -10 interview question with answer about the topic",
                 checkpointer=memory
                 )
    return agnt
agent=agentss()

if "messages" not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt=st.chat_input("Tell me how can i help you ")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("User"):
        st.markdown(prompt)
    
    response = agent.invoke(
        {"messages": [{"role": "user","content": prompt}]},
        {"configurable": {"thread_id": "interview-chat"}})
    
    answer = response["messages"][-1].content
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.markdown(answer)