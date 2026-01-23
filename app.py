import streamlit as st
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Relay Lite: AI Workflow Architect", layout="wide", page_icon="⚡")

# Custom CSS
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; font-weight: 700; margin-bottom: 0rem; color: #333;}
    .sub-header {font-size: 1rem; color: #666; margin-bottom: 2rem;}
    .card {background-color: #f9f9f9; padding: 20px; border-radius: 10px; border: 1px solid #ddd; margin-bottom: 20px;}
    </style>
""", unsafe_allow_html=True)

# --- 2. DATA MODELS ---
class WorkflowStep(BaseModel):
    id: str = Field(description="Unique ID for the step (e.g., 'step_1')")
    app: str = Field(description="The app involved (e.g., 'Slack', 'Gmail', 'Linear')")
    action: str = Field(description="The action to take (e.g., 'Send Message', 'Create Ticket')")
    details: str = Field(description="Short summary of config (e.g., 'Channel: #support')")

class Workflow(BaseModel):
    name: str = Field(description="A creative name for this automation")
    trigger: str = Field(description="The event that starts it (e.g., 'New Email received')")
    steps: List[WorkflowStep] = Field(description="The sequence of actions to take")

# --- 3. STATE MANAGEMENT ---
if "workflow_data" not in st.session_state:
    st.session_state.workflow_data = Workflow(
        name="Untitled Workflow",
        trigger="Manual Trigger",
        steps=[]
    )
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your Workflow Architect. Describe a process you want to automate (e.g., 'When a new lead arrives in Typeform, send them an email and alert the team on Slack')."}
    ]

# --- 4. THE AI ENGINE ---
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
else:
    st.error("🚨 GROQ_API_KEY missing from secrets.")
    st.stop()

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
parser = PydanticOutputParser(pydantic_object=Workflow)
format_instructions = parser.get_format_instructions()

system_prompt = """
You are an expert Automation Architect for a platform like Relay.app or Zapier.
Your goal is to interpret natural language requests and UPDATE the current workflow configuration.

CURRENT WORKFLOW STATE:
{current_state}

SCHEMA REQUIREMENTS:
{format_instructions}

CRITICAL: Each step in the "steps" array MUST have exactly these 4 required fields:
- "id": A unique identifier string (e.g., "step_1", "step_2")
- "app": The app/service name (e.g., "Slack", "Gmail", "Linear", "Customer Support System")
- "action": The action description (e.g., "Send Message", "Create Ticket", "Categorize Issue")
- "details": A short summary of configuration (e.g., "Channel: #support", "Category: refund")

EXAMPLE of a correct step:
{{
  "id": "step_1",
  "app": "Slack",
  "action": "Send Message",
  "details": "Channel: #support"
}}

DO NOT use fields like "name", "next_steps", or "condition" in steps. Only use: id, app, action, details.

INSTRUCTIONS:
1. Analyze the user's request.
2. If they are CREATING a new flow, overwrite the current state.
3. If they are MODIFYING (e.g., "add a delay", "change slack channel"), update the existing steps intelligently.
4. Output ONLY valid JSON matching the Workflow schema. No chat, no markdown.
5. Ensure every step has all 4 required fields: id, app, action, details.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{user_input}"),
])

# --- 5. SIDEBAR: CHAT ---
with st.sidebar:
    st.markdown("### 💬 Architect Chat")
    
    for msg in st.session_state.messages:
        avatar = "⚡" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.write(msg["content"])

    if user_input := st.chat_input("Describe your workflow..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.write(user_input)

        with st.spinner("Architecting solution..."):
            try:
                current_json = st.session_state.workflow_data.model_dump_json()
                chain = prompt | llm | parser
                new_workflow = chain.invoke({
                    "current_state": current_json,
                    "user_input": user_input,
                    "format_instructions": format_instructions
                })
                st.session_state.workflow_data = new_workflow
                bot_msg = f"Updated workflow: **{new_workflow.name}** with {len(new_workflow.steps)} steps."
                st.session_state.messages.append({"role": "assistant", "content": bot_msg})
                st.rerun()
            except Exception as e:
                st.error(f"AI Error: {e}")

# --- 6. MAIN PANEL: VISUALIZATION ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<p class="main-header">⚡ Relay Lite</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">Current Flow: <b>{st.session_state.workflow_data.name}</b></p>', unsafe_allow_html=True)

    # GRAPHVIZ RENDERER
    graph_code = f"""
    digraph G {{
        rankdir=TB;
        node [shape=box, style="filled,rounded", fontname="Helvetica", penwidth=0];
        edge [fontname="Helvetica", color="#aaaaaa"];
        bgcolor="transparent";
        
        start [label="{st.session_state.workflow_data.trigger}", fillcolor="#FF4B4B", fontcolor="white", fontsize=12, margin="0.2,0.1"];
        
        {
            "\n".join([
                f'{step.id} [label=<{step.app}<BR/><FONT POINT-SIZE="10">{step.action}</FONT><BR/><FONT POINT-SIZE="8" COLOR="#555">{step.details}</FONT>>, fillcolor="white", color="#ddd", penwidth=1];' 
                for step in st.session_state.workflow_data.steps
            ])
        }

        start -> {st.session_state.workflow_data.steps[0].id if st.session_state.workflow_data.steps else "end"};
        {
            "\n".join([
                f"{st.session_state.workflow_data.steps[i].id} -> {st.session_state.workflow_data.steps[i+1].id};"
                for i in range(len(st.session_state.workflow_data.steps)-1)
            ])
        }
    }}
    """
    st.graphviz_chart(graph_code, use_container_width=True)

with col2:
    st.markdown("### 🛠️ Configuration")
    st.caption("Live JSON generated by Llama 3")
    st.json(st.session_state.workflow_data.model_dump())