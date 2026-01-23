import streamlit as st
import json
import re
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError
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

# Enable JSON mode for structured output
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    api_key=api_key,
    model_kwargs={"response_format": {"type": "json_object"}}
)
# Separate LLM for conversational responses (without JSON mode)
conversational_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key
)
parser = PydanticOutputParser(pydantic_object=Workflow)
format_instructions = parser.get_format_instructions()

system_prompt = """
You are an expert Automation Architect for a platform like Relay.app or Zapier.
Your goal is to interpret natural language requests and UPDATE the current workflow configuration.

CURRENT WORKFLOW STATE:
{current_state}

SCHEMA REQUIREMENTS:
{format_instructions}

================================================================================
🚫 FORBIDDEN FIELDS - NEVER USE THESE IN STEPS:
================================================================================
The following fields are FORBIDDEN and will cause validation errors:
- "name" (use "id" instead)
- "type" (FORBIDDEN - do not use)
- "config" (FORBIDDEN - do not use)
- "next_steps" (FORBIDDEN - do not use)
- "condition" (FORBIDDEN - do not use)
- Any nested objects or arrays in steps

================================================================================
✅ REQUIRED FIELDS - EVERY STEP MUST HAVE EXACTLY THESE 4 FIELDS:
================================================================================
Each step in the "steps" array MUST have exactly these 4 required fields (no more, no less):
1. "id": A unique identifier string (e.g., "step_1", "step_2", "step_3")
2. "app": The app/service name (e.g., "Slack", "Gmail", "Linear", "Customer Support System", "Ecommerce Platform")
3. "action": The action description (e.g., "Send Message", "Create Ticket", "Categorize Issue", "Find Order")
4. "details": A short summary of configuration (e.g., "Channel: #support", "Category: refund", "Search by customer name")

================================================================================
✅ CORRECT EXAMPLES:
================================================================================

EXAMPLE 1 - Simple Slack Notification:
{{
  "name": "New Lead Alert",
  "trigger": "New Form Submission",
  "steps": [
    {{
      "id": "step_1",
      "app": "Slack",
      "action": "Send Message",
      "details": "Channel: #sales, Message: New lead received"
    }}
  ]
}}

EXAMPLE 2 - Multi-Step Ecommerce Workflow:
{{
  "name": "Order Issue Handler",
  "trigger": "New Customer Message",
  "steps": [
    {{
      "id": "step_1",
      "app": "Customer Support System",
      "action": "Categorize Issue",
      "details": "Categories: refund, replacement, cancellation"
    }},
    {{
      "id": "step_2",
      "app": "Ecommerce Platform",
      "action": "Find Customer Order",
      "details": "Search by customer name and order number"
    }},
    {{
      "id": "step_3",
      "app": "Ecommerce Platform",
      "action": "Get Order Status",
      "details": "Retrieve shipping and delivery status"
    }}
  ]
}}

EXAMPLE 3 - Customer Support Flow:
{{
  "name": "Support Ticket Automation",
  "trigger": "New Support Email",
  "steps": [
    {{
      "id": "step_1",
      "app": "Gmail",
      "action": "Send Auto-Reply",
      "details": "Template: acknowledgment message"
    }},
    {{
      "id": "step_2",
      "app": "Linear",
      "action": "Create Ticket",
      "details": "Project: Support, Priority: High"
    }},
    {{
      "id": "step_3",
      "app": "Slack",
      "action": "Notify Team",
      "details": "Channel: #support-alerts"
    }}
  ]
}}

================================================================================
❌ INCORRECT EXAMPLES - DO NOT GENERATE THESE:
================================================================================

WRONG - Using forbidden "type" and "config" fields:
{{
  "action": "Categorize Issue",
  "type": "AI-Powered Issue Categorization",  ❌ FORBIDDEN
  "config": {{"categories": [...]}}  ❌ FORBIDDEN
}}

WRONG - Missing required fields:
{{
  "name": "Categorize Issue",  ❌ Should be "id", not "name"
  "action": "Categorize Issue"  ❌ Missing "app" and "details"
}}

WRONG - Using nested objects:
{{
  "id": "step_1",
  "app": "Ecommerce Platform",
  "action": "Find Order",
  "config": {{"searchParams": {{"customerName": "..."}}}}  ❌ No nested objects
}}

CORRECT VERSION of the above:
{{
  "id": "step_1",
  "app": "Ecommerce Platform",
  "action": "Find Customer Order",
  "details": "Search by customer name and order number"
}}

================================================================================
INSTRUCTIONS:
================================================================================
1. Analyze the user's request.
2. If they are CREATING a new flow, overwrite the current state.
3. If they are MODIFYING (e.g., "add a delay", "change slack channel"), update the existing steps intelligently.
4. Output ONLY valid JSON matching the Workflow schema. No chat, no markdown, no code blocks.
5. EVERY step MUST have exactly these 4 fields: id, app, action, details.
6. NEVER use: type, config, name (in steps), next_steps, condition, or nested objects.
7. If the user describes complex logic, simplify it into sequential steps with clear actions.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{user_input}"),
])

# Conversational prompt for friendly responses
conversational_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly and helpful Workflow Architect assistant. 
After creating or updating a workflow, provide a conversational, engaging response that:
- Acknowledges what the user asked for
- Explains what workflow was created or modified in a friendly way
- Summarizes the key steps naturally
- Highlights interesting aspects
- Asks helpful follow-up questions when appropriate
- Keeps responses concise (2-3 sentences typically)
- Uses a warm, professional tone

Be conversational and helpful, not robotic."""),
    ("user", """User request: {user_input}

Generated workflow:
{workflow_summary}

Provide a friendly, conversational response explaining what workflow was created or modified. Be engaging and helpful.""")
])

# Function to format workflow as a readable summary for conversational LLM
def format_workflow_summary(workflow: Workflow) -> str:
    """Format a workflow into a readable text summary."""
    summary = f"Workflow Name: {workflow.name}\n"
    summary += f"Trigger: {workflow.trigger}\n"
    summary += f"Steps ({len(workflow.steps)}):\n"
    for i, step in enumerate(workflow.steps, 1):
        summary += f"  {i}. {step.app} - {step.action}"
        if step.details:
            summary += f" ({step.details})"
        summary += "\n"
    return summary

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
                
                # Generate conversational response
                workflow_summary = format_workflow_summary(new_workflow)
                conversational_chain = conversational_prompt | conversational_llm
                conversational_response = conversational_chain.invoke({
                    "user_input": user_input,
                    "workflow_summary": workflow_summary
                })
                bot_msg = conversational_response.content
                st.session_state.messages.append({"role": "assistant", "content": bot_msg})
                st.rerun()
            except ValidationError as e:
                # Parse validation errors to show missing fields clearly
                error_messages = []
                step_errors = {}
                
                for error in e.errors():
                    if "steps" in str(error.get("loc", [])):
                        # Extract step index from error location
                        loc = error.get("loc", ())
                        step_idx = None
                        for i, item in enumerate(loc):
                            if item == "steps" and i + 1 < len(loc):
                                step_idx = loc[i + 1]
                                break
                        
                        field = error.get("loc", ())[-1] if loc else "unknown"
                        msg = error.get("msg", "Validation error")
                        
                        if step_idx is not None:
                            if step_idx not in step_errors:
                                step_errors[step_idx] = []
                            step_errors[step_idx].append(f"Missing required field: '{field}'")
                
                # Build user-friendly error message
                error_msg = "**Workflow Validation Error**\n\n"
                error_msg += "The AI generated workflow steps with missing or incorrect fields.\n\n"
                
                if step_errors:
                    error_msg += "**Issues by step:**\n"
                    for step_idx, errors in sorted(step_errors.items()):
                        error_msg += f"\n**Step {step_idx + 1}:**\n"
                        for err in errors:
                            error_msg += f"  - {err}\n"
                
                error_msg += "\n**Required fields for each step:**\n"
                error_msg += "  - `id`: Unique identifier (e.g., 'step_1')\n"
                error_msg += "  - `app`: App/service name (e.g., 'Slack', 'Gmail')\n"
                error_msg += "  - `action`: Action description (e.g., 'Send Message')\n"
                error_msg += "  - `details`: Configuration summary (e.g., 'Channel: #support')\n"
                
                error_msg += "\n**Forbidden fields:** `type`, `config`, `name` (in steps), `next_steps`, `condition`\n"
                
                st.error(error_msg)
                
                # Generate conversational error response
                error_conversational_prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a friendly Workflow Architect assistant. 
When there's a validation error, provide a helpful, conversational response that:
- Acknowledges the error in a friendly way
- Suggests how to fix it
- Encourages the user to try again with clearer instructions
- Keeps it brief and helpful"""),
                    ("user", """The user requested: {user_input}

A validation error occurred. The workflow I tried to generate had missing or incorrect fields.

Provide a friendly, helpful response asking the user to rephrase their request with simpler, more direct instructions.""")
                ])
                error_response = (error_conversational_prompt | conversational_llm).invoke({
                    "user_input": user_input
                })
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_response.content
                })
            except Exception as e:
                error_msg = f"**Error:** {str(e)}\n\n"
                error_msg += "This might be due to the AI generating invalid JSON or not following the schema. "
                error_msg += "Please try rephrasing your request."
                st.error(error_msg)
                
                # Generate conversational error response
                error_conversational_prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a friendly Workflow Architect assistant. 
When there's an error processing a request, provide a helpful, conversational response that:
- Acknowledges the error in a friendly way
- Suggests the user try again with a clearer description
- Keeps it brief and encouraging"""),
                    ("user", """The user requested: {user_input}

An error occurred while processing the request: {error_message}

Provide a friendly, helpful response asking the user to try again with a clearer description.""")
                ])
                error_response = (error_conversational_prompt | conversational_llm).invoke({
                    "user_input": user_input,
                    "error_message": str(e)
                })
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_response.content
                })

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