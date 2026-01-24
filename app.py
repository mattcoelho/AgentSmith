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
    .step-highlight {background-color: #fff3cd !important; border: 2px solid #ffc107 !important; border-radius: 4px; padding: 8px; margin: 4px 0;}
    .step-anchor {scroll-margin-top: 20px;}
    .json-step-section {padding: 8px; margin: 4px 0; border-radius: 4px; transition: background-color 0.3s;}
    .json-step-section.highlighted {background-color: #fff3cd; border: 2px solid #ffc107;}
    svg g.node {cursor: pointer;}
    svg g.node:hover {opacity: 0.8;}
    </style>
""", unsafe_allow_html=True)

# --- 2. DATA MODELS ---
class Branch(BaseModel):
    condition: str = Field(description="Condition description (e.g., 'If order status is Shipped', 'If amount > $100')")
    next_step_id: str = Field(description="ID of the step to execute if condition is true")

class WorkflowStep(BaseModel):
    id: str = Field(description="Unique ID for the step (e.g., 'step_1')")
    app: str = Field(description="The app involved (e.g., 'Slack', 'Gmail', 'Linear')")
    action: str = Field(description="The action to take (e.g., 'Send Message', 'Create Ticket')")
    details: str = Field(description="Short summary of config (e.g., 'Channel: #support')")
    next_step_id: Optional[str] = Field(default=None, description="Default next step ID for linear flow or default path")
    branches: Optional[List[Branch]] = Field(default=None, description="Conditional branches for decision points")

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
if "selected_step_id" not in st.session_state:
    st.session_state.selected_step_id = None

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
Your role is to help users understand, describe, and modify workflows.

CURRENT WORKFLOW STATE:
{current_state}

SCHEMA REQUIREMENTS:
{format_instructions}

================================================================================
🎯 UNDERSTANDING USER INTENT:
================================================================================
Before modifying the workflow, determine what the user actually wants:

1. QUESTIONS OR DESCRIPTIONS:
   - User asks "what does this workflow do?", "describe the workflow", "explain this"
   - User asks "how does step X work?", "what is the trigger?"
   - User asks general questions about automation or workflows
   - User wants clarification or information
   → RETURN THE CURRENT WORKFLOW UNCHANGED (output the exact current_state as-is)

2. CREATING A NEW WORKFLOW:
   - User says "create a workflow for...", "build an automation that..."
   - User describes a completely new process from scratch
   - → CREATE a new workflow matching their description

3. MODIFYING EXISTING WORKFLOW:
   - User says "add a step", "change the channel to...", "remove step X"
   - User wants to update, edit, or modify the current workflow
   - → UPDATE the workflow intelligently while preserving unchanged parts

4. UNCLEAR REQUESTS:
   - If you're unsure whether the user wants to modify or just ask questions
   - → RETURN THE CURRENT WORKFLOW UNCHANGED (better to be safe than modify unintentionally)

IMPORTANT: Only modify the workflow when the user EXPLICITLY wants to create or change something.
If they're just asking questions, describing, or seeking information, return the current workflow unchanged.

================================================================================
🚫 FORBIDDEN FIELDS - NEVER USE THESE IN STEPS:
================================================================================
The following fields are FORBIDDEN and will cause validation errors:
- "name" (use "id" instead)
- "type" (FORBIDDEN - do not use)
- "config" (FORBIDDEN - do not use)
- Any nested objects or arrays in steps (except "branches" which is allowed)

================================================================================
✅ REQUIRED FIELDS - EVERY STEP MUST HAVE EXACTLY THESE 4 FIELDS:
================================================================================
Each step in the "steps" array MUST have exactly these 4 required fields:
1. "id": A unique identifier string (e.g., "step_1", "step_2", "step_3")
2. "app": The app/service name (e.g., "Slack", "Gmail", "Linear", "Customer Support System", "Ecommerce Platform")
3. "action": The action description (e.g., "Send Message", "Create Ticket", "Categorize Issue", "Find Order")
4. "details": A short summary of configuration (e.g., "Channel: #support", "Category: refund", "Search by customer name")

OPTIONAL FIELDS for branching and decision trees:
- "next_step_id": Optional string - ID of the default next step (for linear flow or default path)
- "branches": Optional array of Branch objects - Conditional branches for decision points
  Each Branch has:
    - "condition": String describing the condition (e.g., "If order status is Shipped", "If amount > $100")
    - "next_step_id": String ID of the step to execute if condition is true

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
      "details": "Template: acknowledgment message",
      "next_step_id": "step_2"
    }},
    {{
      "id": "step_2",
      "app": "Linear",
      "action": "Create Ticket",
      "details": "Project: Support, Priority: High",
      "next_step_id": "step_3"
    }},
    {{
      "id": "step_3",
      "app": "Slack",
      "action": "Notify Team",
      "details": "Channel: #support-alerts"
    }}
  ]
}}

EXAMPLE 4 - Branched Decision Tree Workflow:
{{
  "name": "Order Processing with Conditions",
  "trigger": "New Order Received",
  "steps": [
    {{
      "id": "step_1",
      "app": "Ecommerce Platform",
      "action": "Get Order Details",
      "details": "Retrieve order amount and status",
      "next_step_id": "step_2"
    }},
    {{
      "id": "step_2",
      "app": "Ecommerce Platform",
      "action": "Check Order Amount",
      "details": "Evaluate order value",
      "branches": [
        {{
          "condition": "If order amount > $100",
          "next_step_id": "step_3"
        }},
        {{
          "condition": "If order amount <= $100",
          "next_step_id": "step_4"
        }}
      ]
    }},
    {{
      "id": "step_3",
      "app": "Slack",
      "action": "Notify High Value Order",
      "details": "Channel: #high-value-orders",
      "next_step_id": "step_5"
    }},
    {{
      "id": "step_4",
      "app": "Gmail",
      "action": "Send Standard Confirmation",
      "details": "Template: standard order confirmation",
      "next_step_id": "step_5"
    }},
    {{
      "id": "step_5",
      "app": "Ecommerce Platform",
      "action": "Process Payment",
      "details": "Charge customer and update order"
    }}
  ]
}}

EXAMPLE 5 - Multi-Way Branch:
{{
  "name": "Issue Categorization Workflow",
  "trigger": "New Customer Message",
  "steps": [
    {{
      "id": "step_1",
      "app": "Customer Support System",
      "action": "Categorize Issue",
      "details": "Analyze message content",
      "branches": [
        {{
          "condition": "If category is refund",
          "next_step_id": "step_2"
        }},
        {{
          "condition": "If category is replacement",
          "next_step_id": "step_3"
        }},
        {{
          "condition": "If category is cancellation",
          "next_step_id": "step_4"
        }}
      ]
    }},
    {{
      "id": "step_2",
      "app": "Ecommerce Platform",
      "action": "Process Refund",
      "details": "Initiate refund workflow"
    }},
    {{
      "id": "step_3",
      "app": "Ecommerce Platform",
      "action": "Initiate Replacement",
      "details": "Create replacement order"
    }},
    {{
      "id": "step_4",
      "app": "Ecommerce Platform",
      "action": "Process Cancellation",
      "details": "Cancel order if not shipped"
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

WRONG - Using null in branch.next_step_id:
{{
  "id": "step_1",
  "app": "Ecommerce Platform",
  "action": "Check Eligibility",
  "details": "Evaluate if action is possible",
  "branches": [
    {{
      "condition": "If eligible",
      "next_step_id": "step_2"
    }},
    {{
      "condition": "If not eligible",
      "next_step_id": null  ❌ FORBIDDEN - must be a step ID string
    }}
  ]
}}

CORRECT VERSION - Omit the branch or use default next_step_id:
{{
  "id": "step_1",
  "app": "Ecommerce Platform",
  "action": "Check Eligibility",
  "details": "Evaluate if action is possible",
  "next_step_id": "step_end",
  "branches": [
    {{
      "condition": "If eligible",
      "next_step_id": "step_2"
    }}
  ]
}}

================================================================================
🌳 BRANCHED WORKFLOWS - DECISION TREES:
================================================================================
When the user describes conditional logic (if/then, if/else, decision points), use BRANCHES:

- Use "branches" array when a step has multiple possible next steps based on conditions
- Use "next_step_id" for linear flow or the default path when no branches match
- Each branch MUST have a clear "condition" description and "next_step_id"
- CRITICAL: branch.next_step_id MUST ALWAYS be a valid step ID string - NEVER use null, None, or empty string
- If a branch condition should lead to "end" or "do nothing", either:
  * Omit that branch entirely (let the step's default next_step_id handle it)
  * Create a final "End" step and point the branch to it
  * Use the step's next_step_id as the default path instead of creating a branch
- Steps with branches create decision points in the workflow
- Multiple branches allow for if/else if/else patterns

When to use branches:
- User says "if X then Y, else Z"
- User describes conditional logic or decision points
- User mentions "depending on", "based on", "when", "if"
- User wants different paths for different scenarios

When to use next_step_id:
- Simple linear flow (step 1 -> step 2 -> step 3)
- Default path when branches don't match
- Final step in a branch path

================================================================================
INSTRUCTIONS:
================================================================================
1. FIRST: Determine user intent - are they asking questions, describing, or wanting to modify?
2. If asking questions or seeking information → RETURN CURRENT WORKFLOW UNCHANGED
3. If CREATING a new flow → overwrite the current state with the new workflow
4. If MODIFYING (e.g., "add a delay", "change slack channel") → update existing steps intelligently
5. Output ONLY valid JSON matching the Workflow schema. No chat, no markdown, no code blocks.
6. EVERY step MUST have exactly these 4 required fields: id, app, action, details.
7. OPTIONAL: Add "next_step_id" for linear flow or "branches" for conditional logic.
8. When the user describes conditional logic (if/then, if/else, decision points), USE BRANCHES to create decision trees instead of simplifying to sequential steps.
9. CRITICAL: In branches array, every branch.next_step_id MUST be a valid step ID string - NEVER null, None, or empty.
10. NEVER use: type, config, name (in steps), or nested objects (except branches array).
11. WHEN IN DOUBT: Return the current workflow unchanged rather than making unintended modifications.
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

# Function to render JSON with scrollable anchors for each step
def render_json_with_anchors(workflow: Workflow, selected_step_id: Optional[str] = None) -> str:
    """Render workflow JSON with HTML anchors for each step."""
    import json
    workflow_dict = workflow.model_dump()
    
    # Convert to formatted JSON string
    json_str = json.dumps(workflow_dict, indent=2)
    
    # Split JSON into lines and add anchors for steps
    lines = json_str.split('\n')
    result_lines = []
    in_steps_array = False
    step_brace_indent = None
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Detect when we enter the steps array
        if '"steps"' in line and '[' in line:
            in_steps_array = True
            result_lines.append(line)
            i += 1
            continue
        
        # Detect when we exit the steps array
        if in_steps_array and line.strip() == ']':
            in_steps_array = False
            result_lines.append(line)
            i += 1
            continue
        
        # Detect step object start '{'
        if in_steps_array and line.strip() == '{':
            step_brace_indent = len(line) - len(line.lstrip())
            result_lines.append(line)
            i += 1
            continue
        
        # Detect step ID line to add anchor before the step object
        if in_steps_array and '"id"' in line and step_brace_indent is not None:
            import re
            match = re.search(r'"id"\s*:\s*"([^"]+)"', line)
            if match:
                step_id = match.group(1)
                anchor_id = f"step-{step_id}"
                # Insert anchor right after the opening brace
                # Find the last '{' line we added
                for j in range(len(result_lines) - 1, -1, -1):
                    if result_lines[j].strip() == '{' and len(result_lines[j]) - len(result_lines[j].lstrip()) == step_brace_indent:
                        # Add anchor on the next line with proper indent
                        anchor_line = ' ' * (step_brace_indent + 2) + f'<a id="{anchor_id}" class="step-anchor"></a>'
                        result_lines.insert(j + 1, anchor_line)
                        break
                step_brace_indent = None
        
        result_lines.append(line)
        i += 1
    
    # Join lines and wrap in container
    json_html = '\n'.join(result_lines)
    html_content = f'<div id="json-container" class="json-scroll-container"><pre style="margin: 0; white-space: pre-wrap; word-wrap: break-word; font-family: monospace;">{json_html}</pre></div>'
    return html_content

# Function to format workflow as a readable summary for conversational LLM
def format_workflow_summary(workflow: Workflow) -> str:
    """Format a workflow into a readable text summary."""
    summary = f"Workflow Name: {workflow.name}\n"
    summary += f"Trigger: {workflow.trigger}\n"
    summary += f"Steps ({len(workflow.steps)}):\n"
    for i, step in enumerate(workflow.steps, 1):
        summary += f"  {i}. {step.id}: {step.app} - {step.action}"
        if step.details:
            summary += f" ({step.details})"
        summary += "\n"
        
        # Add branch information
        if step.branches:
            summary += "      Branches:\n"
            for branch in step.branches:
                summary += f"        - {branch.condition} -> {branch.next_step_id}\n"
        elif step.next_step_id:
            summary += f"      Next: {step.next_step_id}\n"
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

    # GRAPHVIZ RENDERER with branch support
    steps = st.session_state.workflow_data.steps
    step_ids = {step.id: step for step in steps}
    
    # Build node definitions
    node_definitions = []
    for step in steps:
        # Use diamond shape for decision points (steps with branches)
        shape = "diamond" if step.branches else "box"
        node_definitions.append(
            f'{step.id} [label=<{step.app}<BR/><FONT POINT-SIZE="10">{step.action}</FONT><BR/><FONT POINT-SIZE="8" COLOR="#555">{step.details}</FONT>>, shape={shape}, style="filled,rounded", fillcolor="white", color="#ddd", penwidth=1];'
        )
    
    # Build edge definitions
    edge_definitions = []
    
    # Connect start to first step (if exists)
    if steps:
        first_step_id = steps[0].id
        edge_definitions.append(f'start -> {first_step_id};')
    
    # Process each step's connections
    for i, step in enumerate(steps):
        if step.branches:
            # Step has branches - create edges for each branch with condition labels
            for branch in step.branches:
                # Escape quotes in condition for Graphviz
                condition = branch.condition.replace('"', '\\"')
                if branch.next_step_id in step_ids:
                    edge_definitions.append(f'{step.id} -> {branch.next_step_id} [label="{condition}", fontsize=9, color="#4A90E2"];')
        elif step.next_step_id:
            # Step has explicit next_step_id
            if step.next_step_id in step_ids:
                edge_definitions.append(f'{step.id} -> {step.next_step_id};')
        else:
            # Fallback to sequential connection (backward compatibility)
            if i + 1 < len(steps):
                edge_definitions.append(f'{step.id} -> {steps[i+1].id};')
    
    graph_code = f"""
    digraph G {{
        rankdir=TB;
        node [fontname="Helvetica"];
        edge [fontname="Helvetica", color="#aaaaaa"];
        bgcolor="transparent";
        
        start [label="{st.session_state.workflow_data.trigger}", fillcolor="#FF4B4B", fontcolor="white", fontsize=12, margin="0.2,0.1", shape=ellipse];
        
        {chr(10).join(node_definitions)}
        
        {chr(10).join(edge_definitions)}
    }}
    """
    st.graphviz_chart(graph_code, use_container_width=True)
    
    # Inject JavaScript to add click handlers to graphviz nodes
    step_ids_list = [step.id for step in steps]
    step_ids_json = json.dumps(step_ids_list)
    
    javascript_code = f"""
    <script>
    (function() {{
        function addClickHandlers() {{
            // Find all SVG elements (graphviz renders as SVG)
            const svgs = document.querySelectorAll('svg');
            if (svgs.length === 0) {{
                // Retry if SVG not loaded yet
                setTimeout(addClickHandlers, 200);
                return;
            }}
            
            const stepIds = {step_ids_json};
            
            svgs.forEach(svg => {{
                // Find all node groups (graphviz nodes are in <g> elements)
                // Graphviz uses the step ID as part of the node ID
                const nodes = svg.querySelectorAll('g');
                
                nodes.forEach(node => {{
                    // Get the node's ID - graphviz format varies, try to extract step ID
                    const nodeId = node.getAttribute('id') || '';
                    const titleElement = node.querySelector('title');
                    const titleText = titleElement ? titleElement.textContent : '';
                    
                    // Try to find matching step ID
                    let matchedStepId = null;
                    for (const stepId of stepIds) {{
                        // Graphviz may use the step ID directly or with modifications
                        if (nodeId.includes(stepId) || titleText.includes(stepId) || 
                            nodeId.replace(/[^a-zA-Z0-9_]/g, '') === stepId.replace(/[^a-zA-Z0-9_]/g, '')) {{
                            matchedStepId = stepId;
                            break;
                        }}
                    }}
                    
                    // Also check text content of the node for step ID
                    if (!matchedStepId) {{
                        const textElements = node.querySelectorAll('text');
                        textElements.forEach(textEl => {{
                            const text = textEl.textContent || '';
                            for (const stepId of stepIds) {{
                                if (text.includes(stepId)) {{
                                    matchedStepId = stepId;
                                    return;
                                }}
                            }}
                        }});
                    }}
                    
                    if (matchedStepId) {{
                        // Make node clickable
                        node.style.cursor = 'pointer';
                        node.style.transition = 'opacity 0.2s';
                        
                        // Add hover effect
                        node.addEventListener('mouseenter', function() {{
                            this.style.opacity = '0.7';
                        }});
                        node.addEventListener('mouseleave', function() {{
                            this.style.opacity = '1';
                        }});
                        
                        // Add click handler
                        node.addEventListener('click', function(e) {{
                            e.stopPropagation();
                            
                            // Scroll to JSON section
                            const anchorId = 'step-' + matchedStepId;
                            const anchor = document.getElementById(anchorId);
                            if (anchor) {{
                                // Scroll the JSON container to show the anchor
                                const jsonContainer = document.getElementById('json-container');
                                if (jsonContainer) {{
                                    const containerRect = jsonContainer.getBoundingClientRect();
                                    const anchorRect = anchor.getBoundingClientRect();
                                    const scrollTop = jsonContainer.scrollTop + anchorRect.top - containerRect.top - 50;
                                    jsonContainer.scrollTo({{
                                        top: scrollTop,
                                        behavior: 'smooth'
                                    }});
                                    
                                    // Highlight the anchor area
                                    anchor.style.backgroundColor = '#fff3cd';
                                    anchor.style.padding = '2px 4px';
                                    anchor.style.borderRadius = '3px';
                                    setTimeout(() => {{
                                        anchor.style.backgroundColor = '';
                                        anchor.style.padding = '';
                                        anchor.style.borderRadius = '';
                                    }}, 2000);
                                }} else {{
                                    anchor.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                                }}
                            }}
                        }});
                    }}
                }});
            }});
        }}
        
        // Wait for page to load and SVG to render
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', addClickHandlers);
        }} else {{
            addClickHandlers();
        }}
        
        // Retry after delays to catch dynamically loaded content
        setTimeout(addClickHandlers, 500);
        setTimeout(addClickHandlers, 1000);
    }})();
    </script>
    """
    st.markdown(javascript_code, unsafe_allow_html=True)

with col2:
    st.markdown("### 🛠️ Configuration")
    st.caption("Live JSON generated by Llama 3")
    
    # Reduce spacing after caption
    st.markdown("""
    <style>
    [data-testid="stCaption"] {
        margin-bottom: 0.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add scrollable container styles for JSON module
    st.markdown("""
    <style>
    .json-scroll-container {
        max-height: 600px;
        overflow-y: auto;
        overflow-x: auto;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 0 !important;
        background-color: #f8f9fa;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
        margin: 0 !important;
    }
    .json-scroll-container > *,
    .json-scroll-container > * > *,
    .json-scroll-container > * > * > * {
        margin: 0 !important;
        padding: 0 !important;
    }
    .json-scroll-container [data-testid="stJson"],
    .json-scroll-container [data-testid="stJson"] > *,
    .json-scroll-container [data-testid="stJson"] > * > *,
    .json-scroll-container [data-testid="stJson"] > * > * > * {
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stJson"],
    div[data-testid="stJson"] > div {
        margin: 0 !important;
        padding: 0 !important;
        background-color: transparent !important;
    }
    /* Remove any Streamlit default spacing */
    .json-scroll-container + *,
    * + .json-scroll-container {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    /* Target Streamlit's JSON widget wrapper directly */
    .json-scroll-container > div[class*="st"],
    .json-scroll-container > div[class*="element-container"],
    .json-scroll-container > div[data-testid] {
        margin: 0 !important;
        padding: 0 !important;
        background: transparent !important;
    }
    /* Remove white space from Streamlit widget containers */
    .json-scroll-container > div {
        background-color: transparent !important;
    }
    /* Ensure JSON content starts at top and remove white background */
    .json-scroll-container [data-testid="stJson"] {
        margin-top: 0 !important;
        padding-top: 0 !important;
        background-color: transparent !important;
    }
    /* Remove any white backgrounds from nested divs */
    .json-scroll-container div {
        background-color: transparent !important;
    }
    /* But keep the container's background */
    .json-scroll-container {
        background-color: #f8f9fa !important;
    }
    .json-scroll-container::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    .json-scroll-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    .json-scroll-container::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    .json-scroll-container::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Wrap JSON directly without extra containers
    st.markdown('<div class="json-scroll-container" id="json-container">', unsafe_allow_html=True)
    st.json(st.session_state.workflow_data.model_dump())
    st.markdown('</div>', unsafe_allow_html=True)