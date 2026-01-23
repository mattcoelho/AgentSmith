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

================================================================================
🌳 BRANCHED WORKFLOWS - DECISION TREES:
================================================================================
When the user describes conditional logic (if/then, if/else, decision points), use BRANCHES:

- Use "branches" array when a step has multiple possible next steps based on conditions
- Use "next_step_id" for linear flow or the default path when no branches match
- Each branch must have a clear "condition" description and "next_step_id"
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
1. Analyze the user's request.
2. If they are CREATING a new flow, overwrite the current state.
3. If they are MODIFYING (e.g., "add a delay", "change slack channel"), update the existing steps intelligently.
4. Output ONLY valid JSON matching the Workflow schema. No chat, no markdown, no code blocks.
5. EVERY step MUST have exactly these 4 required fields: id, app, action, details.
6. OPTIONAL: Add "next_step_id" for linear flow or "branches" for conditional logic.
7. When the user describes conditional logic (if/then, if/else, decision points), USE BRANCHES to create decision trees instead of simplifying to sequential steps.
8. NEVER use: type, config, name (in steps), or nested objects (except branches array).
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
    import html
    workflow_dict = workflow.model_dump()
    
    # Convert to formatted JSON string
    json_str = json.dumps(workflow_dict, indent=2)
    
    # Build a list of anchor positions (step_id -> line_index after opening brace)
    anchor_positions = {}
    lines = json_str.split('\n')
    in_steps_array = False
    step_brace_line = None
    
    for i, line in enumerate(lines):
        if '"steps"' in line and '[' in line:
            in_steps_array = True
            continue
        if in_steps_array and line.strip() == ']':
            in_steps_array = False
            continue
        if in_steps_array and line.strip() == '{':
            step_brace_line = i
            continue
        if in_steps_array and '"id"' in line and step_brace_line is not None:
            import re
            match = re.search(r'"id"\s*:\s*"([^"]+)"', line)
            if match:
                step_id = match.group(1)
                anchor_positions[step_id] = step_brace_line + 1
                step_brace_line = None
    
    # Build HTML with anchors inserted
    result_lines = []
    anchor_inserted = set()
    
    for i, line in enumerate(lines):
        # Check if we need to insert an anchor before this line
        for step_id, anchor_line_idx in anchor_positions.items():
            if i == anchor_line_idx and step_id not in anchor_inserted:
                indent = len(line) - len(line.lstrip()) if line.strip() else 0
                anchor_id = f"step-{step_id}"
                result_lines.append(' ' * indent + f'<a id="{anchor_id}" class="step-anchor"></a>')
                anchor_inserted.add(step_id)
        
        # HTML escape and add the line
        escaped_line = html.escape(line)
        result_lines.append(escaped_line)
    
    # Join lines and wrap in container
    json_html = '\n'.join(result_lines)
    html_content = f'<div id="json-container" style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; max-height: 600px; overflow-y: auto;"><pre style="margin: 0; white-space: pre-wrap; word-wrap: break-word; font-family: monospace;">{json_html}</pre></div>'
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
                // Find graphviz node groups - graphviz creates nodes with class "node"
                const nodes = svg.querySelectorAll('g.node');
                
                console.log('Found', nodes.length, 'graphviz nodes');
                
                nodes.forEach((node, nodeIndex) => {{
                    // Get the node's ID - graphviz uses step ID directly as node ID
                    const nodeId = node.getAttribute('id') || '';
                    console.log('Processing node', nodeIndex, 'with ID:', nodeId);
                    
                    // Try to find matching step ID with simplified matching logic
                    let matchedStepId = null;
                    
                    // Method 1: Direct match (graphviz uses step ID directly)
                    if (stepIds.includes(nodeId)) {{
                        matchedStepId = nodeId;
                        console.log('Direct match found:', matchedStepId);
                    }}
                    // Method 2: Try with "node" prefix (some graphviz versions add this)
                    else if (nodeId.startsWith('node') && stepIds.includes(nodeId.substring(4))) {{
                        matchedStepId = nodeId.substring(4);
                        console.log('Match with node prefix found:', matchedStepId);
                    }}
                    // Method 3: Try removing "node" prefix and matching
                    else if (nodeId.startsWith('node')) {{
                        const withoutPrefix = nodeId.substring(4);
                        if (stepIds.includes(withoutPrefix)) {{
                            matchedStepId = withoutPrefix;
                            console.log('Match after removing node prefix:', matchedStepId);
                        }}
                    }}
                    // Method 4: Check if node ID contains any step ID
                    else {{
                        for (const stepId of stepIds) {{
                            if (nodeId.includes(stepId) || stepId.includes(nodeId)) {{
                                matchedStepId = stepId;
                                console.log('Partial match found:', matchedStepId);
                                break;
                            }}
                        }}
                    }}
                    
                    // Method 5: Fallback - match by index if we can't match by ID
                    if (!matchedStepId && nodeIndex < stepIds.length) {{
                        matchedStepId = stepIds[nodeIndex];
                        console.log('Fallback: matched by index', nodeIndex, '->', matchedStepId);
                    }}
                    
                    if (matchedStepId) {{
                        console.log('Matched step ID:', matchedStepId, 'for node:', nodeId);
                        
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
                            console.log('Step clicked:', matchedStepId);
                            
                            // Visual feedback on clicked node
                            const originalOpacity = this.style.opacity;
                            this.style.opacity = '0.5';
                            this.style.transform = 'scale(0.95)';
                            setTimeout(() => {{
                                this.style.opacity = originalOpacity || '1';
                                this.style.transform = '';
                            }}, 200);
                            
                            // Scroll to JSON section
                            const anchorId = 'step-' + matchedStepId;
                            console.log('Looking for anchor:', anchorId);
                            const anchor = document.getElementById(anchorId);
                            const jsonContainer = document.getElementById('json-container');
                            
                            console.log('Anchor found:', !!anchor, 'Container found:', !!jsonContainer);
                            
                            if (anchor && jsonContainer) {{
                                console.log('Scrolling to anchor');
                                // Calculate scroll position
                                const containerRect = jsonContainer.getBoundingClientRect();
                                const anchorRect = anchor.getBoundingClientRect();
                                
                                // Scroll to show the anchor with some padding
                                const scrollTop = jsonContainer.scrollTop + anchorRect.top - containerRect.top - 20;
                                jsonContainer.scrollTo({{
                                    top: Math.max(0, scrollTop),
                                    behavior: 'smooth'
                                }});
                                
                                // Highlight the step's JSON section
                                // Find the JSON object containing this step
                                let stepElement = anchor.nextSibling;
                                let depth = 0;
                                let stepStart = anchor;
                                
                                // Walk through siblings to find the step object boundaries
                                const highlightElements = [anchor];
                                
                                // Find the opening brace and highlight the entire step object
                                let current = anchor.parentElement;
                                while (current && current !== jsonContainer) {{
                                    const text = current.textContent || '';
                                    if (text.includes('"id": "' + matchedStepId + '"')) {{
                                        // Found the step, highlight this element and its siblings until closing brace
                                        let sibling = current;
                                        let braceCount = 0;
                                        let foundStart = false;
                                        
                                        // Walk backwards to find opening brace
                                        while (sibling && sibling !== jsonContainer) {{
                                            const sibText = sibling.textContent || '';
                                            if (sibText.includes('{{')) {{
                                                braceCount++;
                                                foundStart = true;
                                                highlightElements.push(sibling);
                                                break;
                                            }}
                                            sibling = sibling.previousSibling;
                                        }}
                                        
                                        // Walk forwards to find closing brace
                                        sibling = current;
                                        while (sibling && sibling !== jsonContainer && braceCount > 0) {{
                                            const sibText = sibling.textContent || '';
                                            if (sibText.includes('}}')) {{
                                                braceCount--;
                                                highlightElements.push(sibling);
                                                if (braceCount === 0) break;
                                            }} else if (foundStart) {{
                                                highlightElements.push(sibling);
                                            }}
                                            sibling = sibling.nextSibling;
                                        }}
                                        break;
                                    }}
                                    current = current.parentElement;
                                }}
                                
                                // Apply highlight to all found elements
                                highlightElements.forEach(el => {{
                                    if (el && el.style) {{
                                        el.style.backgroundColor = '#fff3cd';
                                        el.style.transition = 'background-color 0.3s';
                                    }}
                                }});
                                
                                // Remove highlight after 3 seconds
                                setTimeout(() => {{
                                    highlightElements.forEach(el => {{
                                        if (el && el.style) {{
                                            el.style.backgroundColor = '';
                                        }}
                                    }});
                                }}, 3000);
                                
                            }} else if (jsonContainer) {{
                                console.log('Anchor not found, using text-based scrolling fallback');
                                // Fallback: try to find step ID text and scroll to it
                                const jsonText = jsonContainer.textContent || jsonContainer.innerText || '';
                                console.log('JSON text length:', jsonText.length);
                                
                                const escapedStepId = matchedStepId.replace(/[.*+?^${{}}()|[\\\\]\\\\]/g, '\\\\$&');
                                const stepPattern = new RegExp('"id"\\\\s*:\\\\s*"(' + escapedStepId + ')"');
                                const match = stepPattern.exec(jsonText);
                                
                                console.log('Step pattern match:', !!match);
                                
                                if (match) {{
                                    console.log('Found step ID at position:', match.index);
                                    // Find all elements in the JSON container
                                    const allElements = jsonContainer.querySelectorAll('*');
                                    let foundElement = null;
                                    
                                    // Try to find the element containing this text
                                    for (const el of allElements) {{
                                        if (el.textContent && el.textContent.includes('"id": "' + matchedStepId + '"')) {{
                                            foundElement = el;
                                            break;
                                        }}
                                    }}
                                    
                                    if (foundElement) {{
                                        console.log('Found element containing step ID, scrolling to it');
                                        foundElement.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                                        
                                        // Highlight the element
                                        foundElement.style.backgroundColor = '#fff3cd';
                                        foundElement.style.transition = 'background-color 0.3s';
                                        setTimeout(() => {{
                                            foundElement.style.backgroundColor = '';
                                        }}, 3000);
                                    }} else {{
                                        // Approximate scroll position based on text position
                                        const textBeforeMatch = jsonText.substring(0, match.index);
                                        const linesBefore = textBeforeMatch.split('\\\\n').length;
                                        const lineHeight = 20; // Approximate line height
                                        const scrollPosition = linesBefore * lineHeight;
                                        
                                        console.log('Scrolling to approximate position:', scrollPosition);
                                        jsonContainer.scrollTo({{
                                            top: Math.max(0, scrollPosition - 50),
                                            behavior: 'smooth'
                                        }});
                                    }}
                                }} else {{
                                    console.warn('Could not find step ID in JSON text');
                                }}
                            }} else {{
                                console.error('JSON container not found');
                            }}
                        }});
                    }} else {{
                        console.log('No match found for node:', nodeId);
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
    # Wrap JSON in scrollable container
    st.markdown('<div id="json-container" style="max-height: 600px; overflow-y: auto; overflow-x: auto;">', unsafe_allow_html=True)
    st.json(st.session_state.workflow_data.model_dump())
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add JavaScript to dynamically insert anchors for each step
    steps = st.session_state.workflow_data.steps
    if steps:
        step_ids_json = json.dumps([step.id for step in steps])
        anchor_script = f"""
        <script>
        (function() {{
            function insertStepAnchors() {{
                console.log('Attempting to insert step anchors...');
                const container = document.getElementById('json-container');
                if (!container) {{
                    console.log('JSON container not found, retrying...');
                    setTimeout(insertStepAnchors, 200);
                    return;
                }}
                
                console.log('JSON container found');
                
                // Find the JSON element (st.json creates a specific structure)
                const jsonElement = container.querySelector('[data-testid="stJson"], .stJson, pre, code');
                if (!jsonElement) {{
                    console.log('JSON element not found, retrying...');
                    setTimeout(insertStepAnchors, 200);
                    return;
                }}
                
                console.log('JSON element found:', jsonElement.tagName);
                
                const stepIds = {step_ids_json};
                console.log('Step IDs to process:', stepIds);
                const jsonText = jsonElement.textContent || jsonElement.innerText || '';
                console.log('JSON text length:', jsonText.length);
                
                let anchorsInserted = 0;
                
                // For each step, find its position and insert an anchor
                stepIds.forEach((stepId, index) => {{
                    const anchorId = 'step-' + stepId;
                    
                    // Check if anchor already exists
                    if (document.getElementById(anchorId)) {{
                        console.log('Anchor already exists for:', stepId);
                        return;
                    }}
                    
                    // Find the step in JSON - look for "id": "stepId"
                    const escapedStepId = stepId.replace(/[.*+?^${{}}()|[\\\\]\\\\]/g, '\\\\$&');
                    const stepPattern = new RegExp('"id"\\\\s*:\\\\s*"(' + escapedStepId + ')"');
                    const match = stepPattern.exec(jsonText);
                    
                    if (match) {{
                        console.log('Found step ID in JSON text:', stepId, 'at position:', match.index);
                        
                        // Try to find the element containing this text
                        const allElements = jsonElement.querySelectorAll('*');
                        let targetElement = null;
                        
                        // Look for element containing the step ID
                        for (const el of allElements) {{
                            if (el.textContent && el.textContent.includes('"id": "' + stepId + '"')) {{
                                // Check if this is the closest parent to the step object start
                                const text = el.textContent || '';
                                const stepIndex = text.indexOf('"id": "' + stepId + '"');
                                if (stepIndex >= 0) {{
                                    // Look backwards for opening brace
                                    const beforeStep = text.substring(0, stepIndex);
                                    const lastBraceIndex = beforeStep.lastIndexOf('{{');
                                    if (lastBraceIndex >= 0 && lastBraceIndex > beforeStep.length - 50) {{
                                        targetElement = el;
                                        break;
                                    }}
                                }}
                            }}
                        }}
                        
                        // If we found a target element, insert anchor before it
                        if (targetElement && targetElement.parentElement) {{
                            const anchor = document.createElement('span');
                            anchor.id = anchorId;
                            anchor.className = 'step-anchor';
                            anchor.style.display = 'block';
                            anchor.style.height = '1px';
                            anchor.style.width = '1px';
                            anchor.style.position = 'absolute';
                            anchor.style.visibility = 'hidden';
                            
                            targetElement.parentElement.insertBefore(anchor, targetElement);
                            anchorsInserted++;
                            console.log('Inserted anchor for step:', stepId);
                        }} else {{
                            // Fallback: insert at the beginning of the JSON element
                            const anchor = document.createElement('span');
                            anchor.id = anchorId;
                            anchor.className = 'step-anchor';
                            anchor.style.display = 'block';
                            anchor.style.height = '1px';
                            anchor.style.width = '1px';
                            anchor.style.position = 'absolute';
                            anchor.style.visibility = 'hidden';
                            
                            jsonElement.insertBefore(anchor, jsonElement.firstChild);
                            anchorsInserted++;
                            console.log('Inserted anchor at fallback position for step:', stepId);
                        }}
                    }} else {{
                        console.warn('Could not find step ID in JSON:', stepId);
                    }}
                }});
                
                console.log('Total anchors inserted:', anchorsInserted, 'out of', stepIds.length);
            }}
            
            // Wait for JSON to render
            if (document.readyState === 'loading') {{
                document.addEventListener('DOMContentLoaded', insertStepAnchors);
            }} else {{
                insertStepAnchors();
            }}
            setTimeout(insertStepAnchors, 500);
            setTimeout(insertStepAnchors, 1000);
            setTimeout(insertStepAnchors, 2000);
        }})();
        </script>
        """
        st.markdown(anchor_script, unsafe_allow_html=True)