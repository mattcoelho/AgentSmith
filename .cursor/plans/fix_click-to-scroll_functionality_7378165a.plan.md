---
name: Fix Click-to-Scroll Functionality
overview: Debug and fix the click-to-scroll functionality so clicking workflow steps in the graphviz visualization properly scrolls to and highlights the corresponding step in the JSON panel.
todos:
  - id: "1"
    content: Fix graphviz node selection to target g.node elements specifically
    status: completed
  - id: "2"
    content: Simplify step ID matching logic to be more reliable
    status: completed
  - id: "3"
    content: Add console logging for debugging click-to-scroll issues
    status: completed
  - id: "4"
    content: Add text-based scrolling fallback when anchors are not found
    status: completed
---

## Current Issues

The click-to-scroll functionality isn't working. Potential problems:

1. **Graphviz Node Selection**: Currently selecting all `g` elements, but graphviz nodes are typically in `g.node` elements
2. **Anchor Insertion**: The anchor insertion logic might be failing silently or anchors aren't being found
3. **JSON Container**: The wrapper div might not be working correctly with `st.json()` output
4. **Step ID Matching**: The complex matching logic might be failing to match graphviz nodes to step IDs

## Solution

### 1. Fix Graphviz Node Selection

Graphviz typically creates nodes with class "node", so we should target `g.node` instead of all `g` elements. Also, graphviz uses the step ID directly as the node ID in the SVG.

### 2. Simplify Anchor Insertion

The current anchor insertion is complex and might fail. Use a simpler approach:

- Find the JSON text content
- Locate step IDs in the text
- Insert anchors using a more reliable DOM manipulation method
- Or use a text-based scrolling approach as fallback

### 3. Improve Debugging

Add console logging to help identify where the process is failing:

- Log when nodes are found
- Log when step IDs are matched
- Log when anchors are found/not found
- Log scroll attempts

### 4. Simplify Step ID Matching

Since graphviz uses step IDs directly in node IDs, simplify the matching:

- First try direct node ID match
- Then try with "node" prefix
- Use a simpler fallback

### 5. Add Fallback Text-Based Scrolling

If anchors aren't found, use text-based scrolling:

- Find the step ID text in JSON
- Calculate approximate scroll position
- Scroll to that position

## Files to Modify

- [app.py](app.py):
  - Fix graphviz node selection to use `g.node` (around line 667)
  - Simplify step ID matching logic (around line 669-735)
  - Simplify anchor insertion or add fallback (around line 900-990)
  - Add console logging for debugging (throughout JavaScript)
  - Improve click handler with better error handling (around line 751-850)

## Implementation Details

### Graphviz Node Selection

```javascript
// Change from:
const nodes = svg.querySelectorAll('g');

// To:
const nodes = svg.querySelectorAll('g.node');
```

### Simplified Step ID Matching

```javascript
// Graphviz uses step ID directly as node ID
const nodeId = node.getAttribute('id') || '';
// Try direct match first
if (stepIds.includes(nodeId)) {
    matchedStepId = nodeId;
}
// Then try with common graphviz prefixes
else if (nodeId.startsWith('node') && stepIds.includes(nodeId.substring(4))) {
    matchedStepId = nodeId.substring(4);
}
```

### Text-Based Scrolling Fallback

If anchor not found, find step ID text in JSON and scroll to it using text position calculation.