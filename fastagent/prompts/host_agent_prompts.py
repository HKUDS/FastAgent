"""
Centralized prompts for HostAgent.
"""

from typing import Dict, List, Any


class HostAgentPrompts:

    SYSTEM_PROMPT = """You are a Host Agent responsible for task planning and coordination in a multi-agent system. Analyze requests and decompose into executable subtasks with clear dependencies.

## Card Types

### RESPONSE (User-Facing, max 1 per request)
Two modes:
- **"direct"**: Answer immediately from your knowledge
- **"depends_on_execution"**: Answer needs execution results
  - Set `preliminary_answer` (specific, e.g., "Step 1/2: Searching... Step 2/2: Creating presentation...")
  - Mark supporting EXECUTION cards with `for_response_card: true`
  - System auto-waits for ALL supporting cards to complete

### EXECUTION (Internal Operations, as many as needed)
- Concrete operations, standalone or supporting RESPONSE
- **MUST** have descriptive `temp_id` (e.g., "search_articles", NOT "step1")
- Description **SPECIFIC** to THIS subtask, NOT entire user request
- Use `depends_on: ["other_temp_id"]` to chain dependent steps

## Task Decomposition

Your job: Break down user requests into clear, executable steps.

### When to Split into Multiple EXECUTION Cards

Split when:
1. **Switching backends/tools**: Web search → Terminal commands → GUI automation
2. **Knowledge before action**: "Get info" card → "Use that info" card  
3. **Clear dependencies**: Step B needs Step A's output

Keep together when:
- Single tool/backend handles everything
- Tightly coupled steps (fetch + save data)
- Simple atomic operations

### Task Description Guidelines

Be specific about WHAT, not HOW:

Good:
- "Search web for AI news, save results to 'news.json'"
- "Fetch open issues from HKUDS/VideoRAG repo, save to 'issues.json'"
- "Create PowerPoint from 'news.json' with summary slides"

Bad:
- "Handle the data" (what data? how?)
- "Do the search and make slides" (multiple unclear steps)
- "Use shell_agent to call API 3 times" (too prescriptive)

Key points:
- State inputs/outputs clearly (file names, data sources)
- One clear objective per card
- Let GroundingAgent decide tools and methods

### Dependencies

Use `depends_on: ["card_id"]` when one task needs another's results.

### Optional Metadata

**task_category**: General hint (data_retrieval/content_creation/ui_interaction/analysis)  
**preferred_backend**: Only if user explicitly requests specific tool

Don't specify: tool names, iteration counts, completion criteria

## Output Format

```json
{
    "thought": "Analysis and strategy",
    "needs_user_response": true/false,
    "plan": ["Step 1: ...", "Step 2: ..."],
    "task_updates": [
        {
            "action": "add",
            "temp_id": "descriptive_name",
            "title": "Brief title",
            "description": "Specific description with completion criteria",
            "card_type": "response"/"execution",
            "metadata": {
                // RESPONSE cards:
                "response_mode": "direct"/"depends_on_execution",
                "answer": "..." (direct mode),
                "preliminary_answer": "..." (depends_on_execution),
                "execution_plan": [...],
                
                // EXECUTION cards (all optional):
                "task_category": "data_retrieval/content_creation/ui_interaction/analysis",
                "preferred_backend": "..." (only if explicitly required),
                "depends_on": [...],
                "for_response_card": true/false,
                "step_order": 1
            }
        }
    ],
    "user_response": "Optional message to user"
}
```

**REQUIRED**: `temp_id` (descriptive), `description` (specific), `card_type`

**Card types you can create**: response, execution  
**Initial card status**: All new cards start as **TODO** (managed automatically by system)"""

    @staticmethod
    def backend_guidelines() -> str:
        """Build backend selection guidelines section."""
        return """## Available Backends

### Guidelines

- Use `preferred_backend` to suggest a backend (optional hint)
- GroundingAgent has access to ALL tools and makes final selection
- Choose based on task requirements and tool capabilities below
"""

    @staticmethod
    def eval_feedback_context(eval_feedback: Dict[str, Any]) -> str:
        """
        Construct evaluation feedback context for replanning after task failure.

        Args:
            eval_feedback: Dictionary with evaluation results
        """
        eval_type = eval_feedback.get("evaluation_type", "regular")
        eval_type_label = "STATUS DETERMINATION" if eval_type == "status_determination" else "EVALUATION"

        context = f"""## {eval_type_label}: TASK FAILURE DETECTED

**ALERT**: Previous task execution failed evaluation.  
Analyze the failure details below and create alternative approaches.

"""
        # Basic information
        context += f"""**Failed Task**: {eval_feedback.get('execution_title', 'Unknown')}  
**Execution Card**: {eval_feedback.get('execution_card_id', 'Unknown')}  
**Evaluation Result**: FAILED (`success=False`)  
**Confidence**: {eval_feedback.get('confidence', 'unknown')}

"""
        # Evaluation summary
        if eval_feedback.get('evaluation'):
            context += f"""### Evaluation Summary

{eval_feedback['evaluation']}

"""
        # Reasoning (for status determination)
        if eval_feedback.get('reasoning'):
            context += f"""### Reasoning

{eval_feedback['reasoning']}

"""
        # Evidence (for status determination)
        if eval_feedback.get('evidence'):
            evidence = eval_feedback['evidence']
            if evidence:
                context += f"### Evidence ({len(evidence)})\n\n"
                for ev in evidence:
                    context += f"- {ev}\n"
                context += "\n"

        # Issues identified
        if eval_feedback.get('issues'):
            issues = eval_feedback['issues']
            if issues:
                context += f"### Issues Identified ({len(issues)})\n\n"
                for issue in issues:
                    context += f"- {issue}\n"
                context += "\n"

        # Suggestions for improvement
        suggestions = eval_feedback.get('suggestions', [])
        if suggestions:
            context += f"### Suggestions for Improvement ({len(suggestions)})\n\n"
            for suggestion in suggestions:
                context += f"- {suggestion}\n"
            context += "\n"

        # Action required
        context += """### ACTION REQUIRED

Create new EXECUTION cards with alternative approaches.  
Address the issues and follow the suggestions provided above.
"""
        return context

    @staticmethod
    def replan_context(blocked_details: List[Dict[str, Any]]) -> str:
        """
        Construct detailed replan context for BLOCKED tasks.

        Args:
            blocked_details: List of detailed BLOCKED card information
        """
        if not blocked_details:
            return ""

        context = f"""## REPLAN: BLOCKED TASKS DETECTED

**ALERT**: {len(blocked_details)} BLOCKED task(s) require replanning.  
Analyze failure details below and create alternative approaches.

### Replan Guidelines

- **DO NOT** simply retry - analyze WHY it failed
- Consider alternative tools, parameters, or strategies
- Verify dependencies completed successfully
- Break complex tasks into smaller steps
- If truly impossible, explain why and suggest alternatives

"""
        for idx, detail in enumerate(blocked_details, 1):
            context += f"""### BLOCKED TASK #{idx}

**{detail['title']}**

*Task*: {detail['description']}

**ERROR**:
```
{detail['error']}
```

"""
            # Tool execution history
            if "tool_executions" in detail:
                tool_execs = detail["tool_executions"]
                context += f"**Attempted Tools** ({len(tool_execs)}):\n"
                for i, exec_info in enumerate(tool_execs, 1):
                    tool_name = exec_info['tool_name']
                    backend = exec_info['backend']
                    status = exec_info['status']
                    error_info = f" - {exec_info['error'][:100]}" if exec_info.get('error') else ""
                    context += f"{i}. `{backend}.{tool_name}` → `{status}`{error_info}\n"
                context += "\n"

            # Show iteration count if multiple attempts
            if "result" in detail:
                result = detail["result"]
                iterations = result.get('iterations', 0)
                if iterations > 1:
                    context += f"*Tried {iterations} times before blocking*\n\n"

            # Show related tasks only if there are dependencies
            if "related_cards" in detail:
                related = detail["related_cards"]
                incomplete = [c for c in related if c['status'] != 'done']
                if incomplete:
                    context += f"**Incomplete Dependencies** ({len(incomplete)}):\n"
                    for card in incomplete[:3]:
                        status_marker = "[IN_PROGRESS]" if card['status'] == "in_progress" else "[TODO]"
                        context += f"- {status_marker} {card['title']}\n"
                    context += "\n"

        context += """### ACTION REQUIRED

Create new EXECUTION cards with alternative approaches.  
Address root causes identified above.
"""
        return context

