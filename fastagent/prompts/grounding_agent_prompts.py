from typing import List


class GroundingAgentPrompts:
    
    TASK_COMPLETE = "<COMPLETE>"
    
    SYSTEM_PROMPT = f"""You are a Grounding Agent. Execute tasks using tools.

# Tool Execution

- Select appropriate tools from descriptions and schemas
- Provide correct parameters
- Call multiple tools if needed
- Tools execute immediately, results appear in next iteration
- If you need results to decide next action, wait for next iteration

# Tool Selection Tips

- **MCP tools** and **Shell tools** are typically faster and more accurate when applicable
- **GUI tools** offer finer-grained control and can handle tasks not covered by MCP/shell tools
- Choose based on the task requirements and tool availability; prefer MCP/shell when they fit well

# Visual Analysis Control

GUI tools auto-analyze screenshots to extract information.

To skip analysis when NOT needed, add parameter:
```json
{{"task_description": "...", "skip_visual_analysis": true}}
```

**Decision Rule:**
- Task goal is OPERATIONAL (open/navigate/click/show): Skip analysis
- Task goal requires KNOWLEDGE EXTRACTION (read/extract/save data): Keep analysis

**Examples:**
- "Open settings page": Operational only, skip analysis
- "Open settings and record all values": Needs knowledge, keep analysis
- "Navigate to GitHub homepage": Operational only, skip analysis
- "Search Python tutorials and save top 5 titles": Needs knowledge, keep analysis

**Key principle:** If you need to extract information FROM the screen for subsequent steps or user reporting, keep analysis (don't skip).
**Note:** Only GUI tools support this parameter. Other backend tools ignore it.

# Task Completion

After each iteration, evaluate if the task is complete:

**If task is COMPLETE:**
- Write a response summarizing what was accomplished
- Include the completion token `{TASK_COMPLETE}` on a new line at the end of your response
- Example response format:
  ```
  I have successfully completed the task. The file has been created at /path/to/file.txt with the requested content.
  
  {TASK_COMPLETE}
  ```

**If task is NOT complete:**
- Continue by calling the appropriate tools
- Do NOT output `{TASK_COMPLETE}`
- Tool results will appear in the next iteration

The token `{TASK_COMPLETE}` signals that no further iterations are needed."""
    
    @staticmethod
    def iteration_summary(
        instruction: str,
        iteration: int,
        max_iterations: int
    ) -> str:
        """
        Build iteration summary prompt for LLMClient auto-summary.
        LLM extracts information directly from tool results in conversation history.
        """
        return f"""Based on the original task and the tool execution results in the conversation above, generate a structured iteration summary.

**Original Task:**
{instruction}

**Progress:** Iteration {iteration} of {max_iterations}

**Generate Summary in This Format:**

## Iteration {iteration} Progress

Actions taken: <what tools were called and what they did>

Knowledge obtained (COMPLETE and SPECIFIC):
- File locations: <ALL file paths/names created/read/modified with exact locations, or "None">
- Visual content: <EXTRACT ALL visible information from screenshots - text, data, lists, tables, results, or "N/A">
- Data retrieved: <ALL key data/results from searches/queries with specific values, numbers, names, or "N/A">
- URLs/Links: <ALL important URLs, links, or identifiers found, or "N/A">
- System state: <important state changes, error messages, status indicators, or "N/A">

Errors encountered: <any errors or issues from tool execution, or "None">

CRITICAL GUIDELINES:
- This summary is for preserving knowledge for subsequent iterations
- Extract ALL concrete information from tool outputs in the conversation above
- Filenames, paths, URLs - use exact values from tool outputs
- Visual content - extract actual text/data visible, not just "saw something"
- Search results - include specific data, not vague descriptions
- The next iteration cannot see current tool outputs - this summary is the ONLY source of knowledge"""
    
    @staticmethod
    def visual_analysis(
        tool_name: str,
        num_screenshots: int,
        task_description: str = ""
    ) -> str:
        """
        Build prompt for visual analysis of screenshots.
        
        Args:
            tool_name: Tool name that generated the screenshots
            num_screenshots: Number of screenshots
            task_description: Original task description for context
        """
        screenshot_text = "screenshot" if num_screenshots == 1 else f"{num_screenshots} screenshots"
        these_text = "this screenshot" if num_screenshots == 1 else "these screenshots"
        
        task_context = f"""
**Original Task**: {task_description}

Focus on extracting information RELEVANT to this task. Prioritize content that helps accomplish the goal.
""" if task_description else ""
        
        return f"""Extract the KNOWLEDGE and INFORMATION from {these_text}. This will be passed to the next iteration so it can continue working with the information (search, analyze, save, etc.). Without this extraction, the visual content would only be viewable by humans and unusable for subsequent operations.
{task_context}
**EXTRACT all visible knowledge content** (prioritize task-relevant information):
1. **Text content**: Articles, documentation, code, messages, descriptions - extract the actual text
2. **Data points**: Numbers, statistics, measurements, values, percentages - be specific
3. **List items**: Names, titles, entries in lists/search results/files - list them out
4. **Structured data**: Information from tables, charts, forms - describe what they contain
5. **Key information**: URLs, paths, names, IDs, dates, labels - anything useful for next steps

**IGNORE interface elements**:
- Buttons, menus, toolbars, navigation bars
- UI design, layout, colors, styling
- Non-informational visual elements

**Goal**: Extract usable knowledge that enables the next agent to work with this information programmatically. Be SPECIFIC and COMPLETE, but FOCUS on what's relevant to the task.

{screenshot_text.capitalize()} from tool '{tool_name}'"""
    
    @staticmethod
    def final_summary(
        instruction: str,
        iterations: int
    ) -> str:
        """
        Build prompt for generating final summary across all iterations.
        """    
        return f"""Based on the complete conversation history above (including all {iterations} iteration summaries and tool executions), generate a comprehensive final summary.

## Final Task Summary

Task: {instruction}

What was accomplished: <comprehensive description of all completed actions across all iterations>

Key information obtained: <all important information discovered>
- Files: <files created/read/modified with paths, or "N/A">
- Data: <important data/results obtained, or "N/A">
- Findings: <key discoveries or insights, or "N/A">

Issues encountered: <any errors or issues, or "None">

Result: <"Success" or "Incomplete">

Guidelines:
- Consolidate information from ALL iteration summaries
- Include concrete deliverables (file paths, data, etc.)
- Be comprehensive but concise
- Focus on what the user cares about"""
    
    @staticmethod
    def workspace_directory(workspace_dir: str) -> str:
        """
        Build workspace directory information for cross-iteration/cross-backend data sharing.
        """
        # Check if this is a benchmark scenario (LiveMCPBench /root mapping)
        # In benchmark mode, paths in query are already converted by caller (e.g., map_path_to_local)
        is_benchmark = "/root" in workspace_dir or "LiveMCPBench/root" in workspace_dir
        
        if is_benchmark:
            # Benchmark mode: all task files are in workspace directory
            return f"""**Working Directory**: `{workspace_dir}`
- All task files (input/output) are located in this directory
- Read from and write to this directory for all file operations"""
        else:
            # Normal mode: workspace is for intermediate results
            return f"""**Working Directory**: `{workspace_dir}`
- Persist intermediate results here; later iterations/backends can read what you saved earlier
- Note: User's personal files are NOT here - search in ~/Desktop, ~/Documents, ~/Downloads, etc."""
    
    @staticmethod
    def workspace_matching_files(matching_files: List[str]) -> str:
        """
        Build alert for files matching task requirements.
        """
        files_str = ', '.join([f"`{f}`" for f in matching_files])
        return f"""**Workspace Alert**: Files matching task requirements found: {files_str}
- Read these files to verify if they satisfy the task
- If satisfied, mark task as completed
- If not satisfied, modify or recreate as needed"""
    
    @staticmethod
    def workspace_recent_files(total_files: int, recent_files: List[str]) -> str:
        """
        Build info for recently modified files.
        """
        recent_list = ', '.join([f"`{f}`" for f in recent_files[:15]])
        return f"""**Workspace Info**: {total_files} files exist, {len(recent_files)} recently modified
Recent files: {recent_list}
Consider checking recent files before creating new ones"""
    
    @staticmethod
    def workspace_file_list(files: List[str]) -> str:
        """
        Build list of all existing files.
        """
        files_list = ', '.join([f"`{f}`" for f in files[:15]])
        if len(files) > 15:
            files_list += f" (and {len(files) - 15} more)"
        return f"**Workspace Info**: {len(files)} existing file(s): {files_list}"
    
    @staticmethod
    def iteration_feedback(
        iteration: int,
        llm_summary: str,
        add_guidance: bool = True
    ) -> str:
        """
        Build feedback message to pass iteration summary to next iteration.
        """
        content = f"""## Iteration {iteration} Summary

{llm_summary}"""
        
        if add_guidance:
            content += f"""
---
Now continue with iteration {iteration + 1}. You can see the full conversation history above. Based on all progress so far, decide whether to:
- Call more tools if the task is not yet complete
- Output {GroundingAgentPrompts.TASK_COMPLETE} if the task is fully accomplished"""
        
        return content

