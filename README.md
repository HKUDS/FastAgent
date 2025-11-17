<div align="center">

<picture>
    <img src="assets/FastAgent_logo.png" width="18%" style="border: none; box-shadow: none;" alt="FastAgent Logo">
</picture>

# FastAgent: Simple, Fast, and Strong LLM Agents

[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Windows-99C9BF.svg)](https://github.com/HKUDS/FastAgent/)
[![Python](https://img.shields.io/badge/python-3.10+-FCE7D6.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-C1E5F5.svg)](https://opensource.org/licenses/MIT/)
[![Feishu](https://img.shields.io/badge/Feishu-Group-E9DBFC?style=flat&logo=wechat&logoColor=white)](./COMMUNICATION.md) 
[![WeChat](https://img.shields.io/badge/WeChat-Group-C5EAB4?style=flat&logo=wechat&logoColor=white)](./COMMUNICATION.md)

</div>

## 🎯 FastAgent's Mission

FastAgent is designed to tackle complex tasks that require both **DeepResearch** and **Computer Use** capabilities. While DeepResearch excels at web search, knowledge summarization, and reasoning, and Computer Use focuses on operating complex software applications, many real-world tasks demand both capabilities — such as:

📊 **Business Intelligence Reports**:
- Research industry trends and competitor insights across multiple data sources.
- Generate automated PowerPoint reports and dynamic dashboards with real-time analytics and visual insights.
 
📅 **Event Planning & Management**:
- Research and compare venues and vendors based on cost and requirements.
- Create Excel budgets with cost tracking and manage project schedules.

🛒 **Smart Shopping & Price Optimization**:
- Compare products, reviews, and prices across e-commerce platforms.
- Automate shopping, apply discounts, and track delivery schedules.

FastAgent bridges this gap by seamlessly integrating intelligent research capabilities with sophisticated computer operation, enabling users to complete end-to-end workflows that span from information gathering and analysis to practical software manipulation—all within a **Unified**, **Simple**, and **Fast** framework.

---

## 💡 Current Challenges in Multi-Agent Systems

Current agent frameworks face significant challenges when tackling complex, multi-step, real-world tasks:

#### ⚡ Performance Bottlenecks:
- **Slow & Unreliable GUI Operations**. Complex tasks that humans complete in dozens of steps require hundreds of observe-decide-execute cycles. GUI grounding is particularly slow, error-prone, and frequently freezes when operating real-world interfaces.
- **Limited Task Scope**. Current approaches constrain agents to narrow, predefined use cases rather than generalizing across diverse scenarios.

#### ❌ High Failure Rates: 
- **Error Accumulation**. Multi-step workflows accumulate cascading errors across execution stages.
- **Fragile Cross-Application Transitions**. Transitions between heterogeneous software and data sources often cause complete task breakdowns.

#### 🧩 Complex Context Management:
- **Overwhelming Multi-Context Load**. Multi-faceted tasks generate overwhelming knowledge and tool contexts beyond current systems' capacity.
- **Lack of Unified Processing**. Systems cannot handle heterogeneous contexts from DeepResearch and Computer Use paradigms, or dynamically integrate diverse user-provided tools, severely limiting extensibility.
  
#### 🔒 Limited Orchestration & Adaptability: 
- **Lack of Unified Coordination**. Current frameworks lack generalized coordination mechanisms and cannot dynamically adjust to varying task requirements. Each workflow type demands different coordination patterns.
- **Manual Workflow Design**. Existing systems require complex, task-specific workflow engineering for each scenario, making generalized multi-agent orchestration extremely difficult.

---

## 🚀 FastAgent's Key Innovations

FastAgent addresses these critical challenges through three **Efficient** and **Effective** solutions focusing on **Memory Mechanism**, **Tool Integration**, and **Multi-Agent Coordination**:

#### 🧠 Advanced Memory & Context Management:
*Solving Complex Context Handling Challenges*
- **Adaptive Multi-Tier Memory**: Maintains step-, agent-, task- and response-level stores, revealing only the granularity each reasoning hop needs while guarding against knowledge dilution across sprawling workflows
- **Intelligent Context Switching**: Seamlessly bridges heterogeneous contexts from DeepResearch and Computer Use paradigms, auto-classifying outputs so agents pivot between data mining and GUI actions without losing critical breadcrumbs
- **Smart Compression with Dynamic Budgeting**: LLM-powered summarizer adaptively compresses based on content purpose: keeps structured data and key decisions for planning tasks, preserves execution details for validation, prunes repetitive operations. Triggers precisely at token thresholds so massive multi-facet contexts never clog the framework
- **Incremental Cross-Task Knowledge**: Promotes vector embeddings and structured findings to a shared, ever-growing pool updated on every tool call, so future tasks query prior insights instantly and slash redundant computation

#### 🔧 Effortless Tool Integration with Smart Orchestration:
*Eliminating Integration Difficulties & Performance Bottlenecks*
- **Smart Tool RAG System**: Precisely retrieves relevant tools from hundreds of available options across all backends (Shell, GUI, MCP, Web), enabling efficient tool selection and reducing context overhead
- **Unified Backend Architecture**: Provides consistent interface for Shell, GUI, MCP, and Web. Dynamically integrates diverse user-provided tools without manual adaptation, solving extensibility challenges through generalized tool abstraction and lifecycle management
- **Zero-Config MCP Support**: Plug-and-play MCP server integration without complex setup. Just declare servers in config and FastAgent handles connections, protocol negotiation, and session pooling

#### 🎯 Dynamic Multi-Agent Coordination
*Overcoming Limited Orchestration & High Failure Rates*
- **Event-Driven Kanban Architecture**: Provides generalized coordination mechanisms with rule-based workflow routing that dynamically adjusts to varying task requirements. Enables seamless agent communication through shared task states, eliminating manual workflow engineering for every use case
- **Selective Quality Assurance**: Dedicated EvalAgent activates only when needed (e.g., critical operations, error-prone backends) rather than every step, preventing error accumulation while maintaining efficiency. Ensures reliable transitions between software applications and data sources

---

## 📋 Table of Contents

- [📊 System Overview](#-system-overview)
- [🎯 Quick Start](#-quick-start)
- [🏗️ Code Structure](#️-code-structure)
- [🔧 Advanced Usage](#-advanced-usage)
- [🔗 Related Projects](#-related-projects)

---

## 📊 System Overview

<picture align="center">
    <img src="assets/FastAgent_framework.png" alt="FastAgent Framework" width="1000" />
</picture>

FastAgent employs a **simple and fast multi-agent event-driven architecture** that seamlessly coordinates research and execution capabilities through five core components:

### 🏗️ Core Architecture

#### 1. 🤝 **Multi-Agent Coordination**
*Dynamic orchestration with intelligent task management*

**Specialized Agents:**
- **HostAgent**: High-level planning and task decomposition with dependency tracking
- **GroundingAgent**: Cross-backend execution (Shell, GUI, MCP, Web) with smart tool selection
- **EvalAgent**: Continuous quality assurance and automatic replanning

**Event-Driven Workflow:**
- **State Management**: TODO → IN_PROGRESS → DONE/BLOCKED lifecycle
- **Dependency Tracking**: Automatic execution ordering based on task dependencies
- **Rule-Based Routing**: Dynamic agent triggering based on task type and state
- **Failure Recovery**: Automatic replanning when tasks fail validation

#### 2. 🔧 **Unified Tool Integration**
*Seamless plug-and-play ecosystem with 100+ tools*

- **Multi-Backend Architecture**: Unified interface across Shell, GUI, MCP, and Web
- **Smart Tool RAG**: Semantic search retrieves relevant tools from hundreds of options
- **Zero-Config MCP**: Add servers to config file, FastAgent handles discovery and routing
- **Session Pooling**: Efficient resource management and connection reuse

#### 3. 🧠 **Advanced Memory & Context**
*Intelligent compression and cross-phase awareness*

- **Multi-Level Storage**: Task context, agent history, and processed results
- **Smart Compression**: Automatically manages overwhelming multi-faceted contexts
- **Context Switching**: Seamless transitions between research and operation phases
- **Historical Awareness**: Agents access and learn from previous execution results

#### 4. 🔍 **Intelligent Search**
*Multi-source information retrieval and aggregation*

- **Web Integration**: Built-in search and browsing capabilities
- **Tool Discovery**: Semantic matching across all available tools and MCP servers
- **Knowledge Retrieval**: Context-aware search through execution history
- **Source Fusion**: Combines web, local files, APIs, and databases

#### 5. 🛡️ **Security & Control**
*Enterprise-grade safety with comprehensive audit*

- **Command Filtering**: Whitelist/blacklist with pattern matching
- **User Approval**: Optional confirmation for sensitive operations
- **Access Control**: Granular permissions per backend and operation
- **Complete Audit**: Full logging with trajectory recording and video capture

---

## 🎯 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/HKUDS/FastAgent.git
cd FastAgent

# Create and activate conda environment
conda create -n fastagent python=3.12 -y
conda activate fastagent

# Install dependencies
pip install -r requirements.txt
```

> [!NOTE]
> Create a `.env` file and add your API keys (refer to `fastagent/.env.example`).

### 2. Launch FastAgent

#### Start Local Server (Required for Computer Control)

The **local server** is a lightweight Flask service that enables FastAgent to interact with your computer (GUI automation, Python/Bash execution, file operations, screen capture, etc.).

> [!NOTE]
> See [`fastagent/local_server/README.md`](fastagent/local_server/README.md) for complete API documentation and advanced configuration.

> [!IMPORTANT]
> **Platform-specific setup required**: Different operating systems need different dependencies for desktop control. Please install the required dependencies for your OS before starting the local server:

<details>
<summary><b>macOS Setup</b></summary>

```bash
# Install macOS-specific dependencies
pip install pyobjc-core pyobjc-framework-cocoa pyobjc-framework-quartz atomacos
```

**Permissions Required**: macOS will automatically prompt for permissions when you first run the local server. Grant the following:
- **Accessibility** (for GUI control)
- **Screen Recording** (for screenshots and video capture)

> If prompts don't appear, manually grant permissions in System Settings → Privacy & Security.
</details>

<details>
<summary><b>Linux Setup</b></summary>

```bash
# Install Linux-specific dependencies
pip install python-xlib pyatspi numpy

# Install system packages
sudo apt install at-spi2-core python3-tk scrot
```
</details>

<details>
<summary><b>Windows Setup</b></summary>

```bash
# Install Windows-specific dependencies
pip install pywinauto pywin32 PyGetWindow
```
</details>

After installing the platform-specific dependencies, start the local server:

```bash
python -m fastagent.local_server.main
```

> [!TIP]
> **Local server is required** for GUI automation and Python/Bash execution. Without it, only MCP servers and web research capabilities are available.

#### Start FastAgent

Then, launch the FastAgent main process in another terminal:
```bash
python -m fastagent
```

### 3. Execute Any Task You Want 🤗

Simply type your task in natural language. FastAgent seamlessly combines **DeepResearch** and **Computer Use** to handle complex end-to-end workflows:

> [!TIP]
> **MCP Server Configuration**: For tasks requiring specific tools, add relevant MCP servers to `fastagent/config/config_mcp.json`. Unsure which servers to add? Simply add all potentially useful ones, FastAgent's Smart Tool RAG will automatically select the appropriate tools for your task. See [MCP Configuration](#mcp-configuration) for details.

**Simple Example:**
```bash
>>> Please help me search HKUDS on Google.
```

**Complex Example - AI Coding Assistants Competitive Analysis**
```bash
>>> Create a competitive analysis report for AI coding assistants.

Research these 3 products: GitHub Copilot, Cursor, Claude Code.

For each product, find and verify:
1. Supported programming languages and IDE integrations
2. Pricing tiers (in USD) and token limits
3. Key differentiating features (refactoring, test generation, chat capabilities)
4. Security and privacy guarantees

Then create an Excel workbook named "AI_Coding_Assistants_Analysis.xlsx" with:
- Sheet 1: Feature comparison matrix (products as rows, capabilities as columns)
- Sheet 2: Pricing breakdown with cost analysis

Also create a "executive_summary.md" file highlighting the best choice and why.
```

---

## 🏗️ Code Structure

### 📖 Quick Overview

> **Legend**: ⚡ Core modules | 🔧 Supporting modules

```
FastAgent/
├── fastagent/
│   ├── __init__.py                       # Package exports
│   ├── __main__.py                       # CLI entry point
│   ├── fastagent.py                      # Main FastAgent class
│   │
│   ├── ⚡ agents/                         # Multi-Agent System 
│   ├── ⚡ workflow/                       # Event-Driven Workflow 
│   ├── ⚡ kanban/                         # Task Management System 
│   ├── ⚡ grounding/                      # Unified Backend System
│   │   ├── core/                         # Core abstractions
│   │   └── backends/                     # Backend implementations
│   │       ├── shell/                    # Shell command execution
│   │       ├── gui/                      # Anthropic Computer Use
│   │       ├── mcp/                      # Model Context Protocol
│   │       └── web/                      # Web search & browsing
│   │
│   ├── ⚡ memory/                         # Memory & Storage
│   ├── 🔧 llm/                           # LLM Integration
│   ├── 🔧 config/                        # Configuration System 
│   ├── 🔧 local_server/                  # GUI Backend Server 
│   ├── 🔧 recording/                     # Execution Recording
│   ├── 🔧 platform/                      # Platform Integration
│   └── 🔧 utils/                         # Utilities
│
├── .fastagent/                           # Runtime cache
│   └── embedding_cache/                  # Tool embeddings for Smart Tool RAG
│
├── logs/                                 # Execution logs and recordings
│   ├── __main__/                         # Application logs
│   └── recordings/                       # Complete execution audit trail
│
├── requirements.txt                      # Python dependencies
└── README.md                            
```

---

### 📂 Detailed Module Structure

<details open>
<summary><b>⚡ agents/</b> - Multi-Agent System (Core Architecture)</summary>

```
agents/
├── __init__.py
├── base.py                         # Base agent class with common functionality
├── coordinator.py                  # Agent coordination & resource management
├── host_agent.py                   # High-level planning and task decomposition
├── grounding_agent.py              # Cross-backend task execution
├── eval_agent.py                   # Automatic evaluation and quality assurance
├── content_processor.py            # Intelligent content processing
└── agent_data_manager.py           # Agent data storage and retrieval
```

**Key Responsibilities**: Task planning, execution, evaluation, and inter-agent coordination.

</details>

<details open>
<summary><b>⚡ workflow/</b> - Event-Driven Workflow Engine</summary>

```
workflow/
├── __init__.py
├── engine.py                       # Event-driven workflow orchestration
├── rules.py                        # Workflow rule definitions and routing
└── context_manager.py              # Cross-execution context management
```

**Key Responsibilities**: Event processing, rule-based routing, state transitions, and context preservation.

</details>

<details open>
<summary><b>⚡ kanban/</b> - Task Management System</summary>

```
kanban/
├── __init__.py
├── kanban.py                       # Task board and state management
└── enums.py                        # Task types, states, and events
```

**Key Responsibilities**: Task lifecycle management, dependency tracking, and state transitions (TODO → IN_PROGRESS → DONE/BLOCKED).

</details>

<details open>
<summary><b>⚡ grounding/</b> - Unified Backend System (Core Integration Layer)</summary>

#### Core Abstractions

```
grounding/core/
├── grounding_client.py             # Unified interface across all backends
├── provider.py                     # Abstract provider base class
├── session.py                      # Session lifecycle management
├── search_tools.py                 # Smart Tool RAG for semantic search
├── exceptions.py                   # Custom exception definitions
├── types.py                        # Shared type definitions
│
├── tool/                           # Tool abstraction layer
│   ├── base.py                     # Tool base class
│   ├── local_tool.py               # Local tool implementation
│   └── remote_tool.py              # Remote tool implementation
│
├── security/                       # Security & sandboxing 🔧
│   ├── policies.py                 # Security policy enforcement
│   ├── sandbox.py                  # Sandbox abstraction
│   └── e2b_sandbox.py              # E2B sandbox integration
│
├── system/                         # System-level provider
│   ├── provider.py
│   └── tool.py
│
└── transport/                      # Transport layer abstractions 🔧
    ├── connectors/
    │   ├── base.py
    │   └── aiohttp_connector.py
    └── task_managers/
        ├── base.py
        ├── async_ctx.py
        ├── aiohttp_connection_manager.py
        └── placeholder.py
```

#### Backend Implementations

<details>
<summary><b>Shell Backend</b> - Command execution via local server</summary>

```
backends/shell/
├── provider.py                     # Shell provider implementation
├── session.py                      # Shell session management
└── transport/
    └── connector.py                # HTTP connector to local server
```

</details>

<details>
<summary><b>GUI Backend</b> - Anthropic Computer Use integration</summary>

```
backends/gui/
├── provider.py                     # GUI provider implementation
├── session.py                      # GUI session management
├── tool.py                         # GUI-specific tools
├── anthropic_client.py             # Anthropic API client wrapper
├── anthropic_utils.py              # Utility functions
├── config.py                       # GUI configuration
└── transport/
    ├── connector.py                # Computer Use API connector
    └── actions.py                  # Action execution logic
```

</details>

<details>
<summary><b>MCP Backend</b> - Model Context Protocol servers</summary>

```
backends/mcp/
├── provider.py                     # MCP provider implementation
├── session.py                      # MCP session management
├── client.py                       # MCP client
├── config.py                       # MCP configuration loader
├── installer.py                    # MCP server installer
├── tool_converter.py               # Convert MCP tools to unified format
└── transport/
    ├── connectors/                 # Multiple transport types
    │   ├── base.py
    │   ├── stdio.py                # Standard I/O connector
    │   ├── http.py                 # HTTP connector
    │   ├── websocket.py            # WebSocket connector
    │   ├── sandbox.py              # Sandboxed connector
    │   └── utils.py
    └── task_managers/              # Protocol-specific managers
        ├── stdio.py
        ├── sse.py
        ├── streamable_http.py
        └── websocket.py
```

</details>

<details>
<summary><b>Web Backend</b> - Search and browsing</summary>

```
backends/web/
├── provider.py                     # Web provider implementation
└── session.py                      # Web session management
```

</details>

**Key Responsibilities**: Unified tool abstraction, backend routing, session pooling, and Smart Tool RAG.

</details>

<details open>
<summary><b>⚡ memory/</b> - Memory & Storage</summary>

```
memory/
├── __init__.py
├── storage_manager.py              # Unified storage interface
├── memory.py                       # Memory management and context
└── summarizer.py                   # Content compression and summarization
```

**Key Responsibilities**: Multi-level storage, context compression, and historical awareness.

</details>

<details>
<summary><b>🔧 llm/</b> - LLM Integration</summary>

```
llm/
├── __init__.py
└── client.py                       # LiteLLM wrapper with retry logic
```

</details>

<details>
<summary><b>🔧 config/</b> - Configuration System</summary>

```
config/
├── __init__.py
├── loader.py                       # Configuration file loader
├── constants.py                    # System constants
├── grounding.py                    # Grounding configuration dataclasses
├── utils.py                        # Configuration utilities
│
├── config_grounding.json           # Backend-specific settings
├── config_agents.json              # Agent configurations
├── config_workflow.json            # Workflow engine settings
├── config_mcp.json                 # MCP server definitions
├── config_security.json            # Security policies
└── config_dev.json.example         # Development config template
```

**Key Configuration Files**:

**config_mcp.json** - MCP Server Registry
```json
{
  "mcpServers": {
    "ppt": {
      "command": "uvx",
      "args": ["--from", "office-powerpoint-mcp-server", "ppt_mcp_server"],
      "env": {}
    }
  }
}
```
Defines all MCP servers with connection details (command/args for stdio, url for HTTP/WebSocket). Supports environment variable substitution via `${VAR_NAME}`.

**config_agents.json** - Agent Definitions
```json
{
  "agents": [
    {
      "name": "HostAgent",
      "class_name": "HostAgent",
      "backend_scope": []
    },
    {
      "name": "GroundingAgent",
      "class_name": "GroundingAgent",
      "backend_scope": ["gui", "shell", "mcp", "system", "web"],
      "max_iterations": 20
    },
    {
      "name": "EvalAgent",
      "class_name": "EvalAgent",
      "backend_scope": ["shell"]
    }
  ]
}
```
Configures agent roles, backend access scope, and execution limits.

**config_security.json** - Security Policies
```json
{
  "security_policies": {
    "global": {
      "allow_shell_commands": true,
      "blocked_commands": {
        "common": ["rm", "-rf", "shutdown", "reboot"],
        "linux": ["mkfs", "dd", "iptables"],
        "darwin": ["diskutil", "dd", "pfctl"],
        "windows": ["del", "format", "rd"]
      },
      "sandbox_enabled": false
    },
    "backend": {
      "shell": {
        "blocked_commands": { /* platform-specific */ }
      },
      "mcp": {
        "sandbox_enabled": true
      }
    }
  }
}
```
Defines command whitelists/blacklists, sandbox settings, and access control per backend.

**config_dev.json.example** - Development Override Template
```json
{
  "shell": {
    "working_dir": "/path/to/your/workspace"
  },
  "debug": true,
  "log_level": "DEBUG"
}
```
Optional development overrides (copy to `config_dev.json` and customize). Merged with base config for local development.

</details>

<details>
<summary><b>🔧 local_server/</b> - GUI Backend Server</summary>

```
local_server/
├── __init__.py
├── main.py                         # Flask application entry point
├── config.json                     # Server configuration
├── feature_checker.py              # Platform feature detection
├── health_checker.py               # Server health monitoring
├── platform_adapters/              # OS-specific implementations
│   ├── macos_adapter.py            # macOS automation (atomacos, pyobjc)
│   ├── linux_adapter.py            # Linux automation (pyatspi, xlib)
│   ├── windows_adapter.py          # Windows automation (pywinauto)
│   └── pyxcursor.py                # Custom cursor handling
├── utils/
│   ├── accessibility.py            # Accessibility tree utilities
│   └── screenshot.py               # Screenshot capture
└── README.md
```

**Purpose**: Lightweight Flask service enabling computer control (GUI, Shell, Files, Screen capture).

</details>

<details>
<summary><b>🔧 recording/</b> - Execution Recording & Analysis</summary>

```
recording/
├── __init__.py
├── recorder.py                     # Main recording manager
├── manager.py                      # Recording lifecycle management
├── action_recorder.py              # Action-level logging
├── kanban_recorder.py              # Kanban state recording
├── video.py                        # Video capture integration
├── viewer.py                       # Trajectory viewer and analyzer
└── utils.py                        # Recording utilities
```

**Purpose**: Full execution audit with trajectory recording and video capture.

</details>

<details>
<summary><b>🔧 platform/</b> - Platform Integration</summary>

```
platform/
├── __init__.py
├── config.py                       # Platform-specific configuration
├── recording.py                    # Recording integration
├── screenshot.py                   # Screenshot utilities
└── system_info.py                  # System information gathering
```

</details>

<details>
<summary><b>🔧 utils/</b> - Shared Utilities</summary>

```
utils/
├── logging.py                      # Structured logging system
├── ui.py                           # Terminal UI components
├── display.py                      # Display formatting utilities
├── cli_display.py                  # CLI-specific display
├── ui_integration.py               # UI integration helpers
└── telemetry/                      # Usage analytics (opt-in)
    ├── __init__.py
    ├── events.py
    ├── telemetry.py
    └── utils.py
```

</details>

<details>
<summary><b>📊 logs/</b> - Execution Logs & Recordings</summary>

```
logs/
├── __main__/                              # Main application logs
│   └── fastagent_YYYY-MM-DD_HH-MM-SS.log  # Timestamped log files
│
└── recordings/                            # Execution recordings
    └── init_YYYYMMDD_HHMMSS_YYYYMMDD_HHMMSS/  # Individual recording session
        ├── metadata.json                  # Session metadata (task, timestamps, config)
        ├── traj.jsonl                     # Complete execution trajectory
        ├── agent_actions.jsonl            # Agent action history
        ├── kanban_events.jsonl            # Kanban state changes
        ├── screenshots/                   # Visual execution record
        │   ├── init.png                   # Initial screenshot
        │   ├── step_001.png               # Screenshot after step 1
        │   ├── step_002.png               # Screenshot after step 2
        │   └── ...                        # Sequential screenshots
        ├── workspace/                     # Task workspace
        │   └── [generated files]          # Files created during execution
        └── screen_recording.mp4           # Video recording (if enabled)
```

**Log Files Structure**:

- **Main Logs** (`__main__/`): Detailed application logs with initialization, configuration, agent coordination, and execution details
- **Recording Sessions** (`recordings/`): Complete audit trail for each execution with:
  - `metadata.json`: Task description, start/end times, success status, configuration
  - `traj.jsonl`: Line-delimited JSON with full execution trajectory (tool calls, results, timestamps)
  - `agent_actions.jsonl`: Agent-level actions (planning, grounding, evaluation steps)
  - `kanban_events.jsonl`: Task state transitions (TODO → IN_PROGRESS → DONE/BLOCKED)
  - `screenshots/`: Visual snapshots at each execution step
  - `workspace/`: All files created/modified during task execution
  - `screen_recording.mp4`: Full video capture (optional)
</details>

---

## 🔧 Advanced Usage

### CLI Arguments

```bash
# Use different model
python -m fastagent --model "anthropic/claude-sonnet-4-5"

# Single query mode
python -m fastagent --query "Your task"

# Debug mode
python -m fastagent --log-level DEBUG

# Fast mode (disable evaluation)
python -m fastagent --no-eval
```

> [!TIP]
> See [Evaluation Control](#evaluation-control) for detailed evaluation configuration options.

<details>
<summary>All Options</summary>

| Argument | Description |
|----------|-------------|
| `--model` | LLM model name |
| `--query` | Single query mode |
| `--timeout` | Max execution time (seconds) |
| `--log-level` | DEBUG/INFO/WARNING/ERROR |
| `--no-eval` | Disable evaluation |
| `--no-workflow` | Disable workflow |
| `--max-iterations` | Max iterations |
| `--no-ui` | Disable UI |

</details>

---

### Configuration Overview

FastAgent uses a layered configuration system:

- **`config_dev.json`** (highest priority): Local development overrides. Overrides all other configurations.
- **`config_agents.json`**: Agent definitions and backend access control
- **`config_mcp.json`**: MCP server registry
- **`config_grounding.json`**: Backend-specific settings
- **`config_workflow.json`**: Workflow engine and execution control
- **`config_security.json`**: Security policies with runtime user confirmation for sensitive operations

---

### Agent Configuration

**Path**: `fastagent/config/config_agents.json`

**Purpose**: Define agent roles, control backend access scope, and set execution limits to prevent infinite loops.

**Example configuration**:

```json
{
  "agents": [
    {
      "name": "HostAgent",
      "class_name": "HostAgent",
      "backend_scope": []
    },
    {
      "name": "GroundingAgent",
      "class_name": "GroundingAgent",
      "backend_scope": ["gui", "shell", "mcp", "system", "web"],
      "max_iterations": 20
    },
    {
      "name": "EvalAgent",
      "class_name": "EvalAgent",
      "backend_scope": ["shell"]
    }
  ]
}
```

**Key Fields**:

| Field | Description | Options/Example |
|-------|-------------|-----------------|
| `name` | Agent identifier | `"HostAgent"`, `"GroundingAgent"`, `"EvalAgent"` |
| `backend_scope` | Accessible backends | `[]` (none) or any combination of `["gui", "shell", "mcp", "system", "web"]` |
| `max_iterations` | Maximum execution cycles | Any integer (e.g., `20`, `50`) or `null` (unlimited) |

---

### MCP Configuration

**Path**: `fastagent/config/config_mcp.json`

**Purpose**: Register MCP servers with connection details. FastAgent automatically discovers tools from all registered servers and makes them available through Smart Tool RAG.

**Example configuration**:

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

---

<details>
<summary><b>Other Configuration Files</b></summary>

### Backend Configuration

**Path**: `fastagent/config/config_grounding.json`

**Purpose**: Configure backend-specific behaviors, timeouts, and Smart Tool RAG system for efficient tool selection.

**Key Fields**:

| Backend | Field | Description | Options/Default |
|---------|-------|-------------|-----------------|
| **shell** | `timeout` | Command timeout (seconds) | Any integer (default: `60`) |
| | `conda_env` | Auto-activate conda environment | Environment name or `null` (default: `"fastagent"`) |
| | `working_dir` | Working directory for command execution | Any valid path (default: current directory) |
| | `default_shell` | Shell to use | `"/bin/bash"`, `"/bin/zsh"`, etc. |
| **gui** | `timeout` | Operation timeout (seconds) | Any integer (default: `90`) |
| | `screenshot_on_error` | Capture screenshot on failure | `true` or `false` (default: `true`) |
| | `driver_type` | GUI automation driver | `"pyautogui"` or other supported drivers |
| **mcp** | `timeout` | Request timeout (seconds) | Any integer (default: `30`) |
| | `sandbox` | Run in E2B sandbox | `true` or `false` (default: `false`) |
| | `eager_sessions` | Pre-connect all servers at startup | `true` or `false` (default: `false`, lazy connection) |
| **tool_search** | `search_mode` | Tool retrieval strategy | `"semantic"`, `"hybrid"` (semantic + LLM filter), or `"llm"` (default: `"hybrid"`) |
| | `max_tools` | Maximum tools to index | Any integer (default: `300`) |
| | `enable_cache_persistence` | Persist embedding cache | `true` or `false` (default: `true`) |

---

### Workflow Configuration

**Path**: `fastagent/config/config_workflow.json`

**Purpose**: Control workflow engine behavior, evaluation policies, and global execution limits. Determines when and how agents are triggered.

**Key Fields**:

| Section | Field | Description | Options/Default |
|---------|-------|-------------|-----------------|
| **workflow** | `enable` | Enable event-driven workflow engine | `true` or `false` (default: `true`) |
| | `auto_evaluate` | Enable EvalAgent for quality assurance | `true` or `false` (default: `true`) |
| | `poll_interval` | Workflow polling interval (seconds) | Any float (default: `1.0`) |
| | `task_default_timeout` | Per-task timeout (seconds) | Any float (default: `1800`) |
| **execution** | `max_execution_time` | Global timeout for entire task (seconds) | Any float or `null` (default: `3600`) |
| | `max_iterations` | Maximum total iterations across all agents | Any integer or `null` for unlimited (default: `null`) |

---

### Security Configuration

**Path**: `fastagent/config/config_security.json`

**Purpose**: Define security policies with command filtering and access control. When sensitive operations are detected, FastAgent will **prompt for user confirmation at runtime** before execution.

**Key Fields**:

| Section | Field | Description | Options |
|---------|-------|-------------|---------|
| **global** | `allow_shell_commands` | Enable shell command execution | `true` or `false` (default: `true`) |
| | `blocked_commands` | Platform-specific command blacklist | Object with `common`, `linux`, `darwin`, `windows` arrays |
| | `sandbox_enabled` | Enable sandboxing for all operations | `true` or `false` (default: `false`) |
| | `require_user_approval` | Prompt user before sensitive operations | `true` or `false` (default: `false`) |
| **backend** | `shell`, `mcp`, `gui` | Per-backend security overrides | Same fields as global, backend-specific |

**Example blocked commands**: `rm -rf`, `shutdown`, `reboot`, `mkfs`, `dd`, `format`, `iptables`

**Behavior**: 
- Blocked commands are **rejected automatically**
- When `require_user_approval` is `true`, sensitive operations **pause execution** and prompt for user confirmation
- Sandbox mode isolates operations in secure environments (E2B sandbox for MCP)

---

### Development Configuration

**Path**: `fastagent/config/config_dev.json` (copy from `config_dev.json.example`)

**Purpose**: **Highest priority** local development overrides. This file is git-ignored and **overrides ALL other configuration files**. Perfect for testing, debugging, and personal workspace customization without affecting the repository.

**Loading Priority**: `config_grounding.json` → `config_security.json` → `config_dev.json` (dev.json overrides the former ones)

</details>

---

### Evaluation Control

```python
from fastagent import EvaluationConfig

# Disable evaluation (fastest)
config = FastAgentConfig(
    auto_evaluate=False
)

# Evaluate all steps
config = FastAgentConfig(
    auto_evaluate=True,
    evaluation_config=EvaluationConfig.all()
)

# Evaluate last step only
config = FastAgentConfig(
    auto_evaluate=True,
    evaluation_config=EvaluationConfig.last_only()
)

# Selective evaluation (recommended)
config = FastAgentConfig(
    auto_evaluate=True,
    evaluation_config=EvaluationConfig.selective(
        backends=["gui", "mcp"],  # Only eval these backends
        always_eval_last=True     # Always eval final step
    )
)
```

---

### Custom Workflow Rules

Add custom rules to control agent triggers and workflow behavior:

**Basic Rule**:

```python
from fastagent import FastAgent, FastAgentConfig, WorkflowRule, CardType, CardStatus

# Create custom rule
rule = WorkflowRule(
    name="custom_rule",
    card_type=CardType.EXECUTION,
    card_status=CardStatus.TODO,
    agent_name="GroundingAgent",
    priority=100,  # Higher priority = executed first
)

# Add to workflow
config = FastAgentConfig()
agent = FastAgent(config)
await agent.initialize()
agent.workflow_engine.add_rule(rule)
```

**Conditional Rule**:

```python
def check_backend(card):
    """Only trigger for MCP backend tasks"""
    return card.metadata.get("backend") == "mcp"

rule = WorkflowRule(
    name="mcp_only_rule",
    card_type=CardType.EXECUTION,
    card_status=CardStatus.TODO,
    agent_name="GroundingAgent",
    condition=check_backend,  # Custom condition function
    timeout=300.0
)
agent.workflow_engine.add_rule(rule)
```

---

### Python API

<details>
<summary><b>Basic Usage & Batch Processing</b></summary>

**Single Task**:

```python
import asyncio
from fastagent import FastAgent, FastAgentConfig

async def main():
    config = FastAgentConfig(
        llm_model="anthropic/claude-sonnet-4-5",
        max_execution_time=1800.0
    )
    
    agent = FastAgent(config)
    await agent.initialize()
    
    result = await agent.run("Your task")
    print(result["user_response"])
    
    await agent.cleanup()

asyncio.run(main())
```

**Batch Processing**:

```python
async def batch_process(tasks: list[str]):
    """Process multiple tasks in the same session"""
    agent = FastAgent(FastAgentConfig())
    await agent.initialize()  # Create sessions once
    
    results = []
    for i, task in enumerate(tasks, 1):
        print(f"\nProcessing task {i}/{len(tasks)}: {task}")
        result = await agent.run(task)
        results.append(result)
    
    await agent.cleanup()  # Clean up sessions once
    return results

# Example usage
tasks = ["Task 1", "Task 2", "Task 3"]
results = asyncio.run(batch_process(tasks))
```

**💡 Note**: Tasks in batch processing share the same **sessions** (GUI, Shell, MCP servers, etc.) and are executed **sequentially**. This is efficient for related tasks that need to maintain context across executions.

</details>

---

## 🔗 Related Projects

FastAgent builds upon excellent open-source projects, we sincerely thank their authors and contributors:

- **[OSWorld](https://github.com/xlang-ai/OSWorld)**: Comprehensive benchmark for evaluating computer-use agents across diverse operating system tasks, providing standardized evaluation environments and metrics.
- **[mcp-use](https://github.com/mcp-use/mcp-use)**: Platform that simplifies MCP agent development with client SDKs, hosted gateway for routing/authentication, and single-endpoint aggregation of multiple MCP servers.

---

<div align="center">

**🌟 If this project helps you, please give us a Star!**

**🤖 Experience AI's full potential in unified research, analysis, and computer automation!**  

</div>