import asyncio
import argparse
import sys
import logging
from typing import Optional

from fastagent.fastagent import FastAgent, FastAgentConfig
from fastagent.utils.logging import Logger
from fastagent.utils.ui import create_ui, FastAgentUI
from fastagent.utils.ui_integration import UIIntegration
from fastagent.utils.cli_display import CLIDisplay
from fastagent.utils.display import colorize

logger = Logger.get_logger(__name__)


class UIManager:
    def __init__(self, ui: Optional[FastAgentUI], ui_integration: Optional[UIIntegration]):
        self.ui = ui
        self.ui_integration = ui_integration
        self._original_log_levels = {}
    
    async def start_live_display(self):
        if not self.ui or not self.ui_integration:
            return
        
        print()
        print(colorize("  ▣ Starting real-time visualization...", 'c'))
        print()
        await asyncio.sleep(1)
        
        self._suppress_logs()
        
        await self.ui.start_live_display()
        await self.ui_integration.start_monitoring(poll_interval=2.0)
    
    async def stop_live_display(self):
        if not self.ui or not self.ui_integration:
            return
        
        await self.ui_integration.stop_monitoring()
        await self.ui.stop_live_display()
        
        self._restore_logs()
    
    def print_summary(self, result: dict):
        if self.ui:
            self.ui.print_summary(result)
        else:
            CLIDisplay.print_result_summary(result)
    
    def _suppress_logs(self):
        log_names = ["fastagent", "fastagent.grounding", "fastagent.agents", "fastagent.workflow"]
        for name in log_names:
            log = logging.getLogger(name)
            self._original_log_levels[name] = log.level
            log.setLevel(logging.CRITICAL)
    
    def _restore_logs(self):
        for name, level in self._original_log_levels.items():
            logging.getLogger(name).setLevel(level)
        self._original_log_levels.clear()


async def _execute_task(agent: FastAgent, query: str, ui_manager: UIManager):
    await ui_manager.start_live_display()
    result = await agent.run(query)
    await ui_manager.stop_live_display()
    ui_manager.print_summary(result)
    return result


async def interactive_mode(agent: FastAgent, ui_manager: UIManager):
    CLIDisplay.print_interactive_header()
    
    while True:
        try:
            prompt = colorize(">>> ", 'c', bold=True)
            query = input(f"\n{prompt}").strip()
            
            if not query:
                continue
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("\nExiting...")
                break

            if query.lower() == 'status':
                CLIDisplay.print_status(agent)
                continue
            
            if query.lower() == 'help':
                CLIDisplay.print_help()
                continue

            CLIDisplay.print_task_header(query)
            await _execute_task(agent, query, ui_manager)
            
        except KeyboardInterrupt:
            print("\n\nInterrupt signal detected, exiting...")
            break
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print(f"\nError: {e}")


async def single_query_mode(agent: FastAgent, query: str, ui_manager: UIManager):
    CLIDisplay.print_task_header(query, title="▶ Single Query Execution")
    await _execute_task(agent, query, ui_manager)


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='FastAgent - Simple and Fast Agents System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--config', '-c', type=str, help='Configuration file path (JSON format)')
    parser.add_argument('--query', '-q', type=str, help='Single query mode: execute query directly')
    
    parser.add_argument('--model', '-m', type=str, help='LLM model name')
    parser.add_argument('--poll-interval', type=float, help='Workflow polling interval (seconds)')
    
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Log level')
    parser.add_argument('--no-workflow', action='store_true', help='Disable event-driven workflow')
    parser.add_argument('--no-eval', action='store_true', help='Disable automatic evaluation')
    
    parser.add_argument('--max-iterations', type=int, help='Maximum iterations')
    parser.add_argument('--timeout', type=float, help='Maximum execution time (seconds)')
    
    parser.add_argument('--interactive', '-i', action='store_true', help='Force interactive mode')
    parser.add_argument('--no-ui', action='store_true', help='Disable visual UI')
    parser.add_argument('--ui-compact', action='store_true', help='Use compact UI layout')
    
    return parser


def _load_config(args) -> FastAgentConfig:
    cli_overrides = {}
    if args.model:
        cli_overrides['llm_model'] = args.model
    if args.no_workflow:
        cli_overrides['enable_workflow'] = False
    if args.no_eval:
        cli_overrides['auto_evaluate'] = False
    if args.max_iterations is not None:
        cli_overrides['max_iterations'] = args.max_iterations
    if args.timeout is not None:
        cli_overrides['max_execution_time'] = args.timeout
    if args.poll_interval is not None:
        cli_overrides['poll_interval'] = args.poll_interval
    if args.log_level:
        cli_overrides['log_level'] = args.log_level
    
    try:
        config = FastAgentConfig.load(
            user_config_path=args.config,
            cli_overrides=cli_overrides if cli_overrides else None,
        )
        
        if args.config:
            print(f"✓ Loaded from config file: {args.config}")
        else:
            print("✓ Using default configuration")
        
        if cli_overrides:
            print(f"✓ CLI overrides: {', '.join(cli_overrides.keys())}")
        
        if args.log_level:
            Logger.set_level(args.log_level)
        
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)


def _setup_ui(args) -> tuple[Optional[FastAgentUI], Optional[UIIntegration]]:
    if args.no_ui:
        CLIDisplay.print_banner()
        return None, None
    
    ui = create_ui(enable_live=True, compact=args.ui_compact)
    ui.print_banner()
    ui_integration = UIIntegration(ui)
    return ui, ui_integration


async def _initialize_agent(config: FastAgentConfig, args) -> FastAgent:
    agent = FastAgent(config)
    
    init_steps = [("Initializing FastAgent...", "loading")]
    CLIDisplay.print_initialization_progress(init_steps, show_header=False)
    
    if not args.config:
        original_log_level = Logger.get_logger("fastagent").level
        for log_name in ["fastagent", "fastagent.grounding", "fastagent.agents"]:
            Logger.get_logger(log_name).setLevel(logging.WARNING)
    
    await agent.initialize()
    
    if not args.config:
        for log_name in ["fastagent", "fastagent.grounding", "fastagent.agents"]:
            Logger.get_logger(log_name).setLevel(original_log_level)
    
    init_steps = [
        ("AI Model (LLM Client)", "ok"),
        (f"Grounding Backends ({len(agent._grounding_client.list_providers())} available)", "ok"),
        (f"Intelligent Agents ({len(agent.coordinator.list_agents())} active)", "ok"),
    ]
    
    if config.enable_workflow:
        init_steps.append(("Workflow Engine (Event-driven)", "ok"))
    
    if config.enable_recording:
        init_steps.append(("Recording Manager", "ok"))
    
    CLIDisplay.print_initialization_progress(init_steps, show_header=True)
    
    return agent


async def main():
    parser = _create_argument_parser()
    args = parser.parse_args()
    
    config = _load_config(args)
    
    ui, ui_integration = _setup_ui(args)
    
    CLIDisplay.print_configuration(config)
    
    try:
        agent = await _initialize_agent(config, args)
        
        if ui_integration:
            ui_integration.attach_kanban(agent.kanban)
            ui_integration.attach_llm_client(agent._llm_client)
            ui_integration.attach_grounding_client(agent._grounding_client)
            CLIDisplay.print_system_ready()
        
        ui_manager = UIManager(ui, ui_integration)
        
        if args.query:
            await single_query_mode(agent, args.query, ui_manager)
        else:
            await interactive_mode(agent, ui_manager)
        
    except KeyboardInterrupt:
        print("\n\nInterrupt signal detected")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\nError: {e}")
        return 1
    finally:
        print("\nCleaning up resources...")
        await agent.cleanup()
    
    print("\nGoodbye!")
    return 0


def run_main():
    """Run the main function"""
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nProgram interrupted")
        sys.exit(0)


if __name__ == "__main__":
    run_main()