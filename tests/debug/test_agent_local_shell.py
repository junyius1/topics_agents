#!/usr/bin/env python3
"""
Test script to demonstrate and verify the LLM agent's task decomposition and local shell execution.
This script:
1. Loads environment variables from the .env file.
2. Supports flexible configuration for cloud and local models (e.g. Ollama, custom OpenAI endpoints).
3. Initializes a LocalShellBackend (unisolated local shell executor).
4. Instantiates an Open SWE Deep Agent with filesystem and execute tools.
5. Sends a structured command to showcase the LLM decomposing the goal into shell commands and executing them.
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path

# Add the project root to Python path so we can import 'agent'
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# 1. Load the .env file the standard Open SWE way (using python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=project_root / ".env")
    print("✅ Successfully loaded environment variables from .env")
except ImportError:
    print("⚠️ python-dotenv is not installed. Relying on current system environment variables.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test script to verify Open SWE agent task decomposition and local shell execution."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name/ID (e.g., 'gpt-4o', 'llama3', 'qwen2.5-coder', 'openai:gpt-4o', 'ollama:llama3')."
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="LangChain model provider to use (e.g., 'ollama', 'openai', 'anthropic', 'google_genai')."
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Custom base API URL for local/self-hosted model endpoints (e.g., 'http://localhost:11434/v1')."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Custom API key for the model endpoint (if required)."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (default: 0.0)."
    )
    return parser.parse_args()


def get_chat_model(args):
    model_id = args.model
    provider = args.provider
    base_url = args.base_url
    api_key = args.api_key

    # Check for environment overrides if CLI args are not set
    if not base_url:
        base_url = os.getenv("LOCAL_BASE_URL") or os.getenv("LOCAL_LLM_BASE_URL")
    if not api_key:
        api_key = os.getenv("LOCAL_API_KEY") or os.getenv("OLLAMA_API_KEY")

    # If the user passed in or has LLM_MODEL_ID set in their env
    env_model_id = os.getenv("LLM_MODEL_ID") or os.getenv("LOCAL_MODEL_ID")

    # Auto-resolve defaults if nothing is specified
    if not model_id:
        if env_model_id:
            model_id = env_model_id
            if not provider and ("local" in model_id.lower() or "qwen" in model_id.lower()):
                provider = "openai"  # OpenAI compatible
        elif base_url:
            model_id = "llama3"
            if not provider:
                provider = "openai"  # Default to openai-compatible for custom base URL
        else:
            # Fall back to cloud providers based on available keys
            if os.getenv("OPENAI_API_KEY"):
                model_id = "openai:gpt-4o"
            elif os.getenv("ANTHROPIC_API_KEY"):
                model_id = "anthropic:claude-3-5-sonnet-latest"
            elif os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
                model_id = "google:gemini-1.5-pro"
            else:
                # Local developer fallback: default to Ollama
                print("ℹ️ No cloud API keys found. Defaulting to local Ollama configuration...")
                model_id = "llama3"
                provider = "ollama"
                base_url = base_url or "http://localhost:11434"

    # Extract provider from model_id prefix if present (e.g. "ollama:llama3" or "local:model_name")
    if model_id and ":" in model_id:
        parts = model_id.split(":", 1)
        possible_provider = parts[0]
        if possible_provider in ["openai", "anthropic", "google", "google_genai", "ollama", "groq", "fireworks"]:
            if not provider:
                provider = possible_provider
            model_id = parts[1]
        elif possible_provider == "local":
            if not provider:
                provider = "openai"  # local is usually OpenAI compatible
            model_id = parts[1]

    print(f"🤖 Configuring Chat Model:")
    print(f"   - Model Name: {model_id}")
    print(f"   - Provider:   {provider or 'auto'}")
    if base_url:
        print(f"   - Base URL:   {base_url}")
    if api_key:
        print(f"   - API Key:    {'*' * len(api_key) if len(api_key) > 4 else 'configured'}")

    from langchain.chat_models import init_chat_model

    # If this is a standard cloud model without custom endpoint/provider overrides,
    # use open-swe's make_model helper to benefit from default responses API & retry policies.
    if not base_url and not provider and not api_key:
        full_model_id = args.model or model_id
        if ":" not in full_model_id and provider:
            full_model_id = f"{provider}:{model_id}"

        # Ensure cloud keys exist before attempting production make_model, else fallback
        if (
            (full_model_id.startswith("openai:") and os.getenv("OPENAI_API_KEY"))
            or (full_model_id.startswith("anthropic:") and os.getenv("ANTHROPIC_API_KEY"))
            or ((full_model_id.startswith("google:") or full_model_id.startswith("google_genai:")) 
                and (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")))
        ):
            try:
                from agent.utils.model import make_model
                return make_model(full_model_id, temperature=args.temperature)
            except Exception as e:
                print(f"ℹ️ falling back to direct init_chat_model due to make_model error: {e}")

    # Build custom initialization parameters
    init_kwargs = {
        "model": model_id,
        "temperature": args.temperature,
    }
    if provider:
        init_kwargs["model_provider"] = provider
    if base_url:
        init_kwargs["base_url"] = base_url
    if api_key:
        init_kwargs["api_key"] = api_key

    return init_chat_model(**init_kwargs)


from deepagents import create_deep_agent
from agent.integrations.local import create_local_sandbox


def run_e2e_agent_test():
    args = parse_args()

    # 2. Setup a local sandbox directory for testing (isolated from system files)
    temp_dir = tempfile.TemporaryDirectory(prefix="openswe_test_sandbox_")
    os.environ["LOCAL_SANDBOX_ROOT_DIR"] = temp_dir.name
    print(f"📂 Created local sandbox root directory: {temp_dir.name}")

    # Initialize the local shell sandbox backend
    sandbox_backend = create_local_sandbox()
    
    # 3. Configure the LLM
    try:
        model = get_chat_model(args)
    except Exception as e:
        print(f"\n❌ Failed to initialize the model: {e}")
        print("Please check your --model, --provider, or local engine setup.")
        temp_dir.cleanup()
        sys.exit(1)

    # 4. Construct the backend factory callback for deepagents
    # This teaches deepagents to route all file/execute tool commands to our local sandbox backend
    def backend_factory(runtime_instance: object) -> object:
        return sandbox_backend

    # 5. Define a robust system prompt for our agent
    system_prompt = (
        "You are a helpful software engineering assistant with shell execution privileges.\n"
        "You are working in the directory specified by the workspace.\n"
        "Decompose tasks systematically, run shell commands to perform actions, check results, "
        "and explain your reasoning. Keep replies concise and task-focused."
    )

    print("🏗️ Creating deep agent with sandbox support...")
    # create_deep_agent automatically injects the 'execute' (shell), 'write_file', 'edit_file', etc.
    # tools and hooks them up to the backend returned by our backend_factory.
    agent_app = create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=[],  # You can pass additional custom tools here
        backend=backend_factory,
    )

    # 6. Execute a multi-step command that forces task decomposition and local shell execution
    test_task = (
        "1. Perform a web search for the topic '三文鱼的选购' (purchase of salmon). Since you do not have a direct search tool, you should write a Python script that searches the web (e.g., using a simple scraper or API for DuckDuckGo, Bing, or another search engine, or using python packages) to find relevant URLs. Always use relative paths (like './search_salmon.py' or 'result.md') rather than absolute root paths (like '/search_salmon.py') to avoid virtual path resolution issues.\n"
        "2. Extract the URLs of the top 5 relevant search results.\n"
        "3. For each of the 5 URLs, download the webpage content, extract its main article body/text, and format it into a clean Markdown (.md) file.\n"
        "4. Save these 5 Markdown files in the current workspace directory, listing their filenames and summarizing their contents to confirm they are successfully saved."
    )

    print(f"\n🚀 Dispatching task to agent:\n{test_task}\n" + "="*50)

    # Invoke the LangGraph pregel app
    config = {
        "configurable": {
            "thread_id": "test-debug-thread-1",
        }
    }
    
    # Stream the conversational execution steps in real-time
    try:
        for event in agent_app.stream({"messages": [("user", test_task)]}, config=config):
            # Extract messages from the node event dictionary (e.g. {"model": {"messages": [...]}})
            messages = []
            if "messages" in event:
                messages = event["messages"]
            else:
                for node_update in event.values():
                    if isinstance(node_update, dict) and "messages" in node_update:
                        node_msgs = node_update["messages"]
                        if isinstance(node_msgs, list):
                            messages.extend(node_msgs)
                        else:
                            messages.append(node_msgs)

            # Check for tool call execution logs
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        print(f"\n🛠️ LLM invoked Tool: {tool_call['name']}")
                        print(f"   Args: {tool_call['args']}")
                elif msg.type == "tool":
                    print(f"🔌 Tool output: {msg.content[:200]}...")
                elif msg.type == "ai" and msg.content:
                    print(f"\n🤖 Agent: {msg.content}")
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
    finally:
        # Cleanup
        temp_dir.cleanup()
        print("\n🧹 Cleaned up local sandbox temporary directory.")


if __name__ == "__main__":
    run_e2e_agent_test()
