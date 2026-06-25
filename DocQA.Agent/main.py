import sys

from langchain_core.messages import AIMessage, ToolMessage

from app.agent.graph import build_agent

_STREAM_DIVIDER = "-" * 60


def _print_stream(agent, user_message: str) -> None:
    """Stream agent steps to stdout so the reasoning loop is visible."""
    for step in agent.stream({"messages": [{"role": "user", "content": user_message}]}):
        if "agent" in step:
            msg = step["agent"]["messages"][-1]
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for call in msg.tool_calls:
                        print(f"\n[tool call] {call['name']}({call['args']})")
                else:
                    print(f"\n{_STREAM_DIVIDER}\n{msg.content}")
        elif "tools" in step:
            msg = step["tools"]["messages"][-1]
            if isinstance(msg, ToolMessage):
                preview = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
                print(f"[tool result] {preview}")


def main():
    stream = "--stream" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--stream"]

    if not sys.stdin.isatty():
        code = sys.stdin.read().strip()
    elif args:
        code = " ".join(args)
    else:
        print("Usage:")
        print('  echo "<code snippet>" | uv run main.py [--stream]')
        print('  uv run main.py "<code snippet>" [--stream]')
        sys.exit(1)

    user_message = f"Analyze and migrate this Semantic Kernel code:\n\n```\n{code}\n```"
    agent = build_agent()

    if stream:
        _print_stream(agent, user_message)
    else:
        result = agent.invoke({"messages": [{"role": "user", "content": user_message}]})
        print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
