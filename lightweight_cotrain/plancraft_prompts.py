"""Shared Plancraft Main/Sub prompts and rollout history formatting."""

SUB_SYSTEM = (
    "You are a Plancraft sub agent. Inspect the current objective, inventory, and action history.\n"
    "Suggest exactly one next low-level action that may help craft the target.\n"
    "Valid action formats are:\n"
    "move: from [Source] to [Target] with quantity N\n"
    "smelt: from [Source] to [Target] with quantity N\n"
    "impossible: short reason\n"
    "Slots include [A1]-[C3], [I1]-[I36], and output slot [0]."
)

STRUCTURED_SUB_SYSTEM = (
    "You are a Plancraft sub agent. Inspect the current objective, inventory, and action history.\n"
    "Give structured guidance for the main agent.\n"
    "Output exactly this format:\n"
    "<subgoal>local crafting goal</subgoal>\n"
    "<reason>brief reason based on the current state</reason>\n"
    "<action>one recommended low-level action</action>\n"
    "Valid action formats inside <action> are:\n"
    "move: from [Source] to [Target] with quantity N\n"
    "smelt: from [Source] to [Target] with quantity N\n"
    "impossible: short reason\n"
    "Slots include [A1]-[C3], [I1]-[I36], and output slot [0]."
)

MAIN_SYSTEM = (
    "You are a Plancraft main agent. Use the sub agent advice, but output exactly one executable action.\n"
    "Valid action formats are:\n"
    "move: from [Source] to [Target] with quantity N\n"
    "smelt: from [Source] to [Target] with quantity N\n"
    "impossible: short reason\n"
    "Do not output explanations after the action."
)


def history_text(history: list[tuple[str, str, str]], history_steps: int = 3) -> str:
    """Render the same recent Main/Sub trajectory for SFT, evaluation, and RL."""
    if not history:
        return "No actions yet."
    start_step = max(len(history) - history_steps, 0)
    lines = []
    for idx, (advice, action, observation) in enumerate(history[start_step:], start_step + 1):
        lines.append(f"Step {idx} sub advice: {advice}")
        lines.append(f"Step {idx} main action: {action}")
        lines.append(f"Step {idx} observation: {observation}")
    return "\n".join(lines)
