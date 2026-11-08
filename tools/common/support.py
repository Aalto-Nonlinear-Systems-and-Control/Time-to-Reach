"""What the runners hand to the compared tools: only what a tool handles itself.

The comparison shows each tool with its native capabilities. A runner does not add
set handling or case splitting of its own, so a benchmark outside these rules is
reported as unsupported (N/A in the tables) with the reason.
"""

BOX_TO_BOX = "outside the box-to-box setting of the runner"


def box_to_box_reason(bm):
    """Why the benchmark is not a forward problem from one initial box to one target box, or None.

    The set-propagation tools (CORA, JuliaReach) are run on this setting: the tool
    gets the initial box, and each of its reachable sets is compared with the
    target box (disjoint from it, inside it, or neither).
    """
    initial, target = bm.raw["initial_sets"], bm.raw["target_sets"]
    if any(s["type"] != "box" for s in initial + target):
        return f"{BOX_TO_BOX}: the initial or the target set is a ball or a level set"
    if len(initial) != 1 or len(target) != 1:
        return f"{BOX_TO_BOX}: there are several initial or target sets"
    if bm.direction != "forward":
        return f"{BOX_TO_BOX}: the benchmark is a backward reachability problem"
    return None

