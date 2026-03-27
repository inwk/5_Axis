"""Shared dataset schema constants for the Graph-SDF process planner."""

MACRO_CLASS_TO_ID = {
    "3_axis_rough": 0,
    "3_axis_finish": 1,
    "3p2_axis_rough": 2,
    "3p2_axis_finish": 3,
    "5_axis_point_finish": 4,
    "5_axis_flank_finish": 5,
    "stop": 6,
}

TOOL_LIBRARY = (
    ("flat", 20.0),
    ("flat", 16.0),
    ("flat", 12.0),
    ("flat", 10.0),
    ("flat", 8.0),
    ("flat", 6.0),
    ("flat", 4.0),
    ("ball", 8.0),
    ("ball", 6.0),
    ("ball", 4.0),
)


def tool_choice_key(tool_kind: str, tool_diameter: float) -> str:
    """Returns a stable tool-choice key from library attributes."""
    return f"{str(tool_kind).lower()}_{float(tool_diameter):.3f}"


TOOL_CHOICE_TO_ID = {
    tool_choice_key(tool_kind, tool_diameter): idx
    for idx, (tool_kind, tool_diameter) in enumerate(TOOL_LIBRARY)
}

ID_TO_TOOL_CHOICE = {
    idx: (tool_kind, tool_diameter)
    for idx, (tool_kind, tool_diameter) in enumerate(TOOL_LIBRARY)
}

ID_TO_MACRO_CLASS = {value: key for key, value in MACRO_CLASS_TO_ID.items()}


def macro_class_name_from_id(macro_class_id: int) -> str:
    """Returns macro class name from id or 'unknown' when out of range."""
    return ID_TO_MACRO_CLASS.get(int(macro_class_id), "unknown")


def build_tool_choice_mask_for_macro_class(macro_class_id: int) -> list[int]:
    """Builds a binary invalid-mask (1=invalid) for tool choices by macro class."""
    macro_name = macro_class_name_from_id(macro_class_id)
    mask = [1 for _ in TOOL_LIBRARY]

    if macro_name in {"3_axis_rough", "3p2_axis_rough", "5_axis_flank_finish"}:
        for idx, (tool_kind, _) in enumerate(TOOL_LIBRARY):
            if tool_kind == "flat":
                mask[idx] = 0
    elif macro_name == "5_axis_point_finish":
        for idx, (tool_kind, _) in enumerate(TOOL_LIBRARY):
            if tool_kind == "ball":
                mask[idx] = 0
    elif macro_name in {"3_axis_finish", "3p2_axis_finish"}:
        mask = [0 for _ in TOOL_LIBRARY]
    elif macro_name == "stop":
        mask = [1 for _ in TOOL_LIBRARY]
    else:
        mask = [0 for _ in TOOL_LIBRARY]

    return mask
