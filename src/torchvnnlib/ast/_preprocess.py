__all__ = ["preprocess_vnnlib"]

import re


def _remove_comments(lines: list[str]) -> list[str]:
    new_lines = []

    for line in lines:
        line = re.sub(r";.*", "", line)
        line = line.strip()
        if line:
            new_lines.append(line)

    return new_lines


def _remove_declare_clauses(lines: list[str]) -> tuple[list[str], int, int]:
    """Removes all lines that declare variables in the VNNLIB file."""
    new_lines = []
    n_inputs = 0
    n_outputs = 0

    for line in lines:
        if "declare-const" not in line:
            new_lines.append(line)
        elif "X" in line:
            n_inputs += 1
        elif "Y" in line:
            n_outputs += 1

    return new_lines, n_inputs, n_outputs


def _merge_multi_line_expr(lines: list[str]) -> list[str]:
    """Merges multiple lines of a VNNLIB expression into a single line."""
    new_lines = []
    current_expr = ""
    paren_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue

        current_expr += " " + stripped
        paren_count += stripped.count("(")
        paren_count -= stripped.count(")")

        if paren_count == 0 and current_expr.strip():
            new_lines.append(current_expr.strip())
            current_expr = ""

    if current_expr.strip():
        raise ValueError("Unbalanced parentheses in VNNLIB file.")

    return new_lines


def _check_illegal_lines(lines: list[str]) -> None:
    """Checks for illegal lines in the VNNLIB file."""
    for line in lines:
        if not re.match(r"^\(assert\s+.*\)$", line):
            raise ValueError(f"Illegal line found in VNNLIB file: {line}")


def preprocess_vnnlib(lines: list[str]) -> tuple[list[str], int, int]:
    """Pre-processes the lines of a VNNLIB file by removing comments, declare clauses, and merging multi-line expressions."""
    lines = _remove_comments(lines)
    lines, n_inputs, n_outputs = _remove_declare_clauses(lines)
    lines = _merge_multi_line_expr(lines)
    _check_illegal_lines(lines)

    return lines, n_inputs, n_outputs
