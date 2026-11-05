__all__ = ["str_list"]


def str_list(lst: list[float]) -> str:
    if len(lst) == 0:
        return "[]"

    string = "["
    for i in range(len(lst)):
        string += f"{lst[i]:.2f}, "
        if i > 10:
            string += ", ..., "
            break
    string = string[:-2] + "]"

    return string
