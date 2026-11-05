__all__ = ["topo_sort_forward_bfs", "topo_sort_forward_dfs", "topo_sort_backward"]

from ._node import TNode


def _check_input_output_number(nodes: list[TNode], verbose: bool = False):
    """Verify that the graph has exactly one input node and one output node."""
    n_inputs = 0
    n_outputs = 0
    for node in nodes:
        if len(node.pre_nodes) == 0:
            n_inputs += 1
        if len(node.next_nodes) == 0:
            n_outputs += 1

    if verbose:
        print(f"The DAG graph has {n_inputs} inputs and {n_outputs} outputs.")


def topo_sort_forward_dfs(nodes: list[TNode], verbose: bool = False) -> list[TNode]:
    """Perform a DFS (depth-first search) for topological sort of nodes."""
    _check_input_output_number(nodes, verbose)

    visited = set()
    temp_mark = set()
    sorted_nodes = []

    def dfs(node):
        if node in temp_mark:
            raise ValueError("Graph has a cycle, cannot perform topological sort")
        if node not in visited:
            temp_mark.add(node)
            for next_node in node.next_nodes:
                dfs(next_node)
            temp_mark.remove(node)
            visited.add(node)
            sorted_nodes.append(node)

    for node in nodes:
        if len(node.pre_nodes) == 0:
            dfs(node)

    if len(sorted_nodes) != len(nodes):
        raise ValueError("Graph has a cycle, cannot perform topological sort")

    return sorted_nodes[::-1]


def topo_sort_forward_bfs(nodes: list[TNode], verbose: bool = False) -> list[TNode]:
    """Perform a BFS (breadth-first search) for topological sort of nodes."""
    _check_input_output_number(nodes, verbose)

    in_degrees = {node: len(node.pre_nodes) for node in nodes}

    queue = [node for node in nodes if in_degrees[node] == 0]
    sorted_nodes = []
    while queue:
        node = queue.pop(0)
        sorted_nodes.append(node)
        for next_node in node.next_nodes:
            in_degrees[next_node] -= 1
            if in_degrees[next_node] == 0:
                queue.append(next_node)

    if len(sorted_nodes) != len(nodes):
        raise ValueError("Graph has a cycle, cannot perform topological sort")

    return sorted_nodes


def topo_sort_backward(
    nodes: list[TNode], verbose: bool = False
) -> dict[TNode, list[TNode]]:
    """Generate backward topological sorts for each node."""

    def dfs(node, visited, result):
        if node in visited:
            return
        visited.add(node)
        for pre_node in node.pre_nodes:
            dfs(pre_node, visited, result)
        result.append(node)

    backward_sorts = {}
    for node in nodes:
        visited = set()
        result = []
        dfs(node, visited, result)
        backward_sorts[node] = result

    for node, backward_sort in backward_sorts.items():
        backward_sorts[node] = backward_sort[::-1]


    return backward_sorts
