"""
Weighted directed graph object with neighborhood boundary method.

(adapted from https://www.python-course.eu/graphs_python.php)
"""
import numpy as np

class DiGraph(object):
    # Initialize an empty graph object.
    def __init__(self):
        self.__graph_dict = {}

    # Add an edge to the graph.
    # The edge is tuple or list of 3 elements (vertexFrom, vertexTo, weight).
    # Note: does not check for multiple edges.
    def add_edge(self, edge):
        (tail, head, weight) = tuple(edge)
        if tail in self.__graph_dict:
            self.__graph_dict[tail].append((head, weight))
        else:
            self.__graph_dict[tail] = [(head, weight)]

    # Helper method to return a list of edges of the graph.
    def __generate_edges(self):
        edges = []
        for tail in self.__graph_dict:
            for body in self.__graph_dict[tail]:
                edges.append((tail, body[0], body[1]))
        return edges

    def __str__(self):
        res = "vertices from: "
        for v in self.__graph_dict:
            res += str(v) + " "
        res += "\nedges: "
        for edge in self.__generate_edges():
            res += str(edge) + " "
        return res

    # Return a 2 by n matrix where n is the specified total number of nodes in the graph.
    # The first row is the normalized neighborhood boundary at the specified depth starting at the specified root
    # node, and the second row is the same but at one plus the specified depth.
    # Put a zero row if there are no nodes at the specified depth.
    def nnb(self, root, depth, n):
        nnb_mat = np.zeros((2, n))

        # Start breadth-first search at the specified root node and run it to the specified depth.
        # (adapted from https://www.ics.uci.edu/~eppstein/PADS/BFS.py)
        # Keep track of visited nodes.
        visited = set()
        # The current layer is stored as a set of pairs, where each pair is
        #   (node at the current layer, product of edge weights along a BFS tree to the node).
        current_layer = {(root, 1)}
        for layer in range(depth):
            for v in current_layer:
                visited.add(v[0])
            next_layer = set()
            for v in current_layer:
                bodies = self.__graph_dict.get(v[0])
                if bodies is not None:
                    for body in bodies:
                        if body[0] not in visited:
                            next_layer.add((body[0], v[1] * body[1]))
            current_layer = next_layer

        current_layer_size = 0
        for v in current_layer:
            nnb_mat[0, v[0]] = v[1]
            current_layer_size += 1
        if current_layer_size > 0:
            nnb_mat[0, :] /= current_layer_size

        # Run the breadth-first search for one more layer.
        for v in current_layer:
            visited.add(v[0])
        next_layer = set()
        for v in current_layer:
            bodies = self.__graph_dict.get(v[0])
            if bodies is not None:
                for body in bodies:
                    if body[0] not in visited:
                        next_layer.add((body[0], v[1] * body[1]))

        next_layer_size = 0
        for v in next_layer:
            nnb_mat[1, v[0]] = v[1]
            next_layer_size += 1
        if next_layer_size > 0:
            nnb_mat[1, :] /= next_layer_size

        return nnb_mat


if __name__ == "__main__":
    # Unit tests.
    # Simple test.
    print("Graph 1")
    graph = DiGraph()
    graph.add_edge((0, 1, 0.5))
    graph.add_edge((0, 2, 0.4))
    graph.add_edge((1, 3, 0.3))
    graph.add_edge((1, 4, 0.2))
    graph.add_edge((2, 5, 0.1))
    graph.add_edge((5, 6, 0.01))
    print(graph)

    print()
    print("Tests")
    print(graph.nnb(0, 0, 7))
    print(graph.nnb(0, 1, 7))
    print(graph.nnb(0, 2, 7))
    print(graph.nnb(0, 3, 7))
    print(graph.nnb(2, 1, 7))

    print()
    # The directed graph object distinguishes edges in opposite directions.
    print("Graph 2")
    graph = DiGraph()
    graph.add_edge((0, 1, 0.5))
    graph.add_edge((1, 0, 0.4))
    print(graph)

    print()
    print("Tests")
    print(graph.nnb(0, 0, 2))
    print(graph.nnb(0, 1, 2))
    print(graph.nnb(0, 2, 2))
    print(graph.nnb(1, 0, 2))
    print(graph.nnb(1, 1, 2))
    print(graph.nnb(1, 2, 2))

    print()
    print("Graph 3")
    # The neighborhood boundary computation ignores self-edges.
    graph = DiGraph()
    graph.add_edge((0, 1, 0.5))
    graph.add_edge((1, 1, 0.4))
    graph.add_edge((1, 2, 0.3))
    print(graph)

    print()
    print("Tests")
    print(graph.nnb(0, 0, 3))
    print(graph.nnb(0, 1, 3))
    print(graph.nnb(0, 2, 3))
    print(graph.nnb(0, 3, 3))
