# graphviz nutzen um Graphen zu visualisieren
import graphviz

class Digraph:
    def __init__(self, edge_list=None):
        self.adj_list = {}
        self.weights = {}
        self.capacities = {}
        if edge_list:
            for edge in edge_list:
                src, dest = edge[0], edge[1]
                weight = edge[2] if len(edge) > 2 else 1  # Default weight is 1 if not provided
                capacity = edge[3] if len(edge) > 3 else float('inf')  # Default capacity is infinity if not provided
                self.add_edge(src, dest, weight, capacity)

    def add_vertex(self, vertex):
        if vertex not in self.adj_list:
            self.adj_list[vertex] = []

    def add_edge(self, src, dest, weight=1, capacity=float('inf')):
        if src not in self.adj_list:
            self.add_vertex(src)
        if dest not in self.adj_list:
            self.add_vertex(dest)
        self.adj_list[src].append(dest)
        self.weights[(src, dest)] = weight
        self.capacities[(src, dest)] = capacity

    def delete_vertex(self, vertex):
        if vertex in self.adj_list:
            del self.adj_list[vertex]
            for src in self.adj_list:
                if vertex in self.adj_list[src]:
                    self.adj_list[src].remove(vertex)
            self.weights = {(src, dest): w for (src, dest), w in self.weights.items() if src != vertex and dest != vertex}
            self.capacities = {(src, dest): c for (src, dest), c in self.capacities.items() if src != vertex and dest != vertex}

    def delete_edge(self, src, dest):
        if src in self.adj_list and dest in self.adj_list[src]:
            self.adj_list[src].remove(dest)
            if (src, dest) in self.weights:
                del self.weights[(src, dest)]
            if (src, dest) in self.capacities:
                del self.capacities[(src, dest)]

    def __str__(self):
        result = ""
        for src in self.adj_list:
            for dest in self.adj_list[src]:
                result += f"{src} -> {dest} (weight: {self.weights[(src, dest)]}, capacity: {self.capacities[(src, dest)]})\n"
        return result


class Graph(Digraph):
    def __init__(self, edge_list=None):
        super().__init__(edge_list=edge_list)

    def add_edge(self, src, dest, weight=1, capacity=float('inf')):
        super().add_edge(src, dest, weight, capacity)

    # Method to check if a given cycle is a Hamiltonian cycle
    def is_hamiltonian_cycle(self, cycle):
        if not cycle or len(cycle) < 2 or cycle[0] != cycle[-1]:
            return False  # Not a closed cycle

        visited_nodes = set()

        # Check if the cycle visits every node exactly once
        for i in range(len(cycle) - 1):
            node = cycle[i]
            next_node = cycle[i + 1]
            if node in visited_nodes:
                return False  # Node is visited more than once
            if next_node not in self.adj_list[node]:
                return False  # No edge between consecutive nodes
            visited_nodes.add(node)

        # The last node should be the same as the first node, which closes the cycle
        if len(visited_nodes) != len(self.adj_list):
            return False  # Not all nodes are visited

        return True

    def find_eulerian_circuit(self):
        if not self.is_eulerian():
            return None

        adj_list = {v: adj.copy() for v, adj in self.adj_list.items()}
        circuit = []
        current_path = [next(iter(adj_list))]  # Start from an arbitrary vertex

        while current_path:
            current_vertex = current_path[-1]
            if adj_list[current_vertex]:  # If there are unvisited edges
                next_vertex = adj_list[current_vertex].pop()
                adj_list[next_vertex].remove(current_vertex)
                current_path.append(next_vertex)
            else:
                circuit.append(current_path.pop())

        return circuit[::-1]  # Return in the correct order

    def is_eulerian(self):
        # Check if all vertices with nonzero degree are connected
        def is_connected():
            start_vertex = next(iter(self.adj_list))
            visited = set()
            stack = [start_vertex]
            while stack:
                vertex = stack.pop()
                if vertex not in visited:
                    visited.add(vertex)
                    stack.extend(neighbor for neighbor in self.adj_list[vertex] if neighbor not in visited)
            for vertex in self.adj_list:
                if len(self.adj_list[vertex]) > 0 and vertex not in visited:
                    return False
            return True

        if not is_connected():
            return False

        for node in self.adj_list:
            if len(self.adj_list[node]) % 2 != 0:
                return False
        return True
    # Method to check if a given closed trail is an Euler Tour
    def is_euler_tour(self, trail):
        if not trail or len(trail) < 2 or trail[0] != trail[-1]:
            return False  # Not a closed trail

        edge_count = {}
        total_edges = 0

        # Count all edges in the graph
        for node in self.adj_list:
            for neighbor in self.adj_list[node]:
                if (node, neighbor) not in edge_count and (neighbor, node) not in edge_count:
                    edge_count[(node, neighbor)] = 0
                    total_edges += 1

        # Mark edges visited in the trail
        visited_edges = 0
        for i in range(len(trail) - 1):
            edge = (trail[i], trail[i + 1])
            reverse_edge = (trail[i + 1], trail[i])
            if edge in edge_count:
                if edge_count[edge] == 0:
                    edge_count[edge] = 1
                    visited_edges += 1
                else:
                    return False  # Edge is visited more than once
            elif reverse_edge in edge_count:
                if edge_count[reverse_edge] == 0:
                    edge_count[reverse_edge] = 1
                    visited_edges += 1
                else:
                    return False  # Reverse edge is visited more than once
            else:
                return False  # Edge does not exist

        # Check if all edges are visited
        return visited_edges == total_edges
    # add a function, that returns the line graph (as new Graph object)
    def line_graph(self):
        line_graph = Graph()
        edge_to_node = {}
        edge_index = 0

        # Add the edges of the normal graph as nodes of the line graph
        for src in self.adj_list:
            for dest in self.adj_list[src]:
                if (src, dest) not in edge_to_node and (dest, src) not in edge_to_node:
                    edge_to_node[(src, dest)] = f'e{edge_index}'
                    line_graph.add_vertex(f'e{edge_index}')
                    edge_index += 1

        # For each pair of nodes in the original graph, check if they share a common neighbor
        for (src1, dest1) in edge_to_node:
            for (src2, dest2) in edge_to_node:
                if (src1, dest1) != (src2, dest2):
                    # Check if the edges share a common node
                    if src1 == src2 or src1 == dest2 or dest1 == src2 or dest1 == dest2:
                        line_graph.add_edge(edge_to_node[(src1, dest1)], edge_to_node[(src2, dest2)])

        return line_graph

    def delete_edge(self, src, dest):
        super().delete_edge(src, dest)
        super().delete_edge(dest, src)

    def degree(self, node):
        if node in self.adj_list:
            return len(self.adj_list[node])
        return 0

    def count_nodes_with_odd_degree(self):
        return sum(1 for node in self.adj_list if self.degree(node) % 2 != 0)

    def is_regular(self):
        if not self.adj_list:
            return True
        degrees = [self.degree(node) for node in self.adj_list]
        return all(degree == degrees[0] for degree in degrees)

    def is_path(self, sequence):
        if len(sequence) < 2:
            return False
        for i in range(len(sequence) - 1):
            if sequence[i + 1] not in self.adj_list[sequence[i]]:
                return False
        # Check if all nodes are unique, except possibly first and last in case of a cycle
        return len(sequence) == len(set(sequence)) or (sequence[0] == sequence[-1] and len(sequence) - 1 == len(set(sequence[:-1])))

    def is_cycle(self, sequence):
        if len(sequence) < 3 or sequence[0] != sequence[-1]:
            return False
        return self.is_path(sequence)


# Erstellen eines Graphen mit Graphviz
def visualize_graph(pgraph):
    # Check if the graph is directed or not
    if isinstance(pgraph, Digraph):
        dot = graphviz.Digraph()
    else:
        dot = graphviz.Graph()

    for src in pgraph.adj_list:
        dot.node(str(src))
        for dest in pgraph.adj_list[src]:
            if isinstance(pgraph, Digraph):
                # In a directed graph, we just add the edge with weight
                dot.edge(str(src), str(dest), label=str(pgraph.weights[(src, dest)]))
            else:
                # In an undirected graph, we check if the reverse edge was added
                if (dest, src) not in pgraph.weights or (src, dest) in pgraph.weights:
                    dot.edge(str(src), str(dest), label=str(pgraph.weights[(src, dest)]))

    dot.render('graph', format='png', cleanup=True)
    dot.view()

#print("Anzahl der Knoten mit ungeradem Grad:", graph.count_nodes_with_odd_degree())
path = [0,3,1,2,0]
cycle = [0, 3, 1, 2, 0]
#print("Ist die Sequenz ein Pfad?:", graph.is_path(path))
#print("Ist die Sequenz ein Hamiltonkreis?:", graph.is_hamiltonian_cycle(cycle))
# visualize line graph
#line_graph = graph.line_graph()
#visualize_graph(line_graph)

# Beispielcode zur Verwendung der visualize_graph-Funktion
edge_list = [(0, 1), (1, 2), (2, 0)]
graph = Graph(edge_list=edge_list)
#visualize_graph(graph)
print("Eulerian Circuit:", graph.find_eulerian_circuit())
