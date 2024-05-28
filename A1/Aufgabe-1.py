# graphviz nutzen um Graphen zu visualisieren
import graphviz
import numpy as np
import heapq
from collections import deque

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

    # Method to compute the eigenvalue centrality of the graph
    def eigenvalue_centrality(self):
        # Create the adjacency matrix
        nodes = list(self.adj_list.keys())
        n = len(nodes)
        adj_matrix = np.zeros((n, n))

        for i, node in enumerate(nodes):
            for neighbor in self.adj_list[node]:
                j = nodes.index(neighbor)
                adj_matrix[i, j] = 1

        # Compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(adj_matrix)

        # Find the eigenvector corresponding to the largest eigenvalue
        largest_eigenvalue_index = np.argmax(eigenvalues)
        centrality_vector = eigenvectors[:, largest_eigenvalue_index]

        # Normalize the centrality vector
        centrality_vector = np.abs(centrality_vector)
        centrality_vector /= centrality_vector.sum()

        # Map the centrality values back to the nodes
        centrality = {nodes[i]: centrality_vector[i] for i in range(n)}

        return centrality
    # Method to compute the shortest paths using Dijkstra's algorithm
    def dijkstra(self, start_node):
        # Initialisieren von Distanzen und dem Prioritätswarteschlange
        distances = {node: float('inf') for node in self.adj_list}
        distances[start_node] = 0
        priority_queue = [(0, start_node)]

        # Verarbeitete Knoten
        visited = set()

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            if current_node in visited:
                continue

            visited.add(current_node)
            # Überprüfen der Nachbarn
            for neighbor in self.adj_list[current_node]:
                weight = self.weights[(current_node, neighbor)]
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))

        return distances

    def floyd_warshall(self):
        # Initialisierung der Distanz- und Vorgängermatrix
        nodes = list(self.adj_list.keys())
        dist = {i: {j: float('inf') for j in nodes} for i in nodes}
        next_node = {i: {j: None for j in nodes} for i in nodes}

        for node in nodes:
            dist[node][node] = 0
        for (src, dest), weight in self.weights.items():
            dist[src][dest] = weight
            next_node[src][dest] = dest

        # Floyd-Warshall Algorithmus
        for k in nodes:
            for i in nodes:
                for j in nodes:
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_node[i][j] = next_node[i][k]

        # Überprüfung auf negative Zyklen
        for node in nodes:
            if dist[node][node] < 0:
                return self.find_negative_cycle(next_node, node)

        return dist, next_node

    def find_negative_cycle(self, next_node, start_node):
        # Finden eines negativen Zyklus unter Verwendung der Vorgängermatrix
        cycle = []
        visited = {node: False for node in next_node}
        current_node = start_node

        while not visited[current_node]:
            visited[current_node] = True
            current_node = next_node[current_node][start_node]

        cycle_start = current_node
        cycle.append(cycle_start)
        current_node = next_node[current_node][start_node]

        while current_node != cycle_start:
            cycle.append(current_node)
            current_node = next_node[current_node][start_node]

        cycle.append(cycle_start)
        return cycle
    def closeness_centrality(self, dist):
        closeness = {}
        for v in dist:
            total_distance = sum(dist[u][v] for u in dist if u != v)
            if total_distance > 0:
                closeness[v] = 1 / total_distance
            else:
                closeness[v] = 0
        return closeness

    def betweenness_centrality(self, dist, next_node):
        betweenness = {v: 0 for v in dist}
        for s in dist:
            for t in dist:
                if s != t:
                    path_count = self.count_paths(s, t, next_node)
                    for v in dist:
                        if v != s and v != t:
                            path_through_v = self.count_paths(s, v, next_node) * self.count_paths(v, t, next_node)
                            betweenness[v] += path_through_v / path_count
        return betweenness

    def count_paths(self, s, t, next_node):
        if next_node[s][t] is None:
            return 0
        path = []
        current_node = s
        while current_node != t:
            path.append(current_node)
            current_node = next_node[current_node][t]
            if current_node is None:
                return 0
        path.append(t)
        return 1

    def bfs_minimum_spanning_tree(self, start_node):
        visited = set()
        queue = deque([start_node])
        spanning_tree_edges = []

        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                for neighbor in self.adj_list[node]:
                    if neighbor not in visited:
                        spanning_tree_edges.append((node, neighbor))
                        queue.append(neighbor)

        return spanning_tree_edges

    def bfs_girth(self, start_node):
        visited = {start_node: 0}
        queue = deque([(start_node, -1)])
        min_cycle_length = float('inf')

        while queue:
            current_node, parent = queue.popleft()

            for neighbor in self.adj_list[current_node]:
                if neighbor == start_node and parent == start_node:
                    return 2
                if neighbor not in visited:
                    visited[neighbor] = visited[current_node] + 1
                    queue.append((neighbor, current_node))
                elif neighbor != parent:
                    # Found a cycle
                    cycle_length = visited[current_node] + visited[neighbor] + 1
                    min_cycle_length = min(min_cycle_length, cycle_length)

        return min_cycle_length

    def compute_girth(self):
        min_girth = float('inf')
        for node in self.adj_list:
            current_girth = self.bfs_girth(node)
            if current_girth < min_girth:
                min_girth = current_girth

        return min_girth if min_girth != float('inf') else -1  # -1 indicates no cycle

    def prim_jarnik(self, start_vertex):
        if start_vertex not in self.adj_list:
            return None

        mst = Graph()
        total_cost = 0
        visited = set([start_vertex])
        edges = [(weight, src, dest) for (src, dest), weight in self.weights.items() if src == start_vertex]
        heapq.heapify(edges)

        while edges:
            weight, src, dest = heapq.heappop(edges)
            if dest in visited:
                continue

            mst.add_edge(src, dest, weight)
            total_cost += weight
            visited.add(dest)

            for next_dest in self.adj_list[dest]:
                if next_dest not in visited:
                    heapq.heappush(edges, (self.weights[(dest, next_dest)], dest, next_dest))

        return total_cost, mst

    def kruskal(self):
        parent = {}
        rank = {}

        def find(vertex):
            if parent[vertex] != vertex:
                parent[vertex] = find(parent[vertex])
            return parent[vertex]

        def union(vertex1, vertex2):
            root1 = find(vertex1)
            root2 = find(vertex2)
            if root1 != root2:
                if rank[root1] > rank[root2]:
                    parent[root2] = root1
                else:
                    parent[root1] = root2
                    if rank[root1] == rank[root2]:
                        rank[root2] += 1

        mst = Graph()
        edges = sorted(self.weights.items(), key=lambda item: item[1])

        for vertex in self.adj_list:
            parent[vertex] = vertex
            rank[vertex] = 0

        total_cost = 0

        for (src, dest), weight in edges:
            if find(src) != find(dest):
                union(src, dest)
                mst.add_edge(src, dest, weight)
                total_cost += weight

        return total_cost, mst

    def boruvka(self):
        parent = {}
        rank = {}

        def find(vertex):
            if parent[vertex] != vertex:
                parent[vertex] = find(parent[vertex])
            return parent[vertex]

        def union(vertex1, vertex2):
            root1 = find(vertex1)
            root2 = find(vertex2)
            if root1 != root2:
                if rank[root1] > rank[root2]:
                    parent[root2] = root1
                else:
                    parent[root1] = root2
                    if rank[root1] == rank[root2]:
                        rank[root2] += 1

        mst = Graph()
        total_cost = 0

        for vertex in self.adj_list:
            parent[vertex] = vertex
            rank[vertex] = 0

        num_components = len(self.adj_list)

        while num_components > 1:
            cheapest = {}
            for vertex in self.adj_list:
                for neighbor in self.adj_list[vertex]:
                    root1 = find(vertex)
                    root2 = find(neighbor)
                    if root1 != root2:
                        weight = self.weights[(vertex, neighbor)]
                        if root1 not in cheapest or cheapest[root1][0] > weight:
                            cheapest[root1] = (weight, vertex, neighbor)
                        if root2 not in cheapest or cheapest[root2][0] > weight:
                            cheapest[root2] = (weight, vertex, neighbor)

            for root in cheapest:
                weight, src, dest = cheapest[root]
                if find(src) != find(dest):
                    union(src, dest)
                    mst.add_edge(src, dest, weight)
                    total_cost += weight
                    num_components -= 1

        return total_cost, mst

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

# Beispielcode zur Verwendung der visualize_graph-Funktion
edge_list = [
    (0, 1, 2), (1,2, 3), (3, 1,6), (3, 2,8),
    (2, 4,7), (4, 5,1), (4,2,4), (1,3,5)
]
graph = Graph(edge_list=edge_list)
visualize_graph(graph)
start_node = 0
mst_cost, mst_edges = graph.boruvka()
print(f"Minimum Spanning Tree Cost: {mst_cost}")
print(mst_edges)
