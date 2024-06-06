# graphviz nutzen um Graphen zu visualisieren
import graphviz
import numpy as np
import pandas as pd
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
        self.flow = {}  # Dictionary to track the flow of each edge
        super().__init__(edge_list=edge_list)

    def add_edge(self, src, dest, weight=1, capacity=float('inf')):
        super().add_edge(src, dest, weight, capacity)
        if (dest, src) not in self.capacities and capacity != float('inf'):
            super().add_edge(dest, src, weight, 0)  # Add reverse edge with 0 capacity
        self.flow[(src, dest)] = 0
        self.flow[(dest, src)] = 0

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

    def find_path(self, source, sink, path, visited):
        if source == sink:
            return path
        visited.add(source)
        for neighbor in self.adj_list[source]:
            if neighbor not in visited and self.capacities[(source, neighbor)] - self.flow[(source, neighbor)] > 0:
                result = self.find_path(neighbor, sink, path + [(source, neighbor)], visited)
                if result is not None:
                    return result
        return None

    def ford_fulkerson(self, source, sink):
        max_flow = 0
        while True:
            visited = set()
            path = self.find_path(source, sink, [], visited)
            if path is None:
                break
            flow = min(self.capacities[edge] - self.flow[edge] for edge in path)
            for u, v in path:
                self.flow[(u, v)] += flow
                self.flow[(v, u)] -= flow
            max_flow += flow
        return max_flow
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
def visualize_graph(pgraph, directed=False):
    if directed:
        dot = graphviz.Digraph()
    else:
        dot = graphviz.Graph()

    for src in pgraph.adj_list:
        dot.node(str(src))
        for dest in pgraph.adj_list[src]:
            if directed:
                dot.edge(str(src), str(dest), label=str(pgraph.weights[(src, dest)]))
            else:
                if (dest, src) not in pgraph.weights or (src, dest) in pgraph.weights:
                    dot.edge(str(src), str(dest), label=str(pgraph.weights[(src, dest)]))

    dot.render('graph', format='png', cleanup=True)
    dot.view()

def visualize_flow_network(flow_network):
    # Create a directed graph using graphviz
    dot = graphviz.Digraph()

    for src in flow_network.adj_list:
        dot.node(str(src))
        for dest in flow_network.adj_list[src]:
            # Only show edges with capacity greater than 0
            if flow_network.capacities[(src, dest)] > 0:
                # Add the edge with flow and capacity as the label
                capacity = flow_network.capacities[(src, dest)]
                current_flow = flow_network.flow[(src, dest)]
                dot.edge(str(src), str(dest), label=f"{current_flow}/{capacity}")

    dot.render('flow_network', format='png', cleanup=True)
    dot.view()


# edge_list = [
#     ('S', 'A', 1, 16),
#     ('S', 'B', 1, 13),
#     ('B', 'T', 1, 7),
#     ('A', 'B', 1, 10),
#     ('A', 'T', 1, 4),
# ]
#
# graph = Graph(edge_list)
# source, sink = 'S', 'T'
# max_flow = graph.ford_fulkerson(source, sink)
# print(f"Der maximale Fluss von {source} nach {sink} ist {max_flow}.")
# visualize_flow_network(graph)

distance_matrix_part1 = [[0.0, 56.26, 71.42, 104.88, 62.6, 110.37, 147.26, 128.92, 91.77, 106.41, 30.12, 46.22, 66.19, 102.04, 57.57, 79.39, 87.44, 54.43, 31.93, 107.59, 121.23, 87.84, 72.38, 77.29, 45.43],
                         [56.26, 0.0, 55.06, 113.97, 95.67, 135.77, 190.35, 161.31, 143.53, 72.3, 60.08, 77.69, 107.26, 137.8, 70.98, 87.31, 105.82, 34.12, 75.95, 152.23, 158.66, 61.91, 46.6, 69.21, 49.02],
                         [71.42, 55.06, 0.0, 61.79, 65.16, 90.83, 156.84, 119.42, 126.97, 38.16, 48.91, 53.72, 81.71, 101.17, 33.78, 38.2, 59.99, 22.44, 65.76, 123.63, 121.46, 17.73, 11.67, 14.98, 26.93],
                         [104.88, 113.97, 61.79, 0.0, 53.04, 37.98, 111.63, 66.95, 108.37, 80.78, 74.79, 61.01, 66.41, 60.28, 48.29, 27.34, 21.95, 79.91, 79.57, 89.61, 75.45, 67.49, 73.19, 46.86, 68.11],
                         [62.6, 95.67, 65.16, 53.04, 0.0, 47.93, 95.27, 67.6, 63.51, 100.98, 37.5, 18.41, 17.19, 42.33, 31.41, 39.09, 31.32, 68.46, 31.48, 59.16, 63.07, 81.05, 74.95, 57.47, 48.75],
                         [110.37, 135.77, 90.83, 37.98, 47.93, 0.0, 73.7, 29.53, 81.41, 116.62, 83.27, 64.63, 51.4, 26.69, 64.8, 52.69, 30.94, 103.67, 79.38, 56.03, 37.81, 100.85, 102.46, 77.24, 86.78],
                         [147.26, 190.35, 156.84, 111.63, 95.27, 73.7, 0.0, 46.69, 65.69, 187.67, 130.53, 112.73, 83.37, 55.72, 124.81, 121.15, 99.66, 163.54, 117.15, 39.72, 36.47, 169.98, 167.8, 145.43, 143.98],
                         [128.92, 161.31, 119.42, 66.95, 67.6, 29.53, 46.69, 0.0, 75.73, 146.16, 105.05, 85.91, 63.63, 27.4, 91.03, 81.58, 59.52, 130.56, 96.99, 43.81, 17.01, 130.16, 130.96, 106.28, 112.5],
                         [91.77, 143.53, 126.97, 108.37, 63.51, 81.41, 65.69, 75.73, 0.0, 163.97, 85.53, 73.57, 46.32, 55.37, 93.59, 101.93, 87.67, 124.87, 67.96, 32.78, 59.48, 143.72, 135.49, 120.78, 105.44],
                         [106.41, 72.3, 38.16, 80.78, 100.98, 116.62, 187.67, 146.16, 163.97, 0.0, 86.77, 91.45, 118.02, 132.33, 70.42, 66.7, 88.04, 52.19, 103.89, 157.55, 151.36, 20.52, 34.05, 43.66, 64.39],
                         [30.12, 60.08, 48.91, 74.79, 37.5, 83.27, 130.53, 105.05, 85.53, 86.77, 0.0, 19.15, 47.24, 79.65, 27.69, 49.39, 58.03, 40.05, 18.23, 92.15, 100.15, 66.59, 53.94, 50.68, 22.48],
                         [46.22, 77.69, 53.72, 61.01, 18.41, 64.63, 112.73, 85.91, 73.57, 91.45, 19.15, 0.0, 30.44, 60.7, 21.94, 39.3, 41.73, 52.5, 18.69, 75.29, 81.35, 70.96, 61.92, 49.89, 32.62],
                         [66.19, 107.26, 81.71, 66.41, 17.19, 51.4, 83.37, 63.63, 46.32, 118.02, 47.24, 30.44, 0.0, 36.29, 47.97, 55.9, 44.55, 82.88, 34.59, 45.04, 55.03, 97.96, 91.05, 74.6, 62.98],
                         [102.04, 137.8, 101.17, 60.28, 42.33, 26.69, 55.72, 27.4, 55.37, 132.33, 79.65, 60.7, 36.29, 0.0, 69.68, 65.65, 44.72, 108.98, 70.14, 29.75, 20.92, 114.32, 112.21, 89.75, 89.88],
                         [57.57, 70.98, 33.78, 48.29, 31.41, 64.8, 124.81, 91.03, 93.59, 70.42, 27.69, 21.94, 47.97, 69.68, 0.0, 21.85, 35.22, 39.59, 38.43, 90.25, 90.48, 50.14, 43.64, 28.01, 22.03],
                         [79.39, 87.31, 38.2, 27.34, 39.09, 52.69, 121.15, 81.58, 101.93, 66.7, 49.39, 39.3, 55.9, 65.65, 21.85, 0.0, 22.09, 53.52, 57.72, 91.34, 85.03, 48.97, 49.87, 24.71, 40.77],
                         [87.44, 105.82, 59.99, 21.95, 31.32, 30.94, 99.66, 59.52, 87.67, 88.04, 58.03, 41.73, 44.55, 44.72, 35.22, 22.09, 0.0, 73.08, 59.67, 72.22, 63.32, 70.95, 71.59, 46.8, 57.17],
                         [54.43, 34.12, 22.44, 79.91, 68.46, 103.67, 163.54, 130.56, 124.87, 52.19, 40.05, 52.5, 82.88, 108.98, 39.59, 53.52, 73.08, 0.0, 58.27, 127.38, 129.86, 35.01, 18.53, 35.47, 19.9],
                         [31.93, 75.95, 65.76, 79.57, 31.48, 79.38, 117.15, 96.99, 67.96, 103.89, 18.23, 18.69, 34.59, 70.14, 38.43, 57.72, 59.67, 58.27, 0.0, 77.85, 89.56, 83.49, 71.74, 65.15, 40.26],
                         [107.59, 152.23, 123.63, 89.61, 59.16, 56.03, 39.72, 43.81, 32.78, 157.55, 92.15, 75.29, 45.04, 29.75, 90.25, 91.34, 72.22, 127.38, 77.85, 0.0, 27.06, 138.48, 133.88, 114.06, 107.52],
                         [121.23, 158.66, 121.46, 75.45, 63.07, 37.81, 36.47, 17.01, 59.48, 151.36, 100.15, 81.35, 55.03, 20.92, 90.48, 85.03, 63.32, 129.86, 89.56, 27.06, 0.0, 133.97, 132.65, 109.51, 110.8],
                         [87.84, 61.91, 17.73, 67.49, 81.05, 100.85, 169.98, 130.16, 143.72, 20.52, 66.59, 70.96, 97.96, 114.32, 50.14, 48.97, 70.95, 35.01, 83.49, 138.48, 133.97, 0.0, 16.65, 24.59, 44.4],
                         [72.38, 46.6, 11.67, 73.19, 74.95, 102.46, 167.8, 130.96, 135.49, 34.05, 53.94, 61.92, 91.05, 112.21, 43.64, 49.87, 71.59, 18.53, 71.74, 133.88, 132.65, 16.65, 0.0, 26.34, 31.51],
                         [77.29, 69.21, 14.98, 46.86, 57.47, 77.24, 145.43, 106.28, 120.78, 43.66, 50.68, 49.89, 74.6, 89.75, 28.01, 24.71, 46.8, 35.47, 65.15, 114.06, 109.51, 24.59, 26.34, 0.0, 32.07],
                         [45.43, 49.02, 26.93, 68.11, 48.75, 86.78, 143.98, 112.5, 105.44, 64.39, 22.48, 32.62, 62.98, 89.88, 22.03, 40.77, 57.17, 19.9, 40.26, 107.52, 110.8, 44.4, 31.51, 32.07, 0.0]
                        ]



# City names
cities = ["München", "Augsburg", "Ingolstadt", "Regensburg", "Landshut", "Straubing", "Passau", "Deggendorf", "Burghausen", "Weißenburg",
          "Freising", "Moosburg a.d. Isar", "Vilsbiburg", "Landau a.d. Isar", "Mainburg", "Abensberg", "Schierling", "Schrobenhausen",
          "Erding", "Pfarrkirchen", "Osterhofen", "Eichstätt", "Neuburg a.d. Donau", "Kösching", "Pfaffenhofen a.d. Ilm"]

# Convert to edge list format
edge_list = []

for i in range(len(cities)):
    for j in range(i + 1, len(cities)):
        # does edge list already contain
        if (cities[j], cities[i], distance_matrix_part1[j][i]) not in edge_list:
            edge_list.append((cities[i], cities[j], distance_matrix_part1[i][j]))

graph = Graph(edge_list)
# prim jarnik for mst
total_cost, mst = graph.prim_jarnik("München")
print(f"Total cost of the minimum spanning tree: {total_cost}")
visualize_graph(mst)
