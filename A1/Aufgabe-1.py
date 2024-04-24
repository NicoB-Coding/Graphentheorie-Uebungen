class Digraph:
    def __init__(self, edge_list=None):
        self.adj_list = {}
        if edge_list:
            for edge in edge_list:
                self.add_edge(edge[0], edge[1])

    def add_vertex(self, vertex):
        if vertex not in self.adj_list:
            self.adj_list[vertex] = []

    def add_edge(self, src, dest):
        if src not in self.adj_list:
            self.add_vertex(src)
        if dest not in self.adj_list:
            self.add_vertex(dest)
        self.adj_list[src].append(dest)

    def delete_vertex(self, vertex):
        if vertex in self.adj_list:
            del self.adj_list[vertex]
            for src in self.adj_list:
                if vertex in self.adj_list[src]:
                    self.adj_list[src].remove(vertex)

    def delete_edge(self, src, dest):
        if src in self.adj_list and dest in self.adj_list[src]:
            self.adj_list[src].remove(dest)

    def __str__(self):
        return str(self.adj_list)


class Graph(Digraph):
    def __init__(self, edge_list=None):
        super().__init__(edge_list=edge_list)

    def add_edge(self, src, dest):
        super().add_edge(src, dest)
        super().add_edge(dest, src)

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

# Erstellen eines Graphen mit einer Kantenliste
edge_list = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
graph = Graph(edge_list=edge_list)
print("Anfangsgraph:", graph)

# Hinzufügen einer Kante und eines Knotens
graph.add_edge(5, 6)  # Fügt Knoten 6 hinzu und verbindet ihn mit Knoten 5
graph.add_vertex(7)  # Fügt einen isolierten Knoten hinzu
print("Graph nach Hinzufügen von Kante und Knoten:", graph)

# Löschen einer Kante und eines Knotens
graph.delete_edge(0, 1)
graph.delete_vertex(7)
print("Graph nach Löschen von Kante und Knoten:", graph)

# Prüfen, ob der Graph regulär ist
print("Ist der Graph regulär?:", graph.is_regular())

# Zählen der Knoten mit ungeradem Grad
print("Anzahl der Knoten mit ungeradem Grad:", graph.count_nodes_with_odd_degree())


# Definieren eines Pfades und eines Zyklus
path = [3, 4, 5, 6]
cycle = [0, 1, 2, 0]

# Überprüfen, ob es sich um einen Pfad handelt
print("Ist die Sequenz ein Pfad?:", graph.is_path(path))

# Überprüfen, ob es sich um einen Zyklus handelt
print("Ist die Sequenz ein Zyklus?:", graph.is_cycle(cycle))


