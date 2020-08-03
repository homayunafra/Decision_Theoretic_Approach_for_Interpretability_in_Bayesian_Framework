# -*- coding: utf-8 -*-
import numpy as np
import igraph as ig


class BasicNode:  # should be interfaced to a graph object
    def __init__(self, labels=None):
        self.incident_edges = set([])
        self.incident_outward_edges = set([])
        self.incident_inward_edges = set([])
        if labels is None:
            self.labels = set([])
        else:
            self.labels = labels

    def add_edge(self, Edge):
        if self in Edge.ends:
            self.incident_edges.add(Edge)
            if Edge.source() == self:
                self.incident_outward_edges.add(Edge)
            else:
                self.incident_inward_edges.add(Edge)
        else:
            print('Cannot add edge to vertex, vertex not in ends.')

    def remove_edge(self, Edge):
        self.incident_edges.discard(Edge)

    def add_label(self, label):
        self.labels.add(label)

    def remove_label(self, label):
        self.labels.discard(label)


class BasicEdge:  # should be interfaced to from a graph object
    def __init__(self, ends=[], labels=None):
        self.ends = ends
        if labels is None:
            self.labels = set([])
        else:
            self.labels = labels

    def source(self):
        return self.ends[0]

    def target(self):
        return self.ends[1]

    def add_label(self, label):
        self.labels.add(label)

    def remove_label(self, label):
        self.labels.discard(label)


def update_up(class_method):
    def inner(self, *args, **kwargs):
        method_name = class_method
        class_method(self, *args, **kwargs)
        for Supergraph in self.supergraphs:
            getattr(Supergraph, method_name)(*args, **kwargs)

    return inner


def update_up_down(class_method):
    def inner(self, *args, **kwargs):
        method_name = class_method
        if class_method(self, *args, **kwargs):
            for Supergraph in self.supergraphs:
                getattr(Supergraph, method_name)(*args, **kwargs)
            for Subgraph in self.subgraphs:
                getattr(Subgraph, method_name)(*args, **kwargs)

    return inner


class Graph(object):
    def __init__(self, subgraphs=None, supergraphs=None, vertices=None, edges=None, Vertex=BasicNode, Edge=BasicEdge):

        if edges == None:
            edges = []
        self.edges = set(edges)
        if vertices == None:
            vertices = []
        self.vertices = vertices
        if subgraphs == None:
            subgraphs = []
        self.subgraphs = set(subgraphs)
        if supergraphs == None:
            supergraphs = []
        self.supergraphs = set(supergraphs)
        for Supergraph in supergraphs:
            Supergraph.add_subgraph(self)
        for Subgraph in subgraphs:
            Subgraph.add_supergraph(self)
        self.vertex_dict = {}
        self.edges_dict = {}
        self.Vertex = Vertex
        self.Edge = Edge

    @update_up
    def create_vertex(self):
        self.vertices.append(self.Vertex())

    def create_vertices(self, no_create):
        for i in range(no_create):
            self.create_vertex()

    @update_up
    def add_vertex(self, Vertex):
        if Vertex in self.vertices:
            return
        self.vertices.append(Vertex)

    @update_up
    def create_edge(self, ends):
        NewEdge = self.Edge(ends=ends)
        self.edges.add(NewEdge)
        for Vertex in ends:
            Vertex.add_edge(NewEdge)

    @update_up_down
    def remove_edge(self, Edge):
        if not Edge in self.edges:
            return False
        self.edges.discard(Edge)
        return True

    def get_incident_edges(self, Vertex):
        incident_edges = Vertex.incident_edges & self.edges
        return incident_edges

    @update_up_down
    def remove_vertex(self, Vertex):
        if Vertex not in self.vertices:
            return False
        edges_to_remove = self.get_incident_edges(Vertex)
        for Edge in edges_to_remove:
            self.remove_edge(Edge)
        self.vertices.remove(Vertex)
        return True

    def get_degree(self, Vertex):
        return len(self.get_incident_edges(Vertex))

    def get_number_vertices(self):
        return len(self.vertices)

    def get_adjacency_matrix(self):
        adj_list = [self.get_adjacency_list_of_vertex(Vertex) for Vertex in self.vertices]
        # adj_list = map(lambda x: self.get_adjacency_list_of_vertex(x), self.vertices)
        adj_mat = np.array(adj_list)
        return adj_mat

    def get_adjacency_matrix_as_list(self):
        return self.get_adjacency_matrix().tolist()

    def add_subgraph(self, Subgraph):
        self.subgraphs.add(Subgraph)

    def add_supergraph(self, Supergraph):
        self.supergraphs.add(Supergraph)

    def get_incident_outward_edges(self, Vertex):
        return (Vertex.incident_outward_edges & self.edges)

    def add_vertex_label(self, vertex, label):
        self.vertex_dict[label] = vertex
        vertex.add_label(label)

    def get_vertex(self, label):
        if label in self.vertex_dict.keys():
            return self.vertex_dict[label]
        else:
            return None

    def remove_vertex_label(self, label):
        vertex = self.vertex_dict.pop(label, 'Not Found')
        if vertex == 'Not Found':
            return
        else:
            vertex.remove_label(label)

    def get_incident_outward_edges(self, Vertex):
        return (Vertex.incident_outward_edges & self.edges)

    def get_adjacency_list_of_vertex(self, Vertex):
        N = self.get_number_vertices()
        adj_list = [0 for x in range(N)]
        incident_edges = self.get_incident_outward_edges(Vertex)
        for Edge in incident_edges:
            target = Edge.target()
            index = self.vertices.index(target)
            adj_list[index] += 1
        return adj_list

    def set_adjacency_matrix(self, adj_mat):
        shape = np.shape(adj_mat)
        if shape[0] != shape[1]:
            print('Wrong shape, expecting square matrix.')
            return
        n = shape[0]
        self.vertices = []
        self.edges = []
        self.create_vertices(n)
        for row, col in range(n):
            no_edges = adj_mat[row, col]
            Source = self.vertices[row]
            Target = self.vertices[col]
            for Edge in range(no_edges):
                self.create_edge(ends=[Source, Target])

    def plot(self):
        A = self.get_adjacency_matrix_as_list()
        convert_to_igraph = ig.Graph.Adjacency(A)
        ig.plot(convert_to_igraph)


def return_tree_class(directed):
    class Tree(Graph, object):
        def __init__(self, N=0, **kwargs):
            super().__init__(**kwargs)
            self.leaves = set([])
            self.find_leaves()
            self.N = N

        def is_leaf(self, vertex):

            if self.get_degree(vertex) == 1:
                return True
            elif self.get_number_vertices() == 1:
                return True
            else:
                return False

        def set_root(self, vertex):
            if vertex in self.vertices:
                self.remove_vertex_label('Root')
                self.add_vertex_label(vertex, label='Root')

        def get_root(self):
            return self.get_vertex('Root')

        def find_leaves(self):
            self.leaves = set(filter(self.is_leaf, self.vertices))
            return [leaf for leaf in self.leaves]

        def split_vertex(self, vertex):
            if vertex in self.leaves:

                children = [self.Vertex() for i in range(self.N)]
                vertex.children = children
                self.leaves.discard(vertex)
                for Child in children:
                    self.add_vertex(Child)
                    self.leaves.add(Child)
                    Child.parent = vertex
                    self.create_edge(ends=[vertex, Child])

        def fuse_vertex(self, vertex):
            self.leaves.add(vertex)
            try:
                children = vertex.children
            except AttributeError:
                return
            if children == None:
                return

            for child in children:
                self.fuse_vertex(child)
                self.leaves.discard(child)
                self.remove_vertex(child)
                child.parent = None
                vertex.children = None

    return Tree

