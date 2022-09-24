import numpy as np
from graph_tool.all import *
from auxiliarymethods.auxiliary_methods import read_txt

def wl_simple(graph_db, h, degree=False, uniform=False, gram_matrix=True):
    # Create one empty feature vector for each graph
    feature_vectors = []
    for _ in graph_db:
        feature_vectors.append(np.zeros(0, dtype=np.float64))

    if degree:
        for g in graph_db:
            g.vp.nl = g.new_vertex_property("int")
            for v in g.vertices():
                g.vp.nl[v] = v.out_degree()

    if uniform:
        for g in graph_db:
            g.vp.nl = g.new_vertex_property("int")
            for v in g.vertices():
                g.vp.nl[v] = 1

    offset = 0
    graph_indices = []

    for g in graph_db:
        graph_indices.append((offset, offset + g.num_vertices()))
        offset += g.num_vertices()

    colors = []
    for g in graph_db:
        for v in g.vertices():
            colors.append(g.vp.nl[v])

    _, colors = np.unique(colors, return_inverse=True)
    max_all = int(np.amax(colors) + 1)

    feature_vectors = [
        np.concatenate((feature_vectors[i], np.bincount(colors[index[0]:index[1]], minlength=max_all))) for
        i, index in enumerate(graph_indices)]

    i = 0
    while i < h:
        colors = []

        for g in graph_db:
            for v in g.vertices():
                neighbours = []
                for w in g.get_all_neighbors(v):
                    neighbours.append(g.vp.nl[w])

                neighbours.sort()
                neighbours.append(g.vp.nl[v])
                colors.append(hash(tuple(neighbours)))

        _, colors = np.unique(colors, return_inverse=True)

        # Assign new colors to vertices.
        q = 0
        cl = []
        for g in graph_db:
            for v in g.vertices():
                g.vp.nl[v] = colors[q]
                q += 1
                cl.append(g.vp.nl[v])

        max_all = int(np.amax(colors) + 1)

        feature_vectors = [
            np.concatenate((feature_vectors[i], np.bincount(colors[index[0]:index[1]], minlength=max_all))) for
            i, index in enumerate(graph_indices)]

        i += 1

    feature_vectors = np.array(feature_vectors)

    if gram_matrix:
        return np.dot(feature_vectors, feature_vectors.transpose())
    else:
        return feature_vectors


def wl_simple_color_count(graph_db, h, degree=False, uniform=False):

    # Create one empty feature vector for each graph.
    feature_vectors = []
    for _ in graph_db:
        feature_vectors.append(np.zeros(0, dtype=np.float64))

    if degree:
        for g in graph_db:
            g.vp.nl = g.new_vertex_property("int")
            for v in g.vertices():
                g.vp.nl[v] = v.out_degree()

    if uniform:
        for g in graph_db:
            g.vp.nl = g.new_vertex_property("int")
            for v in g.vertices():
                g.vp.nl[v] = 1

    offset = 0
    graph_indices = []

    for g in graph_db:
        graph_indices.append((offset, offset + g.num_vertices()))
        offset += g.num_vertices()

    colors = []
    for g in graph_db:
        for v in g.vertices():
            colors.append(g.vp.nl[v])

    _, colors = np.unique(colors, return_inverse=True)
    max_all = int(np.amax(colors) + 1)

    i = 0

    color_counts = []
    feature_vectors = np.array([
        np.concatenate((feature_vectors[i], np.bincount(colors[index[0]:index[1]], minlength=max_all))) for
        i, index in enumerate(graph_indices)])

    color_counts.append(feature_vectors.shape[1])
    while i < h:
        colors = []

        for g in graph_db:
            for v in g.vertices():
                neighbours = []
                for w in g.get_all_neighbors(v):
                    neighbours.append(g.vp.nl[w])

                neighbours.sort()
                neighbours.append(g.vp.nl[v])
                colors.append(hash(tuple(neighbours)))

        _, colors = np.unique(colors, return_inverse=True)

        # Assign new colors to vertices.
        q = 0
        for g in graph_db:
            for v in g.vertices():
                g.vp.nl[v] = colors[q]
                q += 1

        max_all = int(np.amax(colors) + 1)

        feature_vectors = np.array([
            np.bincount(colors[index[0]:index[1]], minlength=max_all) for
            i, index in enumerate(graph_indices)])

        i += 1
        color_counts.append(feature_vectors.shape[1])

    return color_counts

