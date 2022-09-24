from itertools import chain, combinations

import graph_tool as gt
import networkx as nx
from graph_tool.all import *


def CFI(k):
    # Changed for folklore WL graphs.
    # TODO
    K = nx.complete_graph(k + 1)

    ## graph 1

    G = nx.Graph()

    for i, e in enumerate(K.edges):
        G.add_node((e, 0), data=str("e") + str(i))
        G.add_node((e, 1), data=str("e") + str(i))
        G.add_edge((e, 1), (e, 0))

    for u in K:
        for S in subsetslist(0, K, u):
            G.add_node((u, S), data=str(u))

            for e in incidentedges(K, u):
                G.add_edge((u, S), (e, int(e in S)))

    ## graph 2
    H = nx.Graph()

    for i, e in enumerate(K.edges):
        H.add_node((e, 0), data=str("e") + str(i))
        H.add_node((e, 1), data=str("e") + str(i))
        H.add_edge((e, 1), (e, 0))

    for u in K:
        parity = int(u == 0)  ## vertex 0 in K, the "odd" one out
        for S in subsetslist(parity, K, u):
            H.add_node((u, S), data=str(u))
            for e in incidentedges(K, u):
                H.add_edge((u, S), (e, int(e in S)))

    G = nx.convert_node_labels_to_integers(G)
    H = nx.convert_node_labels_to_integers(H)

    return (G, H)


def attach_gadget(F, u, K):
    for x in K:
        F.add_node(str(u) + '-' + str(x), data=2)
        F.add_edge(str(u) + '-' + str(x), u)

    for e in K.edges:
        (x, y) = e
        F.add_edge(str(u) + '-' + str(x), str(u) + '-' + str(y))

    return F

def CFI_pebble(k):
    Q = 6 * k + 1

    G = nx.cycle_graph(Q)
    H = nx.cycle_graph(Q)

    for v in range(Q):
        G.nodes[v]['data'] = 1
        H.nodes[v]['data'] = 1

    Gk, Hk = CFI(k)

    for v in range(Q):
        if v <= int(Q / 2) + 1:
            G = attach_gadget(G, v, Gk)
        else:
            G = attach_gadget(G, v, Hk)

    for v in range(Q):
        if v <= int(Q / 2):  ## crucial diff  G/H
            H = attach_gadget(H, v, Gk)
        else:
            H = attach_gadget(H, v, Hk)

    return G, H


def CFI_blow(k):
    # Changed for folklore WL graphs.
    K = nx.complete_graph(k + 1)

    ## graph 1

    G = nx.Graph()

    delta = 0#v 3*k+2

    for i, e in enumerate(K.edges):
        G.add_node((e, 0), data=str("e") + str(i))
        G.add_node((e, 1), data=str("e") + str(i))
        G.add_edge((e, 1), (e, 0))

    for u in K:
        for S in subsetslist(0, K, u):
            G.add_node((u, S), data=str(u))

            for e in incidentedges(K, u):
                #G.add_edge((u, S), (e, int(e in S)))

                G.add_node(0, data="-")
                G.add_edge((u, S), 0)
                last = 0
                for i in range(1, delta):
                    G.add_node(i, data="-")
                    G.add_edge(last, i)
                    last = i

                G.add_edge(last, (e, int(e in S)))

    ## graph 2
    H = nx.Graph()

    for i, e in enumerate(K.edges):
        H.add_node((e, 0), data=str("e") + str(i))
        H.add_node((e, 1), data=str("e") + str(i))
        H.add_edge((e, 1), (e, 0))

    for u in K:
        parity = int(u == 0)  ## vertex 0 in K, the "odd" one out
        for S in subsetslist(parity, K, u):
            H.add_node((u, S), data=str(u))
            for e in incidentedges(K, u):
                #H.add_edge((u, S), (e, int(e in S)))

                H.add_node(0, data="-")
                H.add_edge((u, S), 0)

                last = 0
                for i in range(1, delta):
                    H.add_node(i, data="-")
                    H.add_edge(last, i)
                    last = i

                H.add_edge(last, (e, int(e in S)))

    G = nx.convert_node_labels_to_integers(G)
    H = nx.convert_node_labels_to_integers(H)

    return (G, H)


## list of edges incident to a vertex,
## where each edge (i,j) satisfies i < j

def incidentedges(K, u):
    return [tuple(sorted(e)) for e in K.edges(u)]


## generate all edge subsets of odd/even cardinality
## set parameter "parity" 0/1 for odd/even sets resp.

def subsetslist(parity, K, u):
    oddsets = set()
    evensets = set()
    for s in list(powerset(incidentedges(K, u))):
        if (len(s) % 2 == 0):
            evensets.add(frozenset(s))
        else:
            oddsets.add(frozenset(s))
    if parity == 0:
        return evensets
    else:
        return oddsets


## generate all subsets of a set
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def get_prop_type(value, key=None):
    """
    Performs typing and value conversion for the graph_tool PropertyMap class.
    If a key is provided, it also ensures the key is in a format that can be
    used with the PropertyMap. Returns a tuple, (type name, value, key)
    """
    if isinstance(key, unicode):
        # Encode the key as ASCII
        key = key.encode('ascii', errors='replace')

    # Deal with the value
    if isinstance(value, bool):
        tname = 'bool'

    elif isinstance(value, int):
        tname = 'float'
        value = float(value)

    elif isinstance(value, float):
        tname = 'float'

    elif isinstance(value, unicode):
        tname = 'string'
        value = value.encode('ascii', errors='replace')

    elif isinstance(value, dict):
        tname = 'object'

    else:
        tname = 'string'
        value = str(value)

    return tname, value, key


def networkx_to_gt(nxG):
    gtG = gt.Graph(directed=False)

    # Phase 2: Actually add all the nodes and vertices with their properties
    # Add the nodes
    vertices = {}  # vertex mapping for tracking edges later

    data = gtG.new_vertex_property("int")

    for node in nxG.nodes():
        # Create the vertex and annotate for our edges later
        v = gtG.add_vertex()
        #data[v] = nxG.nodes[node]["data"]

        vertices[node] = v

    # Add the edges
    for src, dst in nxG.edges():
        # Look up the vertex structs from our vertices mapping and add edge.
        e = gtG.add_edge(vertices[src], vertices[dst])

    # Done, finally!

    gtG.vp.data = data
    return gtG


def create_gaurav_graphs(k):
    g_1, g_2 = CFI(k)
    # print(g_1.number_of_nodes(), g_2.number_of_nodes())
    # print(g_1.number_of_edges(), g_2.number_of_edges())

    g_1 = networkx_to_gt(g_1)
    g_2 = networkx_to_gt(g_2)

    # g_1.vp.nl = g_1.new_vertex_property("int")
    # for v in g_1.vertices():
    #     if g_1.vp.data[v][0] == "e":
    #         g_1.vp.nl[v] = int(g_1.vp.data[v][1:]) + g_1.num_vertices()
    #     elif g_1.vp.data[v] == "-":
    #         g_1.vp.nl[v] = -1
    #     else:
    #         g_1.vp.nl[v] = int(g_1.vp.data[v])
    #
    # g_2.vp.nl = g_2.new_vertex_property("int")
    # for v in g_2.vertices():
    #     if g_2.vp.data[v][0] == "e":
    #         g_2.vp.nl[v] = int(g_2.vp.data[v][1:]) + g_2.num_vertices()
    #     elif g_2.vp.data[v] == "-":
    #         g_2.vp.nl[v] = -1
    #     else:
    #         g_2.vp.nl[v] = int(g_2.vp.data[v])

    return (g_1, g_2)


def create_gaurav_pepple_graphs(k):
    g_1, g_2 = CFI_pebble(k)

    g_1 = networkx_to_gt(g_1)
    g_2 = networkx_to_gt(g_2)

    return (g_1, g_2)



def create_blow_up(g_1, k):
    remove_list = []

    delta = 3 * k + 2

    # Collect edges to remove and replace.
    for e in g_1.edges():
        remove_list.append(e)

    # Add blow-up vertices for each removed edge.
    for u, v in remove_list:

        new_color = hash(([g_1.vp.nl[u], g_1.vp.nl[v]].sort()))

        new_node = g_1.add_vertex()
        g_1.vp.nl[new_node] = new_color

        g_1.add_edge(u, new_node)

        last = new_node
        for _ in range(0,delta+1):
            new_node = g_1.add_vertex()
            g_1.vp.nl[new_node] = new_color

            g_1.add_edge(last, new_node)
            last = new_node


        g_1.add_edge(last, u)


    for e in remove_list:
        g_1.remove_edge(e)

    return g_1


def create_cycle_pair(k):
    # One large cycle.
    cycle_1 = Graph(directed=False)

    for i in range(0, 2 * k):
        cycle_1.add_vertex()

    for i in range(0, 2 * k):
        cycle_1.add_edge(i, (i + 1) % (2 * k))

    # 2 smaller cycles.
    c_1 = Graph(directed=False)

    for i in range(0, k):
        c_1.add_vertex()

    for i in range(0, k):
        c_1.add_edge(i, (i + 1) % k)

    # Second cycle.
    c_2 = Graph(directed=False)
    for i in range(0, k):
        c_2.add_vertex()

    for i in range(0, k):
        c_2.add_edge(i, (i + 1) % k)

    cycle_2 = graph_union(c_1, c_2)

    return (cycle_1, cycle_2)


# Create cycle counter examples.
def create_pair(k):
    # Graph 1.
    # First cycle.
    c_1 = Graph(directed=False)

    for i in range(0, k + 1):
        c_1.add_vertex()

    for i in range(0, k + 1):
        c_1.add_edge(i, (i + 1) % (k + 1))

    # Second cycle.
    c_2 = Graph(directed=False)
    for i in range(0, k + 1):
        c_2.add_vertex()

    for i in range(0, k + 1):
        c_2.add_edge(i, (i + 1) % (k + 1))

    cycle_union_1 = graph_union(c_1, c_2)
    cycle_union_1.add_edge(0, k + 1)

    c_3 = Graph(directed=False)
    for i in range(0, k + 2):
        c_3.add_vertex()

    for i in range(0, k + 2):
        c_3.add_edge(i, (i + 1) % (k + 2))

    c_4 = Graph(directed=False)
    for i in range(0, k + 2):
        c_4.add_vertex()

    for i in range(0, k + 1):
        c_4.add_edge(i, (i + 1))

    merge = c_4.new_vertex_property("int")
    for v in c_4.vertices():
        merge[v] = -1

    merge[0] = 0
    merge[k + 1] = 1

    cycle_union_2 = graph_union(c_3, c_4, intersection=merge)

    # Draw for visual inspection.
    position = sfdp_layout(cycle_union_1)
    graph_draw(cycle_union_1, pos=position, output="g_1.pdf")
    position = sfdp_layout(cycle_union_2)
    graph_draw(cycle_union_2, pos=position, output="g_2.pdf")

    return (cycle_union_1, cycle_union_2)


def reconstruction(g1, k):
    for i in range(k):
        print(g1.nodes[(i * 3)]["data"])
        g1.remove_node((i * 3))

    return g1


def edge_reconstruction(g1, k):
    c = 0
    for u in g1.edges():
        if g1.edges[u]["data"][0:3] == "0 0":
            print(g1.edges[u]["data"])
            g1.remove_edge(u[0], u[1])
            c += 1

    return g1


def edge_reconstruction_(g1, k):
    c = 0
    for u in g1.edges():

        if g1.edges[u]["data"][0:3] == "0 1":

            if c < k:
                print(g1.edges[u]["data"])
                g1.remove_edge(u[0], u[1])
                c += 1

    return g1
