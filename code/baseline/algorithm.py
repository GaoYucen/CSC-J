import copy
import math
import random
import csv

RANDOM_SEED = 1108
ERROR = 1


class Vertex:
    def __init__(self, x, y, idx):
        self.x = x
        self.y = y
        self.idx = idx


class Edge:
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2
        self.length = dist(v1, v2)


class Graph:
    def __init__(self, width, height, n, random_seed=RANDOM_SEED, points=None):
        random.seed(random_seed)
        self.width = width
        self.height = height
        self.n = n
        
        if points is None:
            text_file = open("data/points.csv", "r")
            data = text_file.read().split()
            text_file.close()
            
            # Generate n vertices in a rectangle with the given width and height.
            self.vertices = [
                Vertex(float(data[idx*2]), float(data[idx*2+1]), idx)
                for idx in range(n)
            ]
        else:
            # Use provided points
            self.vertices = [
                Vertex(float(points[idx][0]), float(points[idx][1]), idx)
                for idx in range(n)
            ]

        #Fully connected graph
        self.edges = {
            (i, j): Edge(self.vertices[i], self.vertices[j])
            for i in range(n) for j in range(i + 1, n)
        }
        #Sort edges by length
        self.sorted_edges = sorted(self.edges.values(), key=lambda e: e.length)

    def edge(self, v1, v2):
        if v1.idx > v2.idx:
            v1, v2 = v2, v1
        return self.edges[(v1.idx, v2.idx)]


class SinkGraph(Graph):
    def __init__(self, width, height, n, k, random_seed=RANDOM_SEED):
        super().__init__(width, height, n, random_seed)
        self.sinks = random.sample(self.vertices, k)


class Cover:
    def __init__(self, graph):
        self.graph = graph
        self.root = {v: v for v in graph.vertices}
        self.neighbors = {v: {v: []} for v in graph.vertices}
        self.cost = {v: 0 for v in graph.vertices}

    def __len__(self):
        return len(self.neighbors)

    def add(self, edge):
        r1, r2 = self.root[edge.v1], self.root[edge.v2]
        if r1 != r2:
            if len(self.neighbors[r1]) < len(self.neighbors[r2]):
                r1, r2 = r2, r1
            self.root.update({v: r1 for v in self.neighbors[r2]})
            self.neighbors[r1].update(self.neighbors[r2])
            self.cost[r1] += self.cost[r2]
            self.neighbors.pop(r2)
            self.cost.pop(r2)

        if edge.v2 not in self.neighbors[r1][edge.v1]:
            self.neighbors[r1][edge.v1].append(edge.v2)
            self.neighbors[r1][edge.v2].append(edge.v1)
            self.cost[r1] += edge.length

    def connected(self, v1, v2):
        return self.root[v1] == self.root[v2]

    def get_non_trivial_cover(self, cover):
        if not hasattr(self.graph, 'sinks'):
            print('self returned')
            return self
        cover.neighbors = {
            r: {v: copy.copy(self.neighbors[r][v]) for v in self.neighbors[r]}
            for r in self.neighbors
            if False in {v in self.graph.sinks for v in self.neighbors[r]}
        }
        cover.root = {v: r for v, r in self.root.items() if r in cover.neighbors}
        cover.cost = {r: c for r, c in self.cost.items() if r in cover.neighbors}
        return cover


class TreeCover(Cover):
    def __init__(self, graph):
        super().__init__(graph)

    def add(self, edge):
        if not self.connected(edge.v1, edge.v2):
            super().add(edge)

    def construct_cycle_cover(self, random_seed=RANDOM_SEED):
        random.seed(random_seed)
        cycle_cover = CycleCover(self.graph)
        for r in self.neighbors:
            neighbors = {v: copy.copy(self.neighbors[r][v]) for v in self.neighbors[r]}
            prev = {v: None for v in self.neighbors[r]}
            p = random.choice(list(self.neighbors[r].keys()))
            vertices = [p]
            prev[p] = p
            while True:
                if len(neighbors[p]) > 0:
                    q = random.choice(neighbors[p])
                    neighbors[p].remove(q)
                    neighbors[q].remove(p)
                    vertices.append(q)
                    prev[q] = p
                    p = q
                elif prev[p] != p:
                    p = prev[p]
                    vertices.append(p)
                else:
                    break

            visited = {v: False for v in self.neighbors[r]}
            visited[vertices[0]], last_vertex = True, vertices[0]
            for v in vertices:
                if not visited[v]:
                    cycle_cover.add(self.graph.edge(last_vertex, v))
                    visited[v], last_vertex = True, v
            if last_vertex != vertices[0]:
                cycle_cover.add(self.graph.edge(last_vertex, vertices[0]))

        return cycle_cover

    def get_non_trivial_cover(self, cover=None):
        return super().get_non_trivial_cover(TreeCover(self.graph))


class CycleCover(Cover):
    def __init__(self, graph):
        super().__init__(graph)


def dist(v1, v2):
    return math.sqrt((v1.x - v2.x) ** 2 + (v1.y - v2.y) ** 2)


def avg(l):
    return (sum(l) / len(l)) if len(l) > 0 else 0


def get_max_period(cover, allocation):
    period_max, r_max = None, None
    for r, c in cover.cost.items():
        period = c / allocation[r]
        if period_max is None or period_max < period:
            period_max, r_max = period, r
    return period_max, r_max


def get_optimal_allocation(cover, m):
    allocation = {r: 1 for r in cover.neighbors}
    for i in range(m - len(allocation)):
        period_max, r_max = get_max_period(cover, allocation)
        allocation[r_max] += 1
    return allocation, get_max_period(cover, allocation)[0]


####################################
#                                  #
# Cooperative Sweep Coverage (CSC) #
#                                  #
####################################

def CoCycle(graph, m):
    tree_cover = TreeCover(graph)
    cycle_cover_best, allocation_best, period_min = None, None, None
    if len(tree_cover) <= m:
        cycle_cover_best = tree_cover.construct_cycle_cover()
        allocation_best, period_min = get_optimal_allocation(cycle_cover_best, m)
    for edge in graph.sorted_edges:
        if not tree_cover.connected(edge.v1, edge.v2):
            tree_cover.add(edge)
            if len(tree_cover) <= m:
                cycle_cover = tree_cover.construct_cycle_cover()
                allocation, period = get_optimal_allocation(cycle_cover, m)
                if period_min is None or period_min > period:
                    cycle_cover_best, allocation_best, period_min = cycle_cover, allocation, period
                print('CoCycle\t{}\t{}'.format(len(cycle_cover), period))
    return cycle_cover_best, allocation_best, period_min


def prim(graph, random_seed=RANDOM_SEED):
    random.seed(random_seed)
    tree_cover = TreeCover(graph)
    r = random.choice(graph.vertices)
    link_edge = {
        v: graph.edge(v, r)
        for v in graph.vertices if v != r
    }
    while len(link_edge) > 0:
        u_min, edge_min = None, None
        for u, edge in link_edge.items():
            if edge_min is None or edge_min.length > edge.length:
                u_min, edge_min = u, edge
        link_edge.pop(u_min)
        tree_cover.add(edge_min)
        for v, edge in link_edge.items():
            tmp_edge = graph.edge(v, u_min)
            if tmp_edge.length < edge.length:
                link_edge[v] = tmp_edge
    return tree_cover


def OSweep(graph, m):
    cycle_cover = prim(graph).construct_cycle_cover()
    allocation, period = get_optimal_allocation(cycle_cover, m)
    print('OSweep\t{}\t{}'.format(len(cycle_cover), period))
    return cycle_cover, allocation, period


def MinExpand(graph, m, random_seed=RANDOM_SEED):
    c_low, c_high = 0, sum([edge.length for edge in graph.edges.values()]) // m
    while math.fabs(c_high - c_low) > ERROR:
        c_mid = c_low + (c_high - c_low) / 2
        cycle_cover_mid, m_mid = MinExpand_sub(graph, c_mid, random_seed)
        if m_mid > m:
            c_low = c_mid
        else:
            c_high = c_mid
        print('MinExpand', c_low, c_high)
    cycle_cover_high, m_high = MinExpand_sub(graph, c_high, random_seed)
    allocation, period = get_optimal_allocation(cycle_cover_high, m_high)
    return cycle_cover_high, allocation, period


def MinExpand_sub(graph, c, random_seed=RANDOM_SEED):
    random.seed(random_seed)
    cycle_cover = CycleCover(graph)
    m = 1
    candidates = copy.copy(graph.vertices)
    sorted_edges = copy.copy(graph.sorted_edges)

    def delete_candidate(v):
        candidates.remove(v)
        for i in range(len(sorted_edges) - 1, -1, -1):
            if sorted_edges[i].v1 == v or sorted_edges[i].v2 == v:
                sorted_edges.pop(i)

    def get_sorted_candidate_entries(x, y):
        return sorted([
            (v, graph.edge(x, v).length + graph.edge(v, y).length - graph.edge(x, y).length)
            for v in candidates
        ], key=lambda item: item[-1])

    while len(candidates) > 1:
        cost = 0
        memo = dict()
        edge_max = sorted_edges[-1]
        seq = [random.choice([edge_max.v1, edge_max.v2])]
        candidate = None
        for edge in sorted_edges:
            if edge.v1 == seq[0]:
                candidate = edge.v2
                break
            elif edge.v2 == seq[0]:
                candidate = edge.v1
                break
        delete_candidate(seq[0])
        delta = 2 * graph.edge(seq[0], candidate).length
        if cost + delta <= c:
            cost += delta
            seq.append(candidate)
            delete_candidate(candidate)
            memo[(seq[0], candidate)] = get_sorted_candidate_entries(seq[0], candidate)
        else:
            m += 1
            continue

        while len(candidates) > 0:
            candidate_min, delta_min, idx_min = None, None, None
            for idx in range(len(seq)):
                v1, v2 = seq[idx], seq[(idx + 1) % len(seq)]
                for candidate_entry in (memo[(v1, v2)] if (v1, v2) in memo else memo[(v2, v1)]):
                    if candidate_entry[0] not in seq:
                        if delta_min is None or delta_min > candidate_entry[1]:
                            candidate_min, delta_min, idx_min = candidate_entry[0], candidate_entry[1], idx
                        break
            if cost + delta_min <= c:
                v1, v2 = seq[idx_min], seq[(idx_min + 1) % len(seq)]
                cost += delta_min
                seq.insert((idx_min + 1) % len(seq), candidate_min)
                delete_candidate(candidate_min)
                memo[(v1, candidate_min)] = get_sorted_candidate_entries(v1, candidate_min)
                memo[(candidate_min, v2)] = get_sorted_candidate_entries(candidate_min, v2)
            else:
                m += 1
                break

        for idx in range(len(seq)):
            cycle_cover.add(graph.edge(seq[idx], seq[(idx + 1) % len(seq)]))

    return cycle_cover, m


def MinExpand_func(w, h, n, m, random_seed=RANDOM_SEED):
    graph = Graph(width=w, height=h, n=n, random_seed=random_seed)
    _, _, period = MinExpand(graph, m, random_seed)
    return period


def PDBA(graph, m, random_seed=RANDOM_SEED):
    c_low, c_high = 0, sum([edge.length for edge in graph.edges.values()]) // m
    while math.fabs(c_high - c_low) > ERROR:
        c_mid = c_low + (c_high - c_low) / 2
        cycle_cover_mid, m_mid = PDBA_sub(graph, c_mid, random_seed)
        if m_mid > m:
            c_low = c_mid
        else:
            c_high = c_mid
        print('PDBA', c_low, c_high)
    cycle_cover_high, m_high = PDBA_sub(graph, c_high, random_seed)
    allocation, period = get_optimal_allocation(cycle_cover_high, m_high)
    return cycle_cover_high, allocation, period


def PDBA_sub(graph, c, random_seed=RANDOM_SEED):
    random.seed(random_seed)
    cycle_cover = CycleCover(graph)
    m = 1
    candidates = copy.copy(graph.vertices)
    sorted_edges = copy.copy(graph.sorted_edges)

    def delete_candidate(v):
        candidates.remove(v)
        for i in range(len(sorted_edges) - 1, -1, -1):
            if sorted_edges[i].v1 == v or sorted_edges[i].v2 == v:
                sorted_edges.pop(i)

    def dist_p(v, p1, p2):
        return math.fabs((p1.y - p2.y) * v.x + (p2.x - p1.x) * v.y + p2.y * p1.x - p1.y * p2.x)

    def get_min_candidate(p1, p2, cost_current):
        candidate_min, delta_min, dist_p_min = None, None, None
        for v in candidates:
            delta_v = graph.edge(p1, v).length + graph.edge(v, p2).length - graph.edge(p1, p2).length
            if cost_current + delta_v <= c:
                dist_p_v = dist_p(v, p1, p2)
                if dist_p_min is None or dist_p_min > dist_p_v:
                    candidate_min, delta_min, dist_p_min = v, delta_v, dist_p_v
        return candidate_min, delta_min

    while len(candidates) > 1:
        cost = 0
        seq = [random.choice(candidates)]
        candidate = None
        for edge in sorted_edges:
            if edge.v1 == seq[0]:
                candidate = edge.v2
                break
            elif edge.v2 == seq[0]:
                candidate = edge.v1
                break
        delete_candidate(seq[0])
        delta = 2 * graph.edge(seq[0], candidate).length
        if cost + delta <= c:
            cost += delta
            seq.append(candidate)
            delete_candidate(candidate)
            candidate, delta = get_min_candidate(seq[0], seq[-1], cost)
        else:
            m += 1
            continue

        while len(candidates) > 0:
            if candidate is not None:
                cost += delta
                seq.append(candidate)
                delete_candidate(candidate)
                candidate, delta = get_min_candidate(seq[0], seq[-1], cost)
            else:
                m += 1
                break

        for idx in range(len(seq)):
            cycle_cover.add(graph.edge(seq[idx], seq[(idx + 1) % len(seq)]))

    return cycle_cover, m


def PDBA_func(w, h, n, m, random_seed=RANDOM_SEED):
    graph = Graph(width=w, height=h, n=n, random_seed=random_seed)
    _, _, period = PDBA(graph, m, random_seed)
    return period


####################################
#                                  #
# Multi-Sink Sweep Coverage (MSSC) #
#                                  #
####################################

def modified_prim(sink_graph):
    tree_cover = TreeCover(sink_graph)
    link_edge = {
        v: sink_graph.edge(v, sink_graph.sinks[0])
        for v in sink_graph.vertices if v not in sink_graph.sinks
    }
    for s in sink_graph.sinks:
        for v in link_edge:
            edge = sink_graph.edge(v, s)
            if link_edge[v].length > edge.length:
                link_edge[v] = edge
    while len(link_edge) > 0:
        u_min, edge_min = None, None
        for u, edge in link_edge.items():
            if edge_min is None or edge_min.length > edge.length:
                u_min, edge_min = u, edge
        link_edge.pop(u_min)
        tree_cover.add(edge_min)
        for v, edge in link_edge.items():
            tmp_edge = sink_graph.edge(v, u_min)
            if tmp_edge.length < edge.length:
                link_edge[v] = tmp_edge
    return tree_cover


def SinkCycle(sink_graph, m):
    tree_cover = modified_prim(sink_graph)
    cycle_cover_best, allocation_best, period_min = None, None, None
    non_trivial_tree_cover = tree_cover.get_non_trivial_cover()
    if len(non_trivial_tree_cover) <= m:
        cycle_cover_best = non_trivial_tree_cover.construct_cycle_cover()
        allocation_best, period_min = get_optimal_allocation(cycle_cover_best, m)
    for edge in sink_graph.sorted_edges:
        if not tree_cover.connected(edge.v1, edge.v2):
            tree_cover.add(edge)
            non_trivial_tree_cover = tree_cover.get_non_trivial_cover()
            if len(non_trivial_tree_cover) <= m:
                cycle_cover = non_trivial_tree_cover.construct_cycle_cover()
                allocation, period = get_optimal_allocation(cycle_cover, m)
                if period_min is None or period_min > period:
                    cycle_cover_best, allocation_best, period_min = cycle_cover, allocation, period
                print('SinkCycle\t{}\t{}'.format(len(cycle_cover), period))
    return cycle_cover_best, allocation_best, period_min


def SCOPe_M_Solver(sink_graph, m, random_seed=RANDOM_SEED):
    random.seed(random_seed)
    c_low, c_high = 0, sum([edge.length for edge in sink_graph.edges.values()]) // m
    while math.fabs(c_high - c_low) > ERROR:
        c_mid = c_low + (c_high - c_low) / 2 + random.uniform(-(c_high - c_low) / 4, (c_high - c_low) / 4)
        cycle_cover_mid, m_mid = SCOPe_M_Solver_sub(sink_graph, c_mid)
        if m_mid is None or m_mid > m:
            c_low = c_mid
        else:
            c_high = c_mid
        print('SCOPe_M_Solver', c_low, c_high)
    cycle_cover_high, m_high = SCOPe_M_Solver_sub(sink_graph, c_high)
    if m_high > m:
        return None, None, None, -m_high
    allocation, period = get_optimal_allocation(cycle_cover_high, m_high)
    return cycle_cover_high, allocation, period, m_high


def SCOPe_M_Solver_sub(sink_graph, c):
    cycle_cover = CycleCover(sink_graph)
    m = 0
    candidates = {s: [] for s in sink_graph.sinks}
    sorted_edges = copy.copy(sink_graph.sorted_edges)

    for p in sink_graph.vertices:
        if p not in sink_graph.sinks:
            s_min = None
            for s in sink_graph.sinks:
                if s_min is None or sink_graph.edge(p, s_min).length > sink_graph.edge(p, s).length:
                    s_min = s
            candidates[s_min].append(p)

    for s, candidates_sub in candidates.items():
        if len(candidates_sub) == 0:
            continue
        m += 1
        sink_edges = [e for e in sorted_edges
                      if ((e.v1 == s and e.v2 in candidates_sub) or (e.v2 == s and e.v1 in candidates_sub))
                      and e.length <= (c / 2)]

        def delete_candidate(v):
            candidates_sub.remove(v)
            for i in range(len(sink_edges) - 1, -1, -1):
                if sink_edges[i].v1 == v or sink_edges[i].v2 == v:
                    sink_edges.pop(i)

        def get_sorted_candidate_entries(x, y):
            return sorted([
                (v, sink_graph.edge(x, v).length + sink_graph.edge(v, y).length - sink_graph.edge(x, y).length)
                for v in candidates_sub
            ], key=lambda item: item[-1])

        while len(candidates_sub) > 0:
            cost = 0
            memo = dict()
            if len(sink_edges) == 0:
                return None, None
            edge_max = sink_edges.pop()
            seq = [s, (edge_max.v2 if s == edge_max.v1 else edge_max.v1)]
            delete_candidate(seq[-1])
            delta = 2 * sink_graph.edge(seq[0], seq[-1]).length
            cost += delta
            memo[(seq[0], seq[-1])] = get_sorted_candidate_entries(seq[0], seq[-1])

            while len(candidates_sub) > 0:
                candidate_min, delta_min, idx_min = None, None, None
                for idx in range(len(seq)):
                    v1, v2 = seq[idx], seq[(idx + 1) % len(seq)]
                    for candidate_entry in (memo[(v1, v2)] if (v1, v2) in memo else memo[(v2, v1)]):
                        if candidate_entry[0] not in seq:
                            if delta_min is None or delta_min > candidate_entry[1]:
                                candidate_min, delta_min, idx_min = candidate_entry[0], candidate_entry[1], idx
                            break
                if cost + delta_min <= c:
                    v1, v2 = seq[idx_min], seq[(idx_min + 1) % len(seq)]
                    cost += delta_min
                    seq.insert((idx_min + 1) % len(seq), candidate_min)
                    delete_candidate(candidate_min)
                    memo[(v1, candidate_min)] = get_sorted_candidate_entries(v1, candidate_min)
                    memo[(candidate_min, v2)] = get_sorted_candidate_entries(candidate_min, v2)
                else:
                    m += 1
                    break

            for idx in range(len(seq)):
                cycle_cover.add(sink_graph.edge(seq[idx], seq[(idx + 1) % len(seq)]))

    return cycle_cover, m


def SCOPe_M_Solver_func(w, h, n, k, m, random_seed=RANDOM_SEED):
    graph = SinkGraph(width=w, height=h, n=n, k=k, random_seed=random_seed)
    cycle_cover, _, period, m_high = SCOPe_M_Solver(graph, m, random_seed)
    return get_optimal_allocation(cycle_cover, m)[1]
