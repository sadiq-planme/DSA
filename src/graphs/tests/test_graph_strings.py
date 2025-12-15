import unittest
import math
import sys
import os

# Add parent directory to path to import graph module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import with hyphenated filename
import importlib.util
spec = importlib.util.spec_from_file_location("graph_strings", os.path.join(os.path.dirname(__file__), "../graph-strings.py"))
graph_strings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_strings)
BaseGraph = graph_strings.BaseGraph
DirectedGraph = graph_strings.DirectedGraph
UndirectedGraph = graph_strings.UndirectedGraph
GraphType = graph_strings.GraphType
DisjointSet = graph_strings.DisjointSet


class TestBaseGraph(unittest.TestCase):
    """Test cases for BaseGraph methods"""

    def test_add_edge_directed(self):
        """Test adding edge to directed graph"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.add_edge("A", "B", 1.0)
        self.assertIn("A", graph._nodes)
        self.assertIn("B", graph._nodes)
        self.assertEqual(len(graph._adjacency_list["A"]), 1)
        self.assertEqual(len(graph._adjacency_list["B"]), 0)

    def test_add_edge_undirected(self):
        """Test adding edge to undirected graph"""
        graph = BaseGraph(GraphType.UNDIRECTED)
        graph.add_edge("A", "B", 1.0)
        self.assertIn("A", graph._nodes)
        self.assertIn("B", graph._nodes)
        self.assertEqual(len(graph._adjacency_list["A"]), 1)
        self.assertEqual(len(graph._adjacency_list["B"]), 1)

    def test_add_edge_default_weight(self):
        """Test adding edge with default weight"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.add_edge("A", "B")
        self.assertEqual(graph._adjacency_list["A"][0][0], 1.0)

    def test_bfs_empty_graph(self):
        """Test BFS on empty graph"""
        graph = BaseGraph(GraphType.DIRECTED)
        parent, components = graph.bfs()
        self.assertEqual(parent, {})
        self.assertEqual(components, [])

    def test_bfs_single_node(self):
        """Test BFS on single node graph"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.add_edge("A", "A", 0.0)  # Self-loop
        parent, components = graph.bfs()
        self.assertEqual(len(components), 1)
        self.assertIn("A", components[0])
        self.assertEqual(parent["A"], None)

    def test_bfs_connected_directed(self):
        """Test BFS on connected directed graph"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)
        parent, components = graph.bfs()
        # In directed graph, components depend on iteration order
        # All nodes should be visited, but may be in different components
        # depending on which node we start from
        total_nodes = sum(len(comp) for comp in components)
        self.assertEqual(total_nodes, 3)
        self.assertGreaterEqual(len(components), 1)

    def test_bfs_disconnected(self):
        """Test BFS on disconnected graph"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("C", "D", 1.0)
        parent, components = graph.bfs()
        # In directed graph, each node may start its own component
        # All nodes should be visited
        total_nodes = sum(len(comp) for comp in components)
        self.assertEqual(total_nodes, 4)
        self.assertGreaterEqual(len(components), 2)

    def test_dfs_empty_graph(self):
        """Test DFS on empty graph"""
        graph = BaseGraph(GraphType.DIRECTED)
        parent, components = graph.dfs()
        self.assertEqual(parent, {})
        self.assertEqual(components, [])

    def test_dfs_single_node(self):
        """Test DFS on single node graph"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.add_edge("A", "A", 0.0)
        parent, components = graph.dfs()
        self.assertEqual(len(components), 1)
        self.assertIn("A", components[0])

    def test_dfs_connected_directed(self):
        """Test DFS on connected directed graph"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)
        parent, components = graph.dfs()
        # In directed graph, components depend on iteration order
        # All nodes should be visited, but may be in different components
        # depending on which node we start from
        total_nodes = sum(len(comp) for comp in components)
        self.assertEqual(total_nodes, 3)
        self.assertGreaterEqual(len(components), 1)

    def test_dfs_iterative_empty_graph(self):
        """Test iterative DFS on empty graph"""
        graph = BaseGraph(GraphType.DIRECTED)
        parent, components = graph.dfs_iterative()
        self.assertEqual(parent, {})
        self.assertEqual(components, [])

    def test_dfs_iterative_single_node(self):
        """Test iterative DFS on single node"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.add_edge("A", "A", 0.0)
        parent, components = graph.dfs_iterative()
        self.assertEqual(len(components), 1)

    def test_sssp_bfs_same_source_dest(self):
        """Test SSSP BFS when source equals destination"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.add_edge("A", "B", 1.0)
        path, dist = graph.sssp_bfs("A", "A")
        self.assertEqual(path, ["A"])
        self.assertEqual(dist, 0.0)

    def test_sssp_bfs_no_path(self):
        """Test SSSP BFS when no path exists"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("C", "D", 1.0)
        path, dist = graph.sssp_bfs("A", "C")
        self.assertEqual(path, [])
        self.assertEqual(dist, math.inf)

    def test_sssp_bfs_simple_path(self):
        """Test SSSP BFS on simple path"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)
        path, dist = graph.sssp_bfs("A", "C")
        self.assertEqual(path, ["A", "B", "C"])
        self.assertEqual(dist, 2.0)

    def test_sssp_dijkstra_same_source_dest(self):
        """Test Dijkstra when source equals destination"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.add_edge("A", "B", 1.0)
        path, dist = graph.sssp_dijkstra("A", "A")
        self.assertEqual(path, ["A"])
        self.assertEqual(dist, 0.0)

    def test_sssp_dijkstra_no_path(self):
        """Test Dijkstra when no path exists"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("C", "D", 1.0)
        path, dist = graph.sssp_dijkstra("A", "C")
        self.assertEqual(path, [])
        self.assertEqual(dist, math.inf)

    def test_sssp_dijkstra_simple_path(self):
        """Test Dijkstra on simple weighted path"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.add_edge("A", "B", 5.0)
        graph.add_edge("B", "C", 3.0)
        path, dist = graph.sssp_dijkstra("A", "C")
        self.assertEqual(path, ["A", "B", "C"])
        self.assertEqual(dist, 8.0)

    def test_sssp_dijkstra_multiple_paths(self):
        """Test Dijkstra with multiple paths, should choose shortest"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.add_edge("A", "B", 10.0)
        graph.add_edge("A", "C", 3.0)
        graph.add_edge("B", "D", 2.0)
        graph.add_edge("C", "B", 1.0)
        graph.add_edge("C", "D", 8.0)
        path, dist = graph.sssp_dijkstra("A", "D")
        # Shortest path: A->C->B->D = 3+1+2 = 6
        self.assertEqual(dist, 6.0)

    def test_sssp_dijkstra_visited_node_optimization(self):
        """Test that Dijkstra correctly handles already visited nodes"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("A", "C", 5.0)
        graph.add_edge("B", "C", 2.0)
        path, dist = graph.sssp_dijkstra("A", "C")
        # Should take A->B->C = 1+2 = 3, not A->C = 5
        self.assertEqual(dist, 3.0)
        self.assertEqual(path, ["A", "B", "C"])

    def test_floyd_warshall_empty(self):
        """Test Floyd-Warshall on empty graph"""
        graph = BaseGraph(GraphType.DIRECTED)
        dist, parent = graph.floyd_warshall()
        self.assertEqual(dist, [])
        self.assertEqual(parent, [])

    def test_floyd_warshall_single_node(self):
        """Test Floyd-Warshall on single node"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.add_edge("A", "A", 0.0)
        dist, parent = graph.floyd_warshall()
        self.assertEqual(len(dist), 1)
        self.assertEqual(dist[0][0], 0.0)

    def test_floyd_warshall_simple_directed(self):
        """Test Floyd-Warshall on simple directed graph"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.add_edge("A", "B", 5.0)
        graph.add_edge("B", "C", 3.0)
        dist, parent = graph.floyd_warshall()
        # Find indices
        nodes = sorted(graph._nodes)
        a_idx = nodes.index("A")
        b_idx = nodes.index("B")
        c_idx = nodes.index("C")
        self.assertEqual(dist[a_idx][a_idx], 0.0)
        self.assertEqual(dist[a_idx][b_idx], 5.0)
        self.assertEqual(dist[a_idx][c_idx], 8.0)
        self.assertEqual(dist[b_idx][c_idx], 3.0)
        self.assertEqual(dist[c_idx][a_idx], math.inf)

    def test_floyd_warshall_undirected(self):
        """Test Floyd-Warshall on undirected graph"""
        graph = BaseGraph(GraphType.UNDIRECTED)
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 2.0)
        graph.add_edge("A", "C", 4.0)
        dist, parent = graph.floyd_warshall()
        nodes = sorted(graph._nodes)
        a_idx = nodes.index("A")
        b_idx = nodes.index("B")
        c_idx = nodes.index("C")
        # All pairs should be symmetric
        self.assertEqual(dist[a_idx][b_idx], dist[b_idx][a_idx])
        self.assertEqual(dist[a_idx][c_idx], dist[c_idx][a_idx])
        # Shortest path A->C should be A->B->C = 3
        self.assertEqual(dist[a_idx][c_idx], 3.0)


class TestDirectedGraph(unittest.TestCase):
    """Test cases for DirectedGraph methods"""

    def test_get_out_degree(self):
        """Test getting out-degree of a node"""
        graph = DirectedGraph()
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("A", "C", 1.0)
        self.assertEqual(graph.get_out_degree("A"), 2)
        self.assertEqual(graph.get_out_degree("B"), 0)
        self.assertEqual(graph.get_out_degree("C"), 0)

    def test_get_out_degree_nonexistent(self):
        """Test getting out-degree of nonexistent node"""
        graph = DirectedGraph()
        self.assertEqual(graph.get_out_degree("X"), 0)

    def test_get_in_degree(self):
        """Test getting in-degree of a node"""
        graph = DirectedGraph()
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("A", "C", 1.0)
        self.assertEqual(graph.get_in_degree("A"), 0)
        self.assertEqual(graph.get_in_degree("B"), 1)
        self.assertEqual(graph.get_in_degree("C"), 1)

    def test_get_in_degree_nonexistent(self):
        """Test getting in-degree of nonexistent node"""
        graph = DirectedGraph()
        self.assertEqual(graph.get_in_degree("X"), 0)

    def test_is_cyclic_dfs_no_cycle(self):
        """Test cycle detection on DAG"""
        graph = DirectedGraph()
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)
        self.assertFalse(graph.is_cyclic_dfs())

    def test_is_cyclic_dfs_with_cycle(self):
        """Test cycle detection on cyclic graph"""
        graph = DirectedGraph()
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)
        graph.add_edge("C", "A", 1.0)
        self.assertTrue(graph.is_cyclic_dfs())

    def test_is_cyclic_dfs_self_loop(self):
        """Test cycle detection with self-loop"""
        graph = DirectedGraph()
        graph.add_edge("A", "A", 1.0)
        self.assertTrue(graph.is_cyclic_dfs())

    def test_is_cyclic_dfs_empty(self):
        """Test cycle detection on empty graph"""
        graph = DirectedGraph()
        self.assertFalse(graph.is_cyclic_dfs())

    def test_is_cyclic_dfs_disconnected_cycle(self):
        """Test cycle detection on disconnected graph with cycle"""
        graph = DirectedGraph()
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("C", "D", 1.0)
        graph.add_edge("D", "C", 1.0)  # Cycle in second component
        self.assertTrue(graph.is_cyclic_dfs())

    def test_topological_sort_kahn_s_bfs_dag(self):
        """Test topological sort on DAG"""
        graph = DirectedGraph()
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("A", "C", 1.0)
        graph.add_edge("B", "D", 1.0)
        graph.add_edge("C", "D", 1.0)
        result = graph.topological_sort_kahn_s_bfs()
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 4)
        # A should come before B, C, D
        self.assertEqual(result[0], "A")

    def test_topological_sort_kahn_s_bfs_cyclic(self):
        """Test topological sort on cyclic graph"""
        graph = DirectedGraph()
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "A", 1.0)
        result = graph.topological_sort_kahn_s_bfs()
        self.assertIsNone(result)

    def test_topological_sort_kahn_s_bfs_empty(self):
        """Test topological sort on empty graph"""
        graph = DirectedGraph()
        result = graph.topological_sort_kahn_s_bfs()
        self.assertEqual(result, [])

    def test_topological_sort_kahn_s_bfs_single_node(self):
        """Test topological sort on single node"""
        graph = DirectedGraph()
        graph.add_edge("A", "A", 0.0)
        result = graph.topological_sort_kahn_s_bfs()
        # Self-loop creates cycle
        self.assertIsNone(result)

    def test_kosaraju_empty(self):
        """Test Kosaraju on empty graph"""
        graph = DirectedGraph()
        sccs = graph.kosaraju()
        self.assertEqual(sccs, [])

    def test_kosaraju_single_node(self):
        """Test Kosaraju on single node"""
        graph = DirectedGraph()
        graph.add_edge("A", "A", 0.0)
        sccs = graph.kosaraju()
        self.assertEqual(len(sccs), 1)
        self.assertIn("A", sccs[0])

    def test_kosaraju_no_cycles(self):
        """Test Kosaraju on DAG (each node is its own SCC)"""
        graph = DirectedGraph()
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)
        sccs = graph.kosaraju()
        # In a DAG, each node is its own SCC
        self.assertEqual(len(sccs), 3)

    def test_kosaraju_strongly_connected(self):
        """Test Kosaraju on strongly connected component"""
        graph = DirectedGraph()
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)
        graph.add_edge("C", "A", 1.0)
        sccs = graph.kosaraju()
        # All nodes should be in one SCC
        self.assertEqual(len(sccs), 1)
        self.assertEqual(set(sccs[0]), {"A", "B", "C"})

    def test_kosaraju_multiple_sccs(self):
        """Test Kosaraju with multiple SCCs"""
        graph = DirectedGraph()
        # Component 1: A->B->A (cycle)
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "A", 1.0)
        # Component 2: C->D->C (cycle)
        graph.add_edge("C", "D", 1.0)
        graph.add_edge("D", "C", 1.0)
        # Component 3: E (isolated)
        graph.add_edge("E", "E", 0.0)
        sccs = graph.kosaraju()
        # Should have 3 SCCs
        self.assertEqual(len(sccs), 3)

    def test_bellman_ford_same_source_dest(self):
        """Test Bellman-Ford when source equals destination"""
        graph = DirectedGraph()
        graph.add_edge("A", "B", 1.0)
        result = graph.bellman_ford("A", "A")
        self.assertEqual(result, (["A"], 0.0))

    def test_bellman_ford_no_path(self):
        """Test Bellman-Ford when no path exists"""
        graph = DirectedGraph()
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("C", "D", 1.0)
        result = graph.bellman_ford("A", "C")
        self.assertEqual(result, ([], math.inf))

    def test_bellman_ford_simple_path(self):
        """Test Bellman-Ford on simple path"""
        graph = DirectedGraph()
        graph.add_edge("A", "B", 5.0)
        graph.add_edge("B", "C", 3.0)
        path, dist = graph.bellman_ford("A", "C")
        self.assertEqual(path, ["A", "B", "C"])
        self.assertEqual(dist, 8.0)

    def test_bellman_ford_negative_weights(self):
        """Test Bellman-Ford with negative weights (no cycle)"""
        graph = DirectedGraph()
        graph.add_edge("A", "B", 5.0)
        graph.add_edge("B", "C", -2.0)
        path, dist = graph.bellman_ford("A", "C")
        self.assertEqual(path, ["A", "B", "C"])
        self.assertEqual(dist, 3.0)

    def test_bellman_ford_negative_cycle(self):
        """Test Bellman-Ford detects negative cycle"""
        graph = DirectedGraph()
        # Create a graph with negative cycle: A->B->C->A with weights 1, 1, -3
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)
        graph.add_edge("C", "A", -3.0)
        result = graph.bellman_ford("A", "C")
        # Should detect negative cycle and return None
        self.assertIsNone(result)

    def test_bellman_ford_negative_cycle_unreachable(self):
        """Test Bellman-Ford with negative cycle not reachable from source"""
        graph = DirectedGraph()
        graph.add_edge("A", "B", 1.0)
        # Negative cycle: C->D->C
        graph.add_edge("C", "D", 1.0)
        graph.add_edge("D", "C", -3.0)
        # Should still find path from A to B
        path, dist = graph.bellman_ford("A", "B")
        self.assertEqual(path, ["A", "B"])
        self.assertEqual(dist, 1.0)


class TestUndirectedGraph(unittest.TestCase):
    """Test cases for UndirectedGraph methods"""

    def test_get_degree(self):
        """Test getting degree of a node"""
        graph = UndirectedGraph()
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("A", "C", 1.0)
        self.assertEqual(graph.get_degree("A"), 2)
        self.assertEqual(graph.get_degree("B"), 1)
        self.assertEqual(graph.get_degree("C"), 1)

    def test_get_degree_nonexistent(self):
        """Test getting degree of nonexistent node"""
        graph = UndirectedGraph()
        self.assertEqual(graph.get_degree("X"), 0)

    def test_is_cyclic_no_cycle(self):
        """Test cycle detection on tree (no cycle)"""
        graph = UndirectedGraph()
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)
        self.assertFalse(graph.is_cyclic())

    def test_is_cyclic_with_cycle(self):
        """Test cycle detection on graph with cycle"""
        graph = UndirectedGraph()
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)
        graph.add_edge("C", "A", 1.0)
        self.assertTrue(graph.is_cyclic())

    def test_is_cyclic_empty(self):
        """Test cycle detection on empty graph"""
        graph = UndirectedGraph()
        self.assertFalse(graph.is_cyclic())

    def test_is_cyclic_single_node(self):
        """Test cycle detection on single node"""
        graph = UndirectedGraph()
        graph.add_edge("A", "A", 0.0)
        # Self-loop in undirected graph creates cycle
        self.assertTrue(graph.is_cyclic())

    def test_is_bipartite_empty(self):
        """Test bipartite check on empty graph"""
        graph = UndirectedGraph()
        self.assertTrue(graph.is_bipartite())

    def test_is_bipartite_single_node(self):
        """Test bipartite check on single node"""
        graph = UndirectedGraph()
        graph.add_edge("A", "A", 0.0)
        # Self-loop makes it not bipartite
        self.assertFalse(graph.is_bipartite())

    def test_is_bipartite_tree(self):
        """Test bipartite check on tree (should be bipartite)"""
        graph = UndirectedGraph()
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)
        self.assertTrue(graph.is_bipartite())

    def test_is_bipartite_even_cycle(self):
        """Test bipartite check on even cycle (should be bipartite)"""
        graph = UndirectedGraph()
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)
        graph.add_edge("C", "D", 1.0)
        graph.add_edge("D", "A", 1.0)
        self.assertTrue(graph.is_bipartite())

    def test_is_bipartite_odd_cycle(self):
        """Test bipartite check on odd cycle (should not be bipartite)"""
        graph = UndirectedGraph()
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)
        graph.add_edge("C", "A", 1.0)
        self.assertFalse(graph.is_bipartite())

    def test_is_bipartite_disconnected(self):
        """Test bipartite check on disconnected graph"""
        graph = UndirectedGraph()
        # Two components: one bipartite (tree), one not (triangle)
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("C", "D", 1.0)
        graph.add_edge("D", "E", 1.0)
        graph.add_edge("E", "C", 1.0)  # Triangle (odd cycle)
        # Should return False because one component is not bipartite
        self.assertFalse(graph.is_bipartite())

    def test_mst_prim_empty(self):
        """Test Prim's MST on empty graph"""
        graph = UndirectedGraph()
        edges, weight = graph.mst_prim_s()
        self.assertEqual(edges, [])
        self.assertEqual(weight, 0.0)

    def test_mst_prim_single_node(self):
        """Test Prim's MST on single node"""
        graph = UndirectedGraph()
        graph.add_edge("A", "A", 0.0)
        edges, weight = graph.mst_prim_s()
        self.assertEqual(weight, 0.0)

    def test_mst_prim_simple(self):
        """Test Prim's MST on simple graph"""
        graph = UndirectedGraph()
        graph.add_edge("A", "B", 5.0)
        graph.add_edge("A", "C", 3.0)
        graph.add_edge("B", "C", 1.0)
        edges, weight = graph.mst_prim_s()
        # MST should be: A-C (3), C-B (1) = total 4
        self.assertEqual(weight, 4.0)
        self.assertEqual(len(edges), 2)

    def test_mst_prim_disconnected(self):
        """Test Prim's MST on disconnected graph (returns MST of first component)"""
        graph = UndirectedGraph()
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("C", "D", 2.0)
        edges, weight = graph.mst_prim_s()
        # Should return MST of first component only
        self.assertEqual(weight, 1.0)
        self.assertEqual(len(edges), 1)

    def test_mst_kruskal_empty(self):
        """Test Kruskal's MST on empty graph"""
        graph = UndirectedGraph()
        edges, weight = graph.mst_kruskal_s()
        self.assertEqual(edges, [])
        self.assertEqual(weight, 0.0)

    def test_mst_kruskal_single_node(self):
        """Test Kruskal's MST on single node"""
        graph = UndirectedGraph()
        graph.add_edge("A", "A", 0.0)
        edges, weight = graph.mst_kruskal_s()
        self.assertEqual(weight, 0.0)

    def test_mst_kruskal_simple(self):
        """Test Kruskal's MST on simple graph"""
        graph = UndirectedGraph()
        graph.add_edge("A", "B", 5.0)
        graph.add_edge("A", "C", 3.0)
        graph.add_edge("B", "C", 1.0)
        edges, weight = graph.mst_kruskal_s()
        # MST should be: A-C (3), C-B (1) = total 4
        self.assertEqual(weight, 4.0)
        self.assertEqual(len(edges), 2)

    def test_mst_kruskal_disconnected(self):
        """Test Kruskal's MST on disconnected graph (returns MST of all components)"""
        graph = UndirectedGraph()
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("C", "D", 2.0)
        edges, weight = graph.mst_kruskal_s()
        # Should return MST of all components
        self.assertEqual(weight, 3.0)  # 1 + 2
        self.assertEqual(len(edges), 2)

    def test_get_bridges_empty(self):
        """Test bridge finding on empty graph"""
        graph = UndirectedGraph()
        bridges = graph.get_bridges()
        self.assertEqual(bridges, [])

    def test_get_bridges_single_node(self):
        """Test bridge finding on single node"""
        graph = UndirectedGraph()
        graph.add_edge("A", "A", 0.0)
        bridges = graph.get_bridges()
        # Self-loop is not a bridge
        self.assertEqual(len(bridges), 0)

    def test_get_bridges_tree(self):
        """Test bridge finding on tree (all edges are bridges)"""
        graph = UndirectedGraph()
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)
        bridges = graph.get_bridges()
        # All edges in a tree are bridges
        self.assertEqual(len(bridges), 2)

    def test_get_bridges_no_bridges(self):
        """Test bridge finding on graph with no bridges"""
        graph = UndirectedGraph()
        # Cycle: A-B-C-A (no bridges)
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)
        graph.add_edge("C", "A", 1.0)
        bridges = graph.get_bridges()
        self.assertEqual(len(bridges), 0)

    def test_get_bridges_mixed(self):
        """Test bridge finding on graph with some bridges"""
        graph = UndirectedGraph()
        # Graph: A-B-C-D-A (cycle) with E-C (bridge)
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)
        graph.add_edge("C", "D", 1.0)
        graph.add_edge("D", "A", 1.0)
        graph.add_edge("C", "E", 1.0)
        bridges = graph.get_bridges()
        # Edge C-E should be a bridge
        self.assertGreater(len(bridges), 0)

    def test_get_articulation_points_empty(self):
        """Test articulation point finding on empty graph"""
        graph = UndirectedGraph()
        points = graph.get_articulation_points()
        self.assertEqual(points, [])

    def test_get_articulation_points_single_node(self):
        """Test articulation point finding on single node"""
        graph = UndirectedGraph()
        graph.add_edge("A", "A", 0.0)
        points = graph.get_articulation_points()
        # Single node with self-loop is not an articulation point
        self.assertEqual(len(points), 0)

    def test_get_articulation_points_tree(self):
        """Test articulation point finding on tree"""
        graph = UndirectedGraph()
        # Tree: A-B-C-D (all non-leaf nodes are articulation points)
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)
        graph.add_edge("C", "D", 1.0)
        points = graph.get_articulation_points()
        # Nodes B and C should be articulation points
        self.assertIn("B", points)
        self.assertIn("C", points)

    def test_get_articulation_points_no_points(self):
        """Test articulation point finding on graph with no articulation points"""
        graph = UndirectedGraph()
        # Cycle: A-B-C-A (no articulation points)
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)
        graph.add_edge("C", "A", 1.0)
        points = graph.get_articulation_points()
        self.assertEqual(len(points), 0)

    def test_get_articulation_points_mixed(self):
        """Test articulation point finding on graph with some articulation points"""
        graph = UndirectedGraph()
        # Graph: A-B-C-D-A (cycle) with E-C (bridge)
        # Node C should be an articulation point
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)
        graph.add_edge("C", "D", 1.0)
        graph.add_edge("D", "A", 1.0)
        graph.add_edge("C", "E", 1.0)
        points = graph.get_articulation_points()
        # Node C should be an articulation point
        self.assertIn("C", points)


class TestDisjointSet(unittest.TestCase):
    """Test cases for DisjointSet"""

    def test_find_ultimate_parent_new_node(self):
        """Test finding parent of new node"""
        ds = DisjointSet()
        self.assertEqual(ds.find_ultimate_parent("A"), "A")

    def test_find_ultimate_parent_single_node(self):
        """Test finding parent of single node component"""
        ds = DisjointSet()
        ds.find_ultimate_parent("A")
        self.assertEqual(ds.find_ultimate_parent("A"), "A")

    def test_union_by_size_two_nodes(self):
        """Test union of two nodes"""
        ds = DisjointSet()
        ds.union_by_size("A", "B")
        # Both should have same root
        self.assertEqual(ds.find_ultimate_parent("A"), ds.find_ultimate_parent("B"))

    def test_union_by_size_already_connected(self):
        """Test union of already connected nodes"""
        ds = DisjointSet()
        ds.union_by_size("A", "B")
        ds.union_by_size("A", "B")  # Should be no-op
        self.assertEqual(ds.find_ultimate_parent("A"), ds.find_ultimate_parent("B"))

    def test_union_by_size_path_compression(self):
        """Test that path compression works"""
        ds = DisjointSet()
        # Create a chain: A->B->C
        ds.union_by_size("A", "B")
        ds.union_by_size("B", "C")
        # After path compression, A should point directly to C
        root = ds.find_ultimate_parent("A")
        self.assertEqual(root, ds.find_ultimate_parent("C"))


class TestEdgeCases(unittest.TestCase):
    """Additional edge case tests"""

    def test_sssp_bfs_undirected_graph(self):
        """Test SSSP BFS on undirected graph"""
        graph = BaseGraph(GraphType.UNDIRECTED)
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)
        path, dist = graph.sssp_bfs("A", "C")
        self.assertEqual(path, ["A", "B", "C"])
        self.assertEqual(dist, 2.0)

    def test_sssp_dijkstra_zero_weights(self):
        """Test Dijkstra with zero weight edges"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.add_edge("A", "B", 0.0)
        graph.add_edge("B", "C", 0.0)
        path, dist = graph.sssp_dijkstra("A", "C")
        self.assertEqual(path, ["A", "B", "C"])
        self.assertEqual(dist, 0.0)

    def test_sssp_dijkstra_large_graph(self):
        """Test Dijkstra on larger graph"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.add_edge("A", "B", 4.0)
        graph.add_edge("A", "C", 2.0)
        graph.add_edge("B", "D", 5.0)
        graph.add_edge("C", "B", 1.0)
        graph.add_edge("C", "D", 8.0)
        graph.add_edge("C", "E", 10.0)
        graph.add_edge("D", "E", 2.0)
        path, dist = graph.sssp_dijkstra("A", "E")
        # Shortest: A->C->B->D->E = 2+1+5+2 = 10 (not A->C->E = 2+10 = 12)
        self.assertEqual(dist, 10.0)

    def test_reconstruct_path_edge_case(self):
        """Test path reconstruction with edge cases"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.add_edge("A", "B", 1.0)
        # Test direct path
        path, dist = graph.sssp_bfs("A", "B")
        self.assertEqual(path, ["A", "B"])
        self.assertEqual(dist, 1.0)

    def test_bfs_all_nodes_visited(self):
        """Test that BFS visits all nodes in connected graph"""
        graph = BaseGraph(GraphType.UNDIRECTED)
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)
        graph.add_edge("C", "A", 1.0)
        parent, components = graph.bfs()
        self.assertEqual(len(components), 1)
        self.assertEqual(len(components[0]), 3)
        self.assertEqual(set(components[0]), {"A", "B", "C"})

    def test_dfs_all_nodes_visited(self):
        """Test that DFS visits all nodes in connected graph"""
        graph = BaseGraph(GraphType.UNDIRECTED)
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)
        graph.add_edge("C", "A", 1.0)
        parent, components = graph.dfs()
        self.assertEqual(len(components), 1)
        self.assertEqual(len(components[0]), 3)
        self.assertEqual(set(components[0]), {"A", "B", "C"})

    def test_topological_sort_complex_dag(self):
        """Test topological sort on complex DAG"""
        graph = DirectedGraph()
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("A", "C", 1.0)
        graph.add_edge("B", "D", 1.0)
        graph.add_edge("C", "D", 1.0)
        graph.add_edge("C", "E", 1.0)
        result = graph.topological_sort_kahn_s_bfs()
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 5)
        # A should come first
        self.assertEqual(result[0], "A")
        # E should come after C
        self.assertLess(result.index("C"), result.index("E"))


if __name__ == '__main__':
    unittest.main()
