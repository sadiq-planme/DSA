import unittest
import math
import sys
import os

# Add parent directory to path to import graph module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import with hyphenated filename
import importlib.util
spec = importlib.util.spec_from_file_location("graph_ints", os.path.join(os.path.dirname(__file__), "../graph-ints.py"))
graph_ints = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_ints)
BaseGraph = graph_ints.BaseGraph
DirectedGraph = graph_ints.DirectedGraph
UndirectedGraph = graph_ints.UndirectedGraph
GraphType = graph_ints.GraphType


class TestBaseGraph(unittest.TestCase):
    """Test cases for BaseGraph methods"""

    def test_build_graph_from_matrix_empty(self):
        """Test building graph from empty matrix"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.build_graph_from_matrix([])
        self.assertEqual(graph.V, 0)
        self.assertEqual(len(graph._adjacency_list), 0)

    def test_build_graph_from_matrix_single_node(self):
        """Test building graph with single node"""
        graph = BaseGraph(GraphType.DIRECTED)
        matrix = [[0]]
        graph.build_graph_from_matrix(matrix)
        self.assertEqual(graph.V, 1)
        self.assertEqual(len(graph._adjacency_list[0]), 0)

    def test_build_graph_from_matrix_directed(self):
        """Test building directed graph from matrix"""
        graph = BaseGraph(GraphType.DIRECTED)
        matrix = [
            [0, 1, math.inf],
            [math.inf, 0, 2],
            [math.inf, math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        self.assertEqual(graph.V, 3)
        self.assertEqual(len(graph._adjacency_list[0]), 1)  # 0 -> 1
        self.assertEqual(len(graph._adjacency_list[1]), 1)  # 1 -> 2
        self.assertEqual(len(graph._adjacency_list[2]), 0)

    def test_build_graph_from_matrix_undirected(self):
        """Test building undirected graph from matrix"""
        graph = BaseGraph(GraphType.UNDIRECTED)
        matrix = [
            [0, 1, math.inf],
            [1, 0, 2],
            [math.inf, 2, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        self.assertEqual(graph.V, 3)
        self.assertEqual(len(graph._adjacency_list[0]), 1)  # 0-1
        self.assertEqual(len(graph._adjacency_list[1]), 2)  # 1-0, 1-2
        self.assertEqual(len(graph._adjacency_list[2]), 1)  # 2-1

    def test_build_graph_from_matrix_invalid_square(self):
        """Test building graph with non-square matrix"""
        graph = BaseGraph(GraphType.DIRECTED)
        matrix = [[0, 1], [0]]
        with self.assertRaises(ValueError):
            graph.build_graph_from_matrix(matrix)

    def test_bfs_empty_graph(self):
        """Test BFS on empty graph"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.build_graph_from_matrix([])
        parent, components = graph.bfs()
        self.assertEqual(parent, [])
        self.assertEqual(components, [])

    def test_bfs_single_node(self):
        """Test BFS on single node graph"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.build_graph_from_matrix([[0]])
        parent, components = graph.bfs()
        self.assertEqual(parent, [-1])
        self.assertEqual(components, [[0]])

    def test_bfs_connected_directed(self):
        """Test BFS on connected directed graph"""
        graph = BaseGraph(GraphType.DIRECTED)
        matrix = [
            [0, 1, math.inf],
            [math.inf, 0, 1],
            [math.inf, math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        parent, components = graph.bfs()
        self.assertEqual(len(components), 1)
        self.assertEqual(components[0], [0, 1, 2])

    def test_bfs_disconnected(self):
        """Test BFS on disconnected graph"""
        graph = BaseGraph(GraphType.DIRECTED)
        matrix = [
            [0, 1, math.inf, math.inf],
            [math.inf, 0, math.inf, math.inf],
            [math.inf, math.inf, 0, 1],
            [math.inf, math.inf, math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        parent, components = graph.bfs()
        self.assertEqual(len(components), 2)

    def test_dfs_empty_graph(self):
        """Test DFS on empty graph"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.build_graph_from_matrix([])
        parent, components = graph.dfs()
        self.assertEqual(parent, [])
        self.assertEqual(components, [])

    def test_dfs_single_node(self):
        """Test DFS on single node graph"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.build_graph_from_matrix([[0]])
        parent, components = graph.dfs()
        self.assertEqual(parent, [-1])
        self.assertEqual(components, [[0]])

    def test_dfs_connected_directed(self):
        """Test DFS on connected directed graph"""
        graph = BaseGraph(GraphType.DIRECTED)
        matrix = [
            [0, 1, math.inf],
            [math.inf, 0, 1],
            [math.inf, math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        parent, components = graph.dfs()
        self.assertEqual(len(components), 1)
        self.assertEqual(len(components[0]), 3)

    def test_sssp_bfs_same_source_dest(self):
        """Test SSSP BFS when source equals destination"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.build_graph_from_matrix([[0]])
        path, dist = graph.sssp_bfs(0, 0)
        self.assertEqual(path, [0])
        self.assertEqual(dist, 0.0)

    def test_sssp_bfs_no_path(self):
        """Test SSSP BFS when no path exists"""
        graph = BaseGraph(GraphType.DIRECTED)
        matrix = [
            [0, math.inf],
            [math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        path, dist = graph.sssp_bfs(0, 1)
        self.assertEqual(path, [])
        self.assertEqual(dist, math.inf)

    def test_sssp_bfs_simple_path(self):
        """Test SSSP BFS on simple path"""
        graph = BaseGraph(GraphType.DIRECTED)
        matrix = [
            [0, 1, math.inf],
            [math.inf, 0, 1],
            [math.inf, math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        path, dist = graph.sssp_bfs(0, 2)
        self.assertEqual(path, [0, 1, 2])
        self.assertEqual(dist, 2.0)

    def test_sssp_dijkstra_same_source_dest(self):
        """Test Dijkstra when source equals destination"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.build_graph_from_matrix([[0]])
        path, dist = graph.sssp_dijkstra(0, 0)
        self.assertEqual(path, [0])
        self.assertEqual(dist, 0.0)

    def test_sssp_dijkstra_no_path(self):
        """Test Dijkstra when no path exists"""
        graph = BaseGraph(GraphType.DIRECTED)
        matrix = [
            [0, math.inf],
            [math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        path, dist = graph.sssp_dijkstra(0, 1)
        self.assertEqual(path, [])
        self.assertEqual(dist, math.inf)

    def test_sssp_dijkstra_simple_path(self):
        """Test Dijkstra on simple weighted path"""
        graph = BaseGraph(GraphType.DIRECTED)
        matrix = [
            [0, 5, math.inf],
            [math.inf, 0, 3],
            [math.inf, math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        path, dist = graph.sssp_dijkstra(0, 2)
        self.assertEqual(path, [0, 1, 2])
        self.assertEqual(dist, 8.0)

    def test_sssp_dijkstra_multiple_paths(self):
        """Test Dijkstra with multiple paths, should choose shortest"""
        graph = BaseGraph(GraphType.DIRECTED)
        matrix = [
            [0, 10, 3, math.inf],
            [math.inf, 0, math.inf, 2],
            [math.inf, 1, 0, 8],
            [math.inf, math.inf, math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        path, dist = graph.sssp_dijkstra(0, 3)
        # Shortest path: 0->2->1->3 = 3+1+2 = 6
        self.assertEqual(dist, 6.0)


class TestDirectedGraph(unittest.TestCase):
    """Test cases for DirectedGraph methods"""

    def test_get_out_degree(self):
        """Test getting out-degree of a node"""
        graph = DirectedGraph()
        matrix = [
            [0, 1, 1],
            [math.inf, 0, math.inf],
            [math.inf, math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        self.assertEqual(graph.get_out_degree(0), 2)
        self.assertEqual(graph.get_out_degree(1), 0)
        self.assertEqual(graph.get_out_degree(2), 0)

    def test_get_in_degree(self):
        """Test getting in-degree of a node"""
        graph = DirectedGraph()
        matrix = [
            [0, 1, 1],
            [math.inf, 0, math.inf],
            [math.inf, math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        self.assertEqual(graph.get_in_degree(0), 0)
        self.assertEqual(graph.get_in_degree(1), 1)
        self.assertEqual(graph.get_in_degree(2), 1)

    def test_is_cyclic_dfs_no_cycle(self):
        """Test cycle detection on DAG"""
        graph = DirectedGraph()
        matrix = [
            [0, 1, math.inf],
            [math.inf, 0, 1],
            [math.inf, math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        self.assertFalse(graph.is_cyclic_dfs())

    def test_is_cyclic_dfs_with_cycle(self):
        """Test cycle detection on cyclic graph"""
        graph = DirectedGraph()
        matrix = [
            [0, 1, math.inf],
            [math.inf, 0, 1],
            [1, math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        self.assertTrue(graph.is_cyclic_dfs())

    def test_is_cyclic_dfs_self_loop(self):
        """Test cycle detection with self-loop"""
        graph = DirectedGraph()
        # Note: build_graph_from_matrix skips self-loops, so we need to test differently
        # For now, test without self-loop
        matrix = [
            [0, 1],
            [1, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        # This creates a cycle: 0->1->0
        self.assertTrue(graph.is_cyclic_dfs())

    def test_is_cyclic_dfs_empty(self):
        """Test cycle detection on empty graph"""
        graph = DirectedGraph()
        graph.build_graph_from_matrix([])
        self.assertFalse(graph.is_cyclic_dfs())

    def test_topological_sort_kahn_s_bfs_dag(self):
        """Test topological sort on DAG"""
        graph = DirectedGraph()
        matrix = [
            [0, 1, 1, math.inf],
            [math.inf, 0, math.inf, 1],
            [math.inf, math.inf, 0, 1],
            [math.inf, math.inf, math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        result = graph.topological_sort_kahn_s_bfs()
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 4)
        # 0 should come before 1, 2, 3
        self.assertEqual(result[0], 0)

    def test_topological_sort_kahn_s_bfs_cyclic(self):
        """Test topological sort on cyclic graph"""
        graph = DirectedGraph()
        matrix = [
            [0, 1],
            [1, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        result = graph.topological_sort_kahn_s_bfs()
        self.assertIsNone(result)

    def test_topological_sort_kahn_s_bfs_empty(self):
        """Test topological sort on empty graph"""
        graph = DirectedGraph()
        graph.build_graph_from_matrix([])
        result = graph.topological_sort_kahn_s_bfs()
        self.assertEqual(result, [])


class TestUndirectedGraph(unittest.TestCase):
    """Test cases for UndirectedGraph methods"""

    def test_get_degree(self):
        """Test getting degree of a node"""
        graph = UndirectedGraph()
        matrix = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        self.assertEqual(graph.get_degree(0), 2)
        self.assertEqual(graph.get_degree(1), 2)
        self.assertEqual(graph.get_degree(2), 2)

    def test_is_cyclic_dfs_no_cycle(self):
        """Test cycle detection on tree (no cycle)"""
        graph = UndirectedGraph()
        matrix = [
            [0, 1, math.inf],
            [1, 0, 1],
            [math.inf, 1, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        self.assertFalse(graph.is_cyclic_dfs())

    def test_is_cyclic_dfs_with_cycle(self):
        """Test cycle detection on graph with cycle"""
        graph = UndirectedGraph()
        matrix = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        self.assertTrue(graph.is_cyclic_dfs())

    def test_is_cyclic_dfs_empty(self):
        """Test cycle detection on empty graph"""
        graph = UndirectedGraph()
        graph.build_graph_from_matrix([])
        self.assertFalse(graph.is_cyclic_dfs())

    def test_is_cyclic_dfs_single_node(self):
        """Test cycle detection on single node"""
        graph = UndirectedGraph()
        graph.build_graph_from_matrix([[0]])
        self.assertFalse(graph.is_cyclic_dfs())


class TestEdgeCases(unittest.TestCase):
    """Additional edge case tests"""

    def test_sssp_bfs_undirected_graph(self):
        """Test SSSP BFS on undirected graph"""
        graph = BaseGraph(GraphType.UNDIRECTED)
        matrix = [
            [0, 1, math.inf],
            [1, 0, 1],
            [math.inf, 1, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        path, dist = graph.sssp_bfs(0, 2)
        self.assertEqual(path, [0, 1, 2])
        self.assertEqual(dist, 2.0)

    def test_sssp_dijkstra_zero_weights(self):
        """Test Dijkstra with zero weight edges"""
        graph = BaseGraph(GraphType.DIRECTED)
        matrix = [
            [0, 0, math.inf],
            [math.inf, 0, 0],
            [math.inf, math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        path, dist = graph.sssp_dijkstra(0, 2)
        self.assertEqual(path, [0, 1, 2])
        self.assertEqual(dist, 0.0)

    def test_sssp_dijkstra_large_graph(self):
        """Test Dijkstra on larger graph"""
        graph = BaseGraph(GraphType.DIRECTED)
        # Create a 5-node graph
        matrix = [
            [0, 4, 2, math.inf, math.inf],
            [math.inf, 0, 1, 5, math.inf],
            [math.inf, math.inf, 0, 8, 10],
            [math.inf, math.inf, math.inf, 0, 2],
            [math.inf, math.inf, math.inf, math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        path, dist = graph.sssp_dijkstra(0, 4)
        # Shortest: 0->1->3->4 = 4+5+2 = 11 (not 0->2->4 = 2+10 = 12)
        self.assertEqual(dist, 11.0)
        self.assertEqual(path, [0, 1, 3, 4])

    def test_sssp_dijkstra_visited_node_optimization(self):
        """Test that Dijkstra correctly handles already visited nodes"""
        graph = BaseGraph(GraphType.DIRECTED)
        matrix = [
            [0, 1, 5],
            [math.inf, 0, 2],
            [math.inf, math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        path, dist = graph.sssp_dijkstra(0, 2)
        # Should take 0->1->2 = 1+2 = 3, not 0->2 = 5
        self.assertEqual(dist, 3.0)
        self.assertEqual(path, [0, 1, 2])

    def test_bfs_all_nodes_visited(self):
        """Test that BFS visits all nodes in connected graph"""
        graph = BaseGraph(GraphType.UNDIRECTED)
        matrix = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        parent, components = graph.bfs()
        self.assertEqual(len(components), 1)
        self.assertEqual(len(components[0]), 3)
        self.assertEqual(set(components[0]), {0, 1, 2})

    def test_dfs_all_nodes_visited(self):
        """Test that DFS visits all nodes in connected graph"""
        graph = BaseGraph(GraphType.UNDIRECTED)
        matrix = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        parent, components = graph.dfs()
        self.assertEqual(len(components), 1)
        self.assertEqual(len(components[0]), 3)
        self.assertEqual(set(components[0]), {0, 1, 2})

    def test_topological_sort_complex_dag(self):
        """Test topological sort on complex DAG"""
        graph = DirectedGraph()
        # Create a more complex DAG
        matrix = [
            [0, 1, 1, math.inf, math.inf],
            [math.inf, 0, math.inf, 1, math.inf],
            [math.inf, math.inf, 0, 1, 1],
            [math.inf, math.inf, math.inf, 0, math.inf],
            [math.inf, math.inf, math.inf, math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        result = graph.topological_sort_kahn_s_bfs()
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 5)
        # 0 should come first
        self.assertEqual(result[0], 0)
        # 4 should come after 2
        self.assertLess(result.index(2), result.index(4))

    def test_get_in_degree_no_incoming(self):
        """Test in-degree for node with no incoming edges"""
        graph = DirectedGraph()
        matrix = [
            [0, 1],
            [math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        self.assertEqual(graph.get_in_degree(0), 0)
        self.assertEqual(graph.get_in_degree(1), 1)

    def test_get_out_degree_no_outgoing(self):
        """Test out-degree for node with no outgoing edges"""
        graph = DirectedGraph()
        matrix = [
            [0, 1],
            [math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        self.assertEqual(graph.get_out_degree(0), 1)
        self.assertEqual(graph.get_out_degree(1), 0)

    def test_is_cyclic_dfs_multiple_components(self):
        """Test cycle detection on graph with multiple components"""
        graph = DirectedGraph()
        matrix = [
            [0, 1, math.inf, math.inf],
            [math.inf, 0, math.inf, math.inf],
            [math.inf, math.inf, 0, 1],
            [math.inf, math.inf, 1, 0]  # Fixed: 3->2 creates cycle in component 2-3
        ]
        graph.build_graph_from_matrix(matrix)
        # Component 0-1: no cycle, Component 2-3: has cycle (2->3->2)
        self.assertTrue(graph.is_cyclic_dfs())

    def test_sssp_bfs_back_edge(self):
        """Test SSSP BFS with back edges (should find shortest)"""
        graph = BaseGraph(GraphType.DIRECTED)
        matrix = [
            [0, 1, math.inf],
            [math.inf, 0, 1],
            [1, math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        path, dist = graph.sssp_bfs(0, 2)
        # Should find 0->1->2 (distance 2), not 0->1->2->0->1->2...
        self.assertEqual(dist, 2.0)

    def test_reconstruct_path_edge_case(self):
        """Test path reconstruction with edge cases"""
        graph = BaseGraph(GraphType.DIRECTED)
        matrix = [
            [0, 1],
            [math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        # Test direct path
        path, dist = graph.sssp_bfs(0, 1)
        self.assertEqual(path, [0, 1])
        self.assertEqual(dist, 1.0)

    def test_invalid_node_indices_sssp_bfs(self):
        """Test input validation for invalid node indices in SSSP BFS"""
        graph = BaseGraph(GraphType.DIRECTED)
        matrix = [[0, 1], [math.inf, 0]]
        graph.build_graph_from_matrix(matrix)
        
        with self.assertRaises(IndexError):
            graph.sssp_bfs(-1, 0)
        with self.assertRaises(IndexError):
            graph.sssp_bfs(0, 2)
        with self.assertRaises(IndexError):
            graph.sssp_bfs(2, 0)

    def test_invalid_node_indices_sssp_dijkstra(self):
        """Test input validation for invalid node indices in Dijkstra"""
        graph = BaseGraph(GraphType.DIRECTED)
        matrix = [[0, 1], [math.inf, 0]]
        graph.build_graph_from_matrix(matrix)
        
        with self.assertRaises(IndexError):
            graph.sssp_dijkstra(-1, 0)
        with self.assertRaises(IndexError):
            graph.sssp_dijkstra(0, 2)
        with self.assertRaises(IndexError):
            graph.sssp_dijkstra(2, 0)

    def test_invalid_node_indices_get_out_degree(self):
        """Test input validation for invalid node indices in get_out_degree"""
        graph = DirectedGraph()
        matrix = [[0, 1], [math.inf, 0]]
        graph.build_graph_from_matrix(matrix)
        
        with self.assertRaises(IndexError):
            graph.get_out_degree(-1)
        with self.assertRaises(IndexError):
            graph.get_out_degree(2)

    def test_invalid_node_indices_get_in_degree(self):
        """Test input validation for invalid node indices in get_in_degree"""
        graph = DirectedGraph()
        matrix = [[0, 1], [math.inf, 0]]
        graph.build_graph_from_matrix(matrix)
        
        with self.assertRaises(IndexError):
            graph.get_in_degree(-1)
        with self.assertRaises(IndexError):
            graph.get_in_degree(2)

    def test_invalid_node_indices_get_degree(self):
        """Test input validation for invalid node indices in get_degree"""
        graph = UndirectedGraph()
        matrix = [[0, 1], [1, 0]]
        graph.build_graph_from_matrix(matrix)
        
        with self.assertRaises(IndexError):
            graph.get_degree(-1)
        with self.assertRaises(IndexError):
            graph.get_degree(2)


class TestUntestedMethods(unittest.TestCase):
    """Test cases for methods that haven't been tested yet"""

    def test_dfs_iterative_empty(self):
        """Test iterative DFS on empty graph"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.build_graph_from_matrix([])
        parent, components = graph.dfs_iterative()
        self.assertEqual(parent, [])
        self.assertEqual(components, [])

    def test_dfs_iterative_single_node(self):
        """Test iterative DFS on single node"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.build_graph_from_matrix([[0]])
        parent, components = graph.dfs_iterative()
        self.assertEqual(parent, [-1])
        self.assertEqual(components, [[0]])

    def test_dfs_iterative_connected(self):
        """Test iterative DFS on connected graph"""
        graph = BaseGraph(GraphType.UNDIRECTED)
        matrix = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        parent, components = graph.dfs_iterative()
        self.assertEqual(len(components), 1)
        self.assertEqual(len(components[0]), 3)
        self.assertEqual(set(components[0]), {0, 1, 2})

    def test_dfs_iterative_disconnected(self):
        """Test iterative DFS on disconnected graph"""
        graph = BaseGraph(GraphType.DIRECTED)
        matrix = [
            [0, 1, math.inf, math.inf],
            [math.inf, 0, math.inf, math.inf],
            [math.inf, math.inf, 0, 1],
            [math.inf, math.inf, math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        parent, components = graph.dfs_iterative()
        self.assertEqual(len(components), 2)

    def test_floyd_warshall_empty(self):
        """Test Floyd-Warshall on empty graph"""
        graph = BaseGraph(GraphType.DIRECTED)
        graph.build_graph_from_matrix([])
        dist, parent = graph.floyd_warshall([])
        self.assertEqual(dist, [])
        self.assertEqual(parent, [])

    def test_floyd_warshall_single_node(self):
        """Test Floyd-Warshall on single node"""
        graph = BaseGraph(GraphType.DIRECTED)
        matrix = [[0]]
        graph.build_graph_from_matrix(matrix)
        dist, parent = graph.floyd_warshall(matrix)
        self.assertEqual(dist, [[0.0]])
        self.assertEqual(parent, [[None]])

    def test_floyd_warshall_simple_directed(self):
        """Test Floyd-Warshall on simple directed graph"""
        graph = BaseGraph(GraphType.DIRECTED)
        matrix = [
            [0, 5, math.inf],
            [math.inf, 0, 3],
            [math.inf, math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        dist, parent = graph.floyd_warshall(matrix)
        self.assertEqual(dist[0][0], 0.0)
        self.assertEqual(dist[0][1], 5.0)
        self.assertEqual(dist[0][2], 8.0)
        self.assertEqual(dist[1][2], 3.0)
        self.assertEqual(dist[2][0], math.inf)

    def test_floyd_warshall_undirected(self):
        """Test Floyd-Warshall on undirected graph"""
        graph = BaseGraph(GraphType.UNDIRECTED)
        matrix = [
            [0, 1, 4],
            [1, 0, 2],
            [4, 2, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        dist, parent = graph.floyd_warshall(matrix)
        # All pairs should be symmetric
        self.assertEqual(dist[0][1], dist[1][0])
        self.assertEqual(dist[0][2], dist[2][0])
        self.assertEqual(dist[1][2], dist[2][1])
        # Shortest path 0->2 should be 0->1->2 = 3
        self.assertEqual(dist[0][2], 3.0)

    def test_bellman_ford_same_source_dest(self):
        """Test Bellman-Ford when source equals destination"""
        graph = DirectedGraph()
        graph.build_graph_from_matrix([[0]])
        result = graph.bellman_ford(0, 0)
        self.assertEqual(result, ([0], 0.0))

    def test_bellman_ford_no_path(self):
        """Test Bellman-Ford when no path exists"""
        graph = DirectedGraph()
        matrix = [
            [0, math.inf],
            [math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        result = graph.bellman_ford(0, 1)
        self.assertEqual(result, ([], math.inf))

    def test_bellman_ford_simple_path(self):
        """Test Bellman-Ford on simple path"""
        graph = DirectedGraph()
        matrix = [
            [0, 5, math.inf],
            [math.inf, 0, 3],
            [math.inf, math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        path, dist = graph.bellman_ford(0, 2)
        self.assertEqual(path, [0, 1, 2])
        self.assertEqual(dist, 8.0)

    def test_bellman_ford_negative_weights(self):
        """Test Bellman-Ford with negative weights (no cycle)"""
        graph = DirectedGraph()
        matrix = [
            [0, 5, math.inf],
            [math.inf, 0, -2],
            [math.inf, math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        path, dist = graph.bellman_ford(0, 2)
        self.assertEqual(path, [0, 1, 2])
        self.assertEqual(dist, 3.0)

    def test_bellman_ford_negative_cycle(self):
        """Test Bellman-Ford detects negative cycle"""
        graph = DirectedGraph()
        # Create a graph with negative cycle: 0->1->2->0 with weights 1, 1, -3
        matrix = [
            [0, 1, math.inf],
            [math.inf, 0, 1],
            [-3, math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        result = graph.bellman_ford(0, 2)
        # Should detect negative cycle and return None
        self.assertIsNone(result)

    def test_bellman_ford_invalid_indices(self):
        """Test Bellman-Ford input validation"""
        graph = DirectedGraph()
        matrix = [[0, 1], [math.inf, 0]]
        graph.build_graph_from_matrix(matrix)
        
        with self.assertRaises(IndexError):
            graph.bellman_ford(-1, 0)
        with self.assertRaises(IndexError):
            graph.bellman_ford(0, 2)

    def test_kosaraju_empty(self):
        """Test Kosaraju on empty graph"""
        graph = DirectedGraph()
        graph.build_graph_from_matrix([])
        sccs = graph.kosaraju()
        self.assertEqual(sccs, [])

    def test_kosaraju_single_node(self):
        """Test Kosaraju on single node"""
        graph = DirectedGraph()
        graph.build_graph_from_matrix([[0]])
        sccs = graph.kosaraju()
        self.assertEqual(len(sccs), 1)
        self.assertEqual(sccs[0], [0])

    def test_kosaraju_no_cycles(self):
        """Test Kosaraju on DAG (each node is its own SCC)"""
        graph = DirectedGraph()
        matrix = [
            [0, 1, math.inf],
            [math.inf, 0, 1],
            [math.inf, math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        sccs = graph.kosaraju()
        # In a DAG, each node is its own SCC
        self.assertEqual(len(sccs), 3)

    def test_kosaraju_strongly_connected(self):
        """Test Kosaraju on strongly connected component"""
        graph = DirectedGraph()
        # Create a cycle: 0->1->2->0
        matrix = [
            [0, 1, math.inf],
            [math.inf, 0, 1],
            [1, math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        sccs = graph.kosaraju()
        # All nodes should be in one SCC
        self.assertEqual(len(sccs), 1)
        self.assertEqual(set(sccs[0]), {0, 1, 2})

    def test_kosaraju_multiple_sccs(self):
        """Test Kosaraju with multiple SCCs"""
        graph = DirectedGraph()
        # Component 1: 0->1->0 (cycle)
        # Component 2: 2->3->2 (cycle)
        # Component 3: 4 (isolated)
        matrix = [
            [0, 1, math.inf, math.inf, math.inf],
            [1, 0, math.inf, math.inf, math.inf],
            [math.inf, math.inf, 0, 1, math.inf],
            [math.inf, math.inf, 1, 0, math.inf],
            [math.inf, math.inf, math.inf, math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        sccs = graph.kosaraju()
        # Should have 3 SCCs
        self.assertEqual(len(sccs), 3)

    def test_is_bipartite_empty(self):
        """Test bipartite check on empty graph"""
        graph = UndirectedGraph()
        graph.build_graph_from_matrix([])
        self.assertTrue(graph.is_bipartite_bfs())

    def test_is_bipartite_single_node(self):
        """Test bipartite check on single node"""
        graph = UndirectedGraph()
        graph.build_graph_from_matrix([[0]])
        self.assertTrue(graph.is_bipartite_bfs())

    def test_is_bipartite_tree(self):
        """Test bipartite check on tree (should be bipartite)"""
        graph = UndirectedGraph()
        # Tree: 0-1-2
        matrix = [
            [0, 1, math.inf],
            [1, 0, 1],
            [math.inf, 1, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        self.assertTrue(graph.is_bipartite_bfs())

    def test_is_bipartite_even_cycle(self):
        """Test bipartite check on even cycle (should be bipartite)"""
        graph = UndirectedGraph()
        # Even cycle: 0-1-2-3-0
        matrix = [
            [0, 1, math.inf, 1],
            [1, 0, 1, math.inf],
            [math.inf, 1, 0, 1],
            [1, math.inf, 1, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        self.assertTrue(graph.is_bipartite_bfs())

    def test_is_bipartite_odd_cycle(self):
        """Test bipartite check on odd cycle (should not be bipartite)"""
        graph = UndirectedGraph()
        # Odd cycle: 0-1-2-0 (triangle)
        matrix = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        self.assertFalse(graph.is_bipartite_bfs())

    def test_is_bipartite_disconnected(self):
        """Test bipartite check on disconnected graph"""
        graph = UndirectedGraph()
        # Two components: one bipartite (tree), one not (triangle)
        matrix = [
            [0, 1, math.inf, math.inf, math.inf, math.inf],
            [1, 0, math.inf, math.inf, math.inf, math.inf],
            [math.inf, math.inf, 0, 1, 1, math.inf],
            [math.inf, math.inf, 1, 0, 1, math.inf],
            [math.inf, math.inf, 1, 1, 0, math.inf],
            [math.inf, math.inf, math.inf, math.inf, math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        # Should return False because one component is not bipartite
        self.assertFalse(graph.is_bipartite_bfs())

    def test_mst_prim_empty(self):
        """Test Prim's MST on empty graph"""
        graph = UndirectedGraph()
        graph.build_graph_from_matrix([])
        edges, weight = graph.mst_prim_s_algorithm()
        self.assertEqual(edges, [])
        self.assertEqual(weight, 0.0)

    def test_mst_prim_single_node(self):
        """Test Prim's MST on single node"""
        graph = UndirectedGraph()
        graph.build_graph_from_matrix([[0]])
        edges, weight = graph.mst_prim_s_algorithm()
        self.assertEqual(edges, [])
        self.assertEqual(weight, 0.0)

    def test_mst_prim_simple(self):
        """Test Prim's MST on simple graph"""
        graph = UndirectedGraph()
        # Graph: 0-1 (weight 5), 0-2 (weight 3), 1-2 (weight 1)
        # MST should be: 0-2 (3), 2-1 (1) = total 4
        matrix = [
            [0, 5, 3],
            [5, 0, 1],
            [3, 1, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        edges, weight = graph.mst_prim_s_algorithm()
        self.assertEqual(weight, 4.0)
        self.assertEqual(len(edges), 2)

    def test_mst_prim_disconnected(self):
        """Test Prim's MST on disconnected graph (returns MST of first component)"""
        graph = UndirectedGraph()
        # Two disconnected components
        matrix = [
            [0, 1, math.inf, math.inf],
            [1, 0, math.inf, math.inf],
            [math.inf, math.inf, 0, 2],
            [math.inf, math.inf, 2, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        edges, weight = graph.mst_prim_s_algorithm()
        # Should return MST of first component only
        self.assertEqual(weight, 1.0)
        self.assertEqual(len(edges), 1)

    def test_mst_kruskal_empty(self):
        """Test Kruskal's MST on empty graph"""
        graph = UndirectedGraph()
        graph.build_graph_from_matrix([])
        edges, weight = graph.mst_kruskal_s_algorithm()
        self.assertEqual(edges, [])
        self.assertEqual(weight, 0.0)

    def test_mst_kruskal_single_node(self):
        """Test Kruskal's MST on single node"""
        graph = UndirectedGraph()
        graph.build_graph_from_matrix([[0]])
        edges, weight = graph.mst_kruskal_s_algorithm()
        self.assertEqual(edges, [])
        self.assertEqual(weight, 0.0)

    def test_mst_kruskal_simple(self):
        """Test Kruskal's MST on simple graph"""
        graph = UndirectedGraph()
        # Graph: 0-1 (weight 5), 0-2 (weight 3), 1-2 (weight 1)
        # MST should be: 0-2 (3), 2-1 (1) = total 4
        matrix = [
            [0, 5, 3],
            [5, 0, 1],
            [3, 1, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        edges, weight = graph.mst_kruskal_s_algorithm()
        self.assertEqual(weight, 4.0)
        self.assertEqual(len(edges), 2)

    def test_mst_kruskal_disconnected(self):
        """Test Kruskal's MST on disconnected graph (returns MST of all components)"""
        graph = UndirectedGraph()
        # Two disconnected components
        matrix = [
            [0, 1, math.inf, math.inf],
            [1, 0, math.inf, math.inf],
            [math.inf, math.inf, 0, 2],
            [math.inf, math.inf, 2, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        edges, weight = graph.mst_kruskal_s_algorithm()
        # Should return MST of all components
        self.assertEqual(weight, 3.0)  # 1 + 2
        self.assertEqual(len(edges), 2)

    def test_get_bridges_empty(self):
        """Test bridge finding on empty graph"""
        graph = UndirectedGraph()
        graph.build_graph_from_matrix([])
        bridges = graph.get_bridges()
        self.assertEqual(bridges, [])

    def test_get_bridges_single_node(self):
        """Test bridge finding on single node"""
        graph = UndirectedGraph()
        graph.build_graph_from_matrix([[0]])
        bridges = graph.get_bridges()
        self.assertEqual(bridges, [])

    def test_get_bridges_tree(self):
        """Test bridge finding on tree (all edges are bridges)"""
        graph = UndirectedGraph()
        # Tree: 0-1-2
        matrix = [
            [0, 1, math.inf],
            [1, 0, 1],
            [math.inf, 1, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        bridges = graph.get_bridges()
        # All edges in a tree are bridges
        self.assertEqual(len(bridges), 2)

    def test_get_bridges_no_bridges(self):
        """Test bridge finding on graph with no bridges"""
        graph = UndirectedGraph()
        # Cycle: 0-1-2-0 (no bridges)
        matrix = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        bridges = graph.get_bridges()
        self.assertEqual(len(bridges), 0)

    def test_get_bridges_mixed(self):
        """Test bridge finding on graph with some bridges"""
        graph = UndirectedGraph()
        # Graph: 0-1-2-3-0 (cycle) with 4-2 (bridge)
        matrix = [
            [0, 1, math.inf, 1, math.inf],
            [1, 0, 1, math.inf, math.inf],
            [math.inf, 1, 0, 1, 1],
            [1, math.inf, 1, 0, math.inf],
            [math.inf, math.inf, 1, math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        bridges = graph.get_bridges()
        # Edge 2-4 should be a bridge
        self.assertGreater(len(bridges), 0)

    def test_get_articulation_points_empty(self):
        """Test articulation point finding on empty graph"""
        graph = UndirectedGraph()
        graph.build_graph_from_matrix([])
        points = graph.get_articulation_points()
        self.assertEqual(points, [])

    def test_get_articulation_points_single_node(self):
        """Test articulation point finding on single node"""
        graph = UndirectedGraph()
        graph.build_graph_from_matrix([[0]])
        points = graph.get_articulation_points()
        self.assertEqual(points, [])

    def test_get_articulation_points_tree(self):
        """Test articulation point finding on tree"""
        graph = UndirectedGraph()
        # Tree: 0-1-2-3 (all non-leaf nodes are articulation points)
        matrix = [
            [0, 1, math.inf, math.inf],
            [1, 0, 1, math.inf],
            [math.inf, 1, 0, 1],
            [math.inf, math.inf, 1, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        points = graph.get_articulation_points()
        # Nodes 1 and 2 should be articulation points
        self.assertIn(1, points)
        self.assertIn(2, points)

    def test_get_articulation_points_no_points(self):
        """Test articulation point finding on graph with no articulation points"""
        graph = UndirectedGraph()
        # Cycle: 0-1-2-0 (no articulation points)
        matrix = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        points = graph.get_articulation_points()
        self.assertEqual(len(points), 0)

    def test_get_articulation_points_mixed(self):
        """Test articulation point finding on graph with some articulation points"""
        graph = UndirectedGraph()
        # Graph: 0-1-2-3-0 (cycle) with 4-2 (bridge)
        # Node 2 should be an articulation point
        matrix = [
            [0, 1, math.inf, 1, math.inf],
            [1, 0, 1, math.inf, math.inf],
            [math.inf, 1, 0, 1, 1],
            [1, math.inf, 1, 0, math.inf],
            [math.inf, math.inf, 1, math.inf, 0]
        ]
        graph.build_graph_from_matrix(matrix)
        points = graph.get_articulation_points()
        # Node 2 should be an articulation point
        self.assertIn(2, points)


if __name__ == '__main__':
    unittest.main()
