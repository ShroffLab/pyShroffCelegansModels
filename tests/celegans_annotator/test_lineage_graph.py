"""Tests for celegans_annotator.lineage_graph module."""

import networkx as nx
import numpy as np
import pytest

from celegans_annotator.lineage_graph import (
    build_lineage_graph,
    get_daughters,
    graph_to_solution_tracks,
)

# ---- Test data fixtures ----


def _make_test_data():
    """Create test nuclei data with known division patterns.

    Lineage:
        t=0: P0
        t=1: AB (from P0), P1 (from P0)
        t=2: ABa (from AB), ABp (from AB), EMS (from P1), P2 (from P1)
        t=3: ABa, ABp, MS (from EMS), E (from EMS), P2
        t=4: ABa, ABp, MS, E, P2
    """
    coords = []
    labels = []

    # t=0: P0
    coords.append([0, 10, 20, 30])
    labels.append("P0")

    # t=1: AB and P1 (from P0 division, hard-coded rule)
    coords.append([1, 10, 20, 32])
    labels.append("AB")
    coords.append([1, 15, 25, 35])
    labels.append("P1")

    # t=2: ABa, ABp (from AB, prefix rule), EMS, P2 (from P1, hard-coded)
    coords.append([2, 10, 20, 34])
    labels.append("ABa")
    coords.append([2, 12, 22, 36])
    labels.append("ABp")
    coords.append([2, 15, 25, 37])
    labels.append("EMS")
    coords.append([2, 17, 27, 39])
    labels.append("P2")

    # t=3: ABa, ABp continue; MS, E from EMS; P2 continues
    coords.append([3, 10, 20, 36])
    labels.append("ABa")
    coords.append([3, 12, 22, 38])
    labels.append("ABp")
    coords.append([3, 15, 25, 39])
    labels.append("MS")
    coords.append([3, 16, 26, 40])
    labels.append("E")
    coords.append([3, 17, 27, 41])
    labels.append("P2")

    # t=4: all continue
    coords.append([4, 10, 20, 38])
    labels.append("ABa")
    coords.append([4, 12, 22, 40])
    labels.append("ABp")
    coords.append([4, 15, 25, 41])
    labels.append("MS")
    coords.append([4, 16, 26, 42])
    labels.append("E")
    coords.append([4, 17, 27, 43])
    labels.append("P2")

    return np.array(coords, dtype=float), labels


# ---- get_daughters tests ----


class TestGetDaughters:
    def test_hardcoded_rule_P0(self):
        """P0 should divide into AB and P1 via hard-coded rules."""
        candidates = [("AB", 1), ("P1", 2), ("SomeOther", 3)]
        result = get_daughters("P0", candidates)
        assert sorted(result) == [1, 2]

    def test_hardcoded_rule_EMS(self):
        """EMS should divide into MS and E via hard-coded rules."""
        candidates = [("MS", 10), ("E", 11), ("EMSa", 12)]
        result = get_daughters("EMS", candidates)
        assert sorted(result) == [10, 11]

    def test_prefix_fallback_AB(self):
        """AB should divide into ABa and ABp via prefix matching."""
        candidates = [("ABa", 5), ("ABp", 6), ("ABal", 7)]
        result = get_daughters("AB", candidates)
        # ABa and ABp are len(AB)+1=3, ABal is len 4 so excluded
        assert sorted(result) == [5, 6]

    def test_no_daughters(self):
        """Cell with no matching daughters returns empty list."""
        candidates = [("X", 1), ("Y", 2)]
        result = get_daughters("ABa", candidates)
        assert result == []

    def test_custom_rules(self):
        """Custom division rules override the defaults."""
        custom_rules = {"Foo": ["Bar", "Baz"]}
        candidates = [("Bar", 1), ("Baz", 2), ("AB", 3)]
        result = get_daughters("Foo", candidates, division_rules=custom_rules)
        assert sorted(result) == [1, 2]

    def test_hardcoded_rule_partial_match(self):
        """If only one daughter from a rule is present, return just that one."""
        # P2 -> C + P3, but only C is present
        candidates = [("C", 5)]
        result = get_daughters("P2", candidates)
        assert result == [5]


# ---- build_lineage_graph tests ----


class TestBuildLineageGraph:
    @pytest.fixture()
    def test_data(self):
        return _make_test_data()

    def test_node_count(self, test_data):
        coords, labels = test_data
        graph = build_lineage_graph(coords, labels)
        assert graph.number_of_nodes() == len(labels)

    def test_continuation_edges(self, test_data):
        """Cells with the same name at adjacent timepoints should be connected."""
        coords, labels = test_data
        graph = build_lineage_graph(coords, labels)

        # ABa at t=2 (index 3) should connect to ABa at t=3 (index 7)
        assert graph.has_edge(3, 7)
        # ABa at t=3 (index 7) should connect to ABa at t=4 (index 12)
        assert graph.has_edge(7, 12)

    def test_hardcoded_division_P0(self, test_data):
        """P0 at t=0 should have edges to AB and P1 at t=1."""
        coords, labels = test_data
        graph = build_lineage_graph(coords, labels)

        # P0 is index 0, AB is index 1, P1 is index 2
        assert graph.has_edge(0, 1)
        assert graph.has_edge(0, 2)

    def test_prefix_division_AB(self, test_data):
        """AB at t=1 should have edges to ABa and ABp at t=2."""
        coords, labels = test_data
        graph = build_lineage_graph(coords, labels)

        # AB is index 1, ABa is index 3, ABp is index 4
        assert graph.has_edge(1, 3)
        assert graph.has_edge(1, 4)

    def test_hardcoded_division_EMS(self, test_data):
        """EMS at t=2 should have edges to MS and E at t=3."""
        coords, labels = test_data
        graph = build_lineage_graph(coords, labels)

        # EMS is index 5, MS is index 9, E is index 10
        assert graph.has_edge(5, 9)
        assert graph.has_edge(5, 10)

    def test_node_attributes(self, test_data):
        """Nodes should have time, z, y, x, pos, and name attributes."""
        coords, labels = test_data
        graph = build_lineage_graph(coords, labels)

        node_0 = graph.nodes[0]
        assert node_0["time"] == 0
        assert node_0["z"] == 10.0
        assert node_0["y"] == 20.0
        assert node_0["x"] == 30.0
        assert node_0["pos"] == [10.0, 20.0, 30.0]
        assert node_0["name"] == "P0"

    def test_no_self_loops(self, test_data):
        """Graph should have no self-loops."""
        coords, labels = test_data
        graph = build_lineage_graph(coords, labels)
        assert nx.number_of_selfloops(graph) == 0

    def test_last_timepoint_no_outgoing(self, test_data):
        """Nodes at the last timepoint should have no outgoing edges."""
        coords, labels = test_data
        graph = build_lineage_graph(coords, labels)

        # Last timepoint is t=4, indices 12-16
        for idx in range(12, 17):
            assert graph.out_degree(idx) == 0


# ---- graph_to_solution_tracks tests ----


class TestGraphToSolutionTracks:
    def test_creates_solution_tracks(self):
        coords, labels = _make_test_data()
        graph = build_lineage_graph(coords, labels)
        tracks = graph_to_solution_tracks(graph)
        assert len(list(tracks.graph.node_ids())) == len(labels)

    def test_custom_scale(self):
        coords, labels = _make_test_data()
        graph = build_lineage_graph(coords, labels)
        scale = [1.0, 0.5, 0.3, 0.3]
        tracks = graph_to_solution_tracks(graph, scale=scale)
        assert list(tracks.scale) == scale
