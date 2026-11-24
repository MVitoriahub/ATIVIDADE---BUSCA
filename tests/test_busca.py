import random

import pytest

from puzzle.state import State, GOAL_STATE
from algorithms.bfs import BFS
from algorithms.dfs import DFS
from algorithms.a_star import AStar


def apply(tiles, actions):
    """Aplica uma sequência de ações a partir de `tiles` e devolve o estado final."""
    state = State(tiles)
    for action in actions:
        state = next(s for s in state.neighbors() if s.action == action)
    return state.tiles


def random_state(moves, seed):
    rng = random.Random(seed)
    state = State(GOAL_STATE)
    for _ in range(moves):
        state = rng.choice(state.neighbors())
    return State(state.tiles)


CASES = [random_state(m, seed) for seed, m in enumerate([0, 1, 5, 10, 15, 25, 40])]


def test_neighbors_corner_and_center():
    assert {s.action for s in State(GOAL_STATE).neighbors()} == {"UP", "LEFT"}
    center = State((1, 2, 3, 4, 0, 5, 6, 7, 8))
    assert len(center.neighbors()) == 4
    assert all(child.cost == 1 and child.parent is center for child in center.neighbors())


def test_path_and_actions():
    s = State((1, 2, 3, 4, 5, 6, 0, 7, 8))
    child = next(c for c in s.neighbors() if c.action == "RIGHT")
    grandchild = next(c for c in child.neighbors() if c.action == "RIGHT")
    assert grandchild.is_goal
    assert grandchild.path() == [s, child, grandchild]
    assert grandchild.actions() == ["RIGHT", "RIGHT"]


def test_heuristic():
    assert AStar().heuristic(State(GOAL_STATE)) == 0
    assert AStar().heuristic(State((1, 2, 3, 4, 5, 6, 0, 7, 8))) == 2


@pytest.mark.parametrize("initial", CASES)
def test_bfs_and_astar_are_optimal_and_agree(initial):
    bfs = BFS().search(initial)
    astar = AStar().search(initial)
    assert bfs.found and astar.found
    assert bfs.depth == astar.depth == bfs.path_cost == astar.path_cost
    assert apply(initial.tiles, bfs.actions) == GOAL_STATE
    assert apply(initial.tiles, astar.actions) == GOAL_STATE
    assert astar.nodes_expanded <= bfs.nodes_expanded or bfs.depth <= 1


@pytest.mark.parametrize("initial", CASES)
def test_dfs_finds_valid_solution(initial):
    result = DFS().search(initial)
    assert result.found
    assert result.depth <= 50
    assert len(result.actions) == result.depth
    assert apply(initial.tiles, result.actions) == GOAL_STATE


def test_unsolvable():
    initial = State((2, 8, 3, 1, 6, 4, 7, 0, 5))
    assert not initial.is_solvable
    for algorithm in (BFS(), DFS(), AStar()):
        assert not algorithm.search(initial).found


def test_already_goal():
    for algorithm in (BFS(), DFS(), AStar()):
        result = algorithm.search(State(GOAL_STATE))
        assert result.found and result.depth == 0 and result.actions == []
