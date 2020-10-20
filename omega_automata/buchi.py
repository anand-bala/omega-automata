from typing import (
    Type,
    Tuple,
    Union,
    Callable,
    Hashable,
    Optional,
    FrozenSet,
    NamedTuple
)
from collections.abc import Set, Iterable

import networkx as nx
from dd.autoref import BDD  # TODO: Do I need to make this abstract?
from dd.autoref import Function

from omega_automata.lts import OmegaAutomaton

State = Hashable


class Edge(NamedTuple):
    src: State
    dst: State
    key: Hashable


class BuchiAutomaton(OmegaAutomaton):

    _bdd_mgr: BDD
    _epsilon: Function

    _graph: nx.MultiDiGraph
    _initial_state: State
    _acceptance_set: Set[Edge]

    def __init__(self, *, bdd: Optional[BDD] = None):
        self._bdd_mgr = bdd if bdd is not None else BDD()
        self._epsilon = self._bdd_mgr.false
        self._atomic_props = self._bdd_mgr

        self._graph = nx.MultiDiGraph()

        self._initial_state = 0
        self._acceptance_set = set()

    @property
    def bdd_manager(self) -> BDD:
        return self._bdd_mgr

    @property
    def acceptance_set(self) -> Set[Edge]:
        return self.acceptance_set

    @property
    def atomic_propositions(self) -> FrozenSet[str]:
        return frozenset(self._bdd_mgr.vars.keys())

    def add_atomic_proposition(self, ap: str):
        self._bdd_mgr.add_var(ap)

    def add_state(self, s: State):
        self._graph.add_node(s)

    @property
    def num_states(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def states(self) -> Set[State]:
        return frozenset(self._graph.nodes)

    def add_edge(self, src: State, dst: State, label: Function):
        self._graph.add_edge(src, dst, label=label)

    @property
    def num_edges(self) -> int:
        return self._graph.number_of_edges()

    @property
    def edges(self) -> Set[Edge]:
        return frozenset(self.edges)

    def is_accepting(self, edge: Edge) -> bool:
        return edge in self.acceptance_set

    @property
    def initial_state(self) -> State:
        return self._initial_state

    @initial_state.setter
    def initial_state(self, s: State):
        if s in self._graph.nodes:
            self._initial_state = s
        raise ValueError(
            "can't make State {} an initial state as it is not in the Automaton."
        )

    def successors(self, src: State, letter: Function) -> Iterable[Edge]:
        for dst in self._graph.successors(src):
            for key in self._graph[dst].keys():
                label: Function = self._graph[src][dst][key]["label"]
                if letter == self.epsilon_prop and self.epsilon_prop == label:
                    yield Edge(src=src, dst=dst, key=key)
                elif letter <= label:
                    yield Edge(src=src, dst=dst, key=key)

    @property
    def epsilon_prop(self) -> Function:
        return self._epsilon

    @epsilon_prop.setter
    def epsilon_prop(self, eps: Function):
        self._epsilon = eps

    def is_deterministic(self, complete: bool = True) -> bool:
        if self.num_states == 0:
            return not complete
        for src in self._graph.nodes:
            whole: Function = self.bdd_manager.false
            for dst in self._graph[src].keys():
                for key in self._graph[dst].keys():
                    label: Function = self._graph[src][dst][key]["label"]
                    if label <= (~whole):
                        whole |= label
                    else:
                        return False

            if complete and whole != self.bdd_manager.true:
                return False
        return True

    def is_limit_deterministic(self) -> bool:
        pass

    def is_complete(self) -> bool:
        if self.num_states == 0:
            return False
        for src in self._graph.nodes:
            whole: Function = self.bdd_manager.false
            for dst in self._graph[src].keys():
                for key in self._graph[dst].keys():
                    label: Function = self._graph[src][dst][key]["label"]
                    whole = whole | label
            if whole != self.bdd_manager.true:
                return False
        return True

    def strongly_connected_components(
        self,
        *,
        state_filter: Optional[Callable[[State], bool]] = None,
        edge_filter: Optional[Callable[[State, State], bool]] = None,
    ) -> Iterable[Set[State]]:
        G = nx.subgraph_view(
            self._graph, filter_node=state_filter, filter_edge=edge_filter
        )
        return nx.strongly_connected_components(G)
