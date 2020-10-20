import abc
from abc import ABC, abstractmethod
from typing import Set, Tuple, Union, Callable, Hashable, Iterable, Optional, FrozenSet

from dd._abc import BDD as AbstractBDD
from dd._abc import Operator

State = Hashable
Edge = Hashable

AtomicProposition = Hashable


class OmegaAutomaton(ABC):
    @property
    @abstractmethod
    def bdd_manager(self) -> AbstractBDD:
        """Get the BDD manager for this automaton."""
        pass

    @property
    @abstractmethod
    def atomic_propositions(self) -> FrozenSet[AtomicProposition]:
        """The set of atomic propositions that can be present in any transition label."""
        pass

    @abstractmethod
    def add_atomic_propositions(self, *ap: str):
        pass

    @abstractmethod
    def add_state(self, s: State):
        """Add a state to the automaton."""
        pass

    @property
    @abstractmethod
    def num_states(self) -> int:
        """Number of states in the automaton."""
        pass

    @property
    @abstractmethod
    def states(self) -> Iterable[State]:
        """The set of states in the automaton."""
        pass

    @abstractmethod
    def add_edge(self, src: State, dst: State, label):
        """Add a labelled transition to the automaton."""
        pass

    @property
    @abstractmethod
    def num_edges(self) -> int:
        """Number of edges in the automaton."""
        pass

    @property
    @abstractmethod
    def edges(self) -> Iterable[Edge]:
        """Set of all edges in the automaton."""
        pass

    @property
    @abstractmethod
    def initial_state(self) -> State:
        """Get or set the initial state in the automaton."""
        pass

    @abstractmethod
    def successors(self, src: State, letter) -> Iterable[Edge]:
        """Get the set of successors for the state, given an expression of atomic predicates."""
        pass

    @abstractmethod
    def is_deterministic(self, complete: bool = True) -> bool:
        pass

    @abstractmethod
    def is_limit_deterministic(self) -> bool:
        pass

    @abstractmethod
    def is_complete(self) -> bool:
        pass

    @abstractmethod
    def strongly_connected_components(
        self,
        *,
        state_filter: Optional[Callable[[State], bool]] = None,
        edge_filter: Optional[Callable[[State, State], bool]] = None,
    ) -> Iterable[Set[State]]:
        pass
