import sys
from collections import deque
import functools
import heapq
from pydantic import BaseModel, conint, model_validator, NonNegativeInt
from typing import Optional, Tuple, List
from typing_extensions import Self
from enum import Enum
import random
import numpy as np
import math
import argparse


class Problem:
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return any(x is state for x in self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value. Hill Climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError


class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [
            self.child_node(problem, action) for action in problem.actions(self.state)
        ]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(
            next_state,
            self,
            action,
            problem.path_cost(self.path_cost, self.state, action, next_state),
        )
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # We use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)


class Solution:
    """
    Represents a solution to a search problem, with a given solution node, final
    frontier, and explored note set.
    """

    def __init__(self, node: Optional[Node], frontier: list, explored: set):
        self.node = node
        self.frontier = frontier
        self.explored = explored

    def __repr__(self):
        if not self.node:
            return "No solution found."
        lines = []

        path = self.node.path()
        for i, node in enumerate(path):
            if hasattr(node, "h"):
                # Print using A* algorithm representation
                lines.append(
                    f"({node.depth}, {node.path_cost}, {node.action}, {node.h}, {node.state})"
                )
            else:
                # Print using blind search representation
                lines.append(
                    f"({node.depth}, {node.path_cost}, {node.action}, {node.state})"
                )

                # If not the final node, print the operator leading to the next node
                if i < len(path) - 1:
                    next_node = path[i + 1]
                    lines.append(f"{next_node.action}")

        # Print the final statistics
        # lines.append(f"({self.node.depth}, {self.node.path_cost}, {self.node.state})")
        lines.append(f"Total number of items in explored list: {len(self.explored)}")
        lines.append(f"Total number of items in frontier: {len(self.frontier)}")

        return "\n".join(lines)


def depth_first_graph_search(problem) -> Solution:
    """
    [Figure 3.7]
    Search the deepest nodes in the search tree first.
    Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Does not get trapped by loops.
    If two paths reach a state, only use the first one.
    """
    frontier = [(Node(problem.initial))]  # Stack

    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return Solution(node, frontier, explored)
        explored.add(node.state)
        frontier.extend(
            child
            for child in node.expand(problem)
            if child.state not in explored and child not in frontier
        )
    return Solution(None, frontier, explored)


def breadth_first_graph_search(problem):
    """[Figure 3.11]
    Note that this function can be implemented in a
    single line as below:
    return graph_search(problem, FIFOQueue())
    """
    node = Node(problem.initial)
    frontier = deque([node])
    explored = set()
    if problem.goal_test(node.state):
        return Solution(node, frontier, explored)
    while frontier:
        node = frontier.popleft()
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return Solution(child, frontier, explored)
                frontier.append(child)
    return Solution(None, frontier, explored)


class PriorityQueue:
    """A Queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first.
    If order is 'min', the item with minimum f(x) is
    returned first; if order is 'max', then it is the item with maximum f(x).
    Also supports dict-like lookup."""

    def __init__(self, order="min", f=lambda x: x):
        self.heap = []
        if order == "min":
            self.f = f
        elif order == "max":  # now item with max f(x)
            self.f = lambda x: -f(x)  # will be popped first
        else:
            raise ValueError("Order must be either 'min' or 'max'.")

    def append(self, item):
        """Insert item at its correct position."""
        heapq.heappush(self.heap, (self.f(item), item))

    def extend(self, items):
        """Insert each item in items at its correct position."""
        for item in items:
            self.append(item)

    def pop(self):
        """Pop and return the item (with min or max f(x) value)
        depending on the order."""
        if self.heap:
            return heapq.heappop(self.heap)[1]
        else:
            raise Exception("Trying to pop from empty PriorityQueue.")

    def __len__(self):
        """Return current capacity of PriorityQueue."""
        return len(self.heap)

    def __contains__(self, key):
        """Return True if the key is in PriorityQueue."""
        return any([item == key for _, item in self.heap])

    def __getitem__(self, key):
        """Returns the first value associated with key in PriorityQueue.
        Raises KeyError if key is not present."""
        for value, item in self.heap:
            if item == key:
                return value
        raise KeyError(str(key) + " is not in the priority queue")

    def __delitem__(self, key):
        """Delete the first occurrence of key."""
        try:
            del self.heap[[item == key for _, item in self.heap].index(True)]
        except ValueError:
            raise KeyError(str(key) + " is not in the priority queue")
        heapq.heapify(self.heap)


def best_first_graph_search(problem, f, display=False):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, "f")
    node = Node(problem.initial)
    frontier = PriorityQueue("min", f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            if display:
                print(
                    len(explored),
                    "paths have been expanded and",
                    len(frontier),
                    "paths remain in the frontier",
                )
            return Solution(node, frontier, explored)
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    return Solution(None, frontier, explored)


def memoize(fn, slot=None, maxsize=32):
    """Memoize fn: make it remember the computed value for any argument list.
    If slot is specified, store result in that slot of first argument.
    If slot is false, use lru_cache for caching the values."""
    if slot:

        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val

    else:

        @functools.lru_cache(maxsize=maxsize)
        def memoized_fn(*args):
            return fn(*args)

    return memoized_fn


def astar_search(problem, h=None, display=False):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, "h")
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n), display)


class DrillingActions(Enum):
    """
    Defines the possible actions for the drilling problem:
      - ADVANCE: Move forward in the current direction.
      - TURN_LEFT: Rotate 45 degrees to the left.
      - TURN_RIGHT: Rotate 45 degrees to the right.
    """

    ADVANCE = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2


class Orientation(Enum):
    """
    Represents the orientation of the drilling machine:
      - NORTH, NORTHEAST, EAST, SOUTHEAST, SOUTH, SOUTHWEST, WEST, NORTHWEST
        Provides methods to check directions, convert to offsets, and turn left/right.
    """

    NORTH = 0
    NORTHEAST = 1
    EAST = 2
    SOUTHEAST = 3
    SOUTH = 4
    SOUTHWEST = 5
    WEST = 6
    NORTHWEST = 7

    def is_west(self):
        """Checks if the orientation is Westward."""
        return self in [Orientation.WEST, Orientation.NORTHWEST, Orientation.SOUTHWEST]

    def is_east(self):
        """Checks if the orientation is Eastward."""
        return self in [Orientation.EAST, Orientation.NORTHEAST, Orientation.SOUTHEAST]

    def is_north(self):
        """Checks if the orientation is Northward."""
        return self in [Orientation.NORTH, Orientation.NORTHEAST, Orientation.NORTHWEST]

    def is_south(self):
        """Checks if the orientation is Southward."""
        return self in [Orientation.SOUTH, Orientation.SOUTHEAST, Orientation.SOUTHWEST]

    def to_offset(self):
        """
        Converts the orientation to an (x, y) offset.

        Returns:
          tuple: A tuple representing the (x, y) offset for the orientation.
        """
        if self == Orientation.NORTH:
            return (0, -1)
        elif self == Orientation.NORTHEAST:
            return (1, -1)
        elif self == Orientation.EAST:
            return (1, 0)
        elif self == Orientation.SOUTHEAST:
            return (1, 1)
        elif self == Orientation.SOUTH:
            return (0, 1)
        elif self == Orientation.SOUTHWEST:
            return (-1, 1)
        elif self == Orientation.WEST:
            return (-1, 0)
        elif self == Orientation.NORTHWEST:
            return (-1, -1)

    def turn_left(self):
        """
        Rotates the orientation 45 degrees to the left.

        Returns:
          Orientation: The new orientation after turning left.
        """
        return Orientation((self.value - 1) % 8)

    def turn_right(self):
        """
        Rotates the orientation 45 degrees to the right.

        Returns:
          Orientation: The new orientation after turning right.
        """
        return Orientation((self.value + 1) % 8)


class DrillingState(BaseModel):
    """
    Represents the state of the drilling machine.

    Attributes:
      x (int): The x-coordinate of the drilling machine.
      y (int): The y-coordinate of the drilling machine.
      orientation (Orientation): The orientation of the drilling machine. If
        None, the drilling machine is assumed to be north. If this state
        represents a goal state, the goal orientation is ignored.
    """

    x: NonNegativeInt
    y: NonNegativeInt
    orientation: Optional[Orientation] = None

    def __hash__(self):
        """
        Returns the hash value of the state.

        Returns:
          int: The hash value of the state.
        """
        return hash(tuple(self.__dict__.values()))

    def __lt__(self, node):
        """
        Checks if the current state is less than another state.

        Args:
          node (DrillingState): The other state to compare with.

        Returns:
          bool: True if the current state is less than the other state, False otherwise.
        """
        return self.x < node.x and self.y < node.y


class DrillingMap(BaseModel):
    """
    Represents the map of the drilling machine.

    Attributes:
      width (int): The width of the map.
      height (int): The height of the map.
      map (list[list[int]]): The map of the drilling machine.
    """

    width: NonNegativeInt
    height: NonNegativeInt
    map: List[List[conint(ge=1, le=9)]]

    @model_validator(mode="after")
    def validate_map_size(self):
        if len(self.map) != self.height:
            raise ValueError(f"The number of rows in the map must be {self.height}")
        if any(len(row) != self.width for row in self.map):
            raise ValueError(f"All rows in the map must have {self.width} columns")
        return self


class DrillingSearch(Problem):

    def __init__(self, initial: DrillingState, goal: DrillingState, map: DrillingMap):
        # If not specified, assume north orientation for the machine.
        initial.orientation = initial.orientation or Orientation.NORTH
        # Ensure start and goal positions are actually on the map.
        assert 0 <= initial.x < map.width
        assert 0 <= initial.y < map.height
        assert 0 <= goal.x < map.width
        assert 0 <= goal.y < map.height
        # Initialize as normal.
        super().__init__(initial, goal)
        self.map = map

    @classmethod
    def from_file(cls, file_path: str) -> Self:
        with open(file_path, "r") as file:
            content = file.read()
        return DrillingSearch.from_str(content)

    @classmethod
    def from_str(cls, string: str) -> Self:
        # Separate into multiple lines, ignoring empty lines.
        lines = [line for line in string.splitlines() if line]

        # Parse map size in first line
        height, width = map(int, lines[0].strip().split())

        # Parse map configuration from second to height+1 lines.
        map_data = []
        for line in lines[1:-2]:
            map_data.append(list(map(int, line.strip().split())))

        # Create DrillingMap object from map information
        drilling_map = DrillingMap(width=width, height=height, map=map_data)

        # Parse initial state from the second to last line.
        # The orientation represented as a number coincides with the Enum.
        # If set to 8, set it to None to use the default orientation value.
        init_y, init_x, init_o = map(int, lines[-2].strip().split())
        initial_orientation = None if init_o == 8 else Orientation(init_o)
        initial_state = DrillingState(
            x=init_x, y=init_y, orientation=initial_orientation
        )

        # Parse goal state from the last line.
        goal_x, goal_y, goal_o = map(int, lines[-1].strip().split())
        target_orientation = None if goal_o == 8 else Orientation(goal_o)
        goal_state = DrillingState(x=goal_x, y=goal_y, orientation=target_orientation)

        return cls(initial_state, goal_state, drilling_map)

    def to_str(self) -> str:
        res = f"{self.map.height} {self.map.width}\n"
        for row in self.map.map:
            res += ' '.join([str(x) for x in row]) + '\n'

        orient = self.initial.orientation.value if self.initial.orientation else 8
        res += f"{self.initial.y} {self.initial.x} {orient}\n"

        orient = self.goal.orientation.value if self.goal.orientation else 8
        res += f"{self.goal.y} {self.goal.x} {orient}"

        return res
    
    def to_file(self, file_path: str):
        with open(file_path, "w") as file:
            content = self.to_str()
            file.write(content)

    @classmethod
    def random(cls, width: int, height: int, seed: int = None) -> Self:
        if seed:
            random.seed(seed)
        random_map = DrillingMap(
            width=width,
            height=height,
            map=[[random.randint(1, 9) for _ in range(width)] for _ in range(height)],
        )
        initial_state = DrillingState(x=0, y=0, orientation=None)
        goal_state = DrillingState(x=width - 1, y=height - 1, orientation=None)
        return cls(initial_state, goal_state, random_map)

    def actions(self, state: DrillingState):
        # Initialy we cant take all possible actions.
        possible_actions = {a:"" for a in DrillingActions}

        # Remove the advance action if we are at the border
        # and advancing would result in us leaving the map.
        if state.x == 0 and state.orientation.is_west():
            possible_actions.pop(DrillingActions.ADVANCE, None)
        elif state.x == self.map.width - 1 and state.orientation.is_east():
            possible_actions.pop(DrillingActions.ADVANCE, None)

        if state.y == 0 and state.orientation.is_north():
            possible_actions.pop(DrillingActions.ADVANCE, None)
        elif state.y == self.map.height - 1 and state.orientation.is_south():
            possible_actions.pop(DrillingActions.ADVANCE, None)

        return list(possible_actions.keys())

    def result(self, state: DrillingState, action: DrillingActions):
        if action == DrillingActions.ADVANCE:
            # If advancing, just move (x,y) based on the orientation.
            x_offset, y_offset = state.orientation.to_offset()
            return DrillingState(
                x=state.x + x_offset,
                y=state.y + y_offset,
                orientation=state.orientation,
            )
        elif action == DrillingActions.TURN_LEFT:
            return DrillingState(
                x=state.x, y=state.y, orientation=state.orientation.turn_left()
            )
        elif action == DrillingActions.TURN_RIGHT:
            return DrillingState(
                x=state.x, y=state.y, orientation=state.orientation.turn_right()
            )

    def goal_test(self, state: DrillingState):
        # Check if we are at the goal position (if the goal
        # orientation is not specified, we ignore it).
        return (
            state.x == self.goal.x
            and state.y == self.goal.y
            and (
                not self.goal.orientation or self.goal.orientation == state.orientation
            )
        )

    def path_cost(
        self,
        c: int,
        state1: DrillingState,
        action: DrillingActions,
        state2: DrillingState,
    ): 
        if action == DrillingActions.ADVANCE:
            # The cost of advancing is the hardness of
            # the rock we are advancing into.
            hardness = self.map.map[state2.y][state2.x]
            return c + hardness
        elif action == DrillingActions.TURN_LEFT:
            # Turning just increses the cost by 1.
            return c + 1
        elif action == DrillingActions.TURN_RIGHT:
            # Turning just increses the cost by 1.
            return c + 1

    def h(self, node):
        """
        Args:
            node (Node): The current state of the robot, including its position, orientation, and other attributes.

        Returns:
            float: Estimated cost to the goal using Manhattan distance, considering the average terrain hardness.
        """
        # Chebyshev distance
        x = node.state.x
        y = node.state.y
        goal_x = self.goal.x
        goal_y = self.goal.y

        average_hardness = sum(sum(row) for row in self.map.map) / (
            self.map.width * self.map.height
        )
        return max(abs(x - goal_x), abs(y - goal_y)) + average_hardness

    def h2(self, node):
        """
        Args:
            node (Node): The current state of the robot, including its position, orientation, and other attributes.

        Returns:
            float: Estimated cost to the goal using Euclidean distance and a rotation penalty if applicable.
        """
        # weightedL2
        x = node.state.x
        y = node.state.y
        goal_x = self.goal.x
        goal_y = self.goal.y
        penalty = 0
        if (
            node.action == DrillingActions.TURN_LEFT
            or node.action == DrillingActions.TURN_RIGHT
        ):
            penalty = self.map.map[y][x]
        return np.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2) + penalty


def parse_arguments():
    parser = argparse.ArgumentParser(description="Search space input options")

    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument('-f', '--file', type=str, help='File path for the map, start, and goal state')
    input_group.add_argument('-s', '--size', nargs=2, metavar=('W', 'H'), type=int, help='Size of the random input map (W columns, H rows)')
    input_group.add_argument('--stdin', action='store_true', help='Read input from stdin (default if no file or size is provided)')
    parser.add_argument('-a', '--algorithm', type=str, choices=['bfs', 'dfs', 'astr'], help='Search algorithm to use (Breadth-first, Depth-first, or A*)')
    parser.add_argument('--save', type=str, help='File path to save the randomly generated map (only applicable if using --size)')
    parser.add_argument('--seed', type=int, help='Seed for generating the random map (only applicable if using --size)')
    heuristic_group = parser.add_mutually_exclusive_group()
    heuristic_group.add_argument('--h1', action='store_true', help='Use heuristic h1 (only applicable for A* algorithm)')
    heuristic_group.add_argument('--h2', action='store_true', help='Use heuristic h2 (only applicable for A* algorithm)')
    args = parser.parse_args()

    if args.algorithm != 'astr' and (args.h1 or args.h2):
        parser.error("Heuristics are only applicable for the A* algorithm.")

    if not args.size and (args.save or args.seed):
        parser.error("--save and --seed are only applicable when generating a random map using --size.")

    return args

if __name__ == "__main__":
    args = parse_arguments()

    if args.file:
        problem = DrillingSearch.from_file(args.file)
    elif args.size:
        w,h = args.size
        problem = DrillingSearch.random(w,h, seed=args.seed)
    else:
        input_data = sys.stdin.read()
        problem = DrillingSearch.from_str(input_data)

    if not args.algorithm:
        print(problem.to_str())
        if args.save:
            problem.to_file(args.save)
    else:
        print(f"Map: {problem.map}")
        print(f"Initial: {problem.initial}")
        print(f"Goal:    {problem.goal}")
        print(f"Algorithm: {args.algorithm}")
        if args.algorithm == "astr":
            print(f"Heuristic: {'h1' if args.h1 else 'h2' if args.h2 else None}")
        print("\nTrace:")
        if args.algorithm == "astr":
            h = problem.h if args.h1 else problem.h2 if args.h2 else None
            sol = astar_search(problem, h=h)
        elif args.algorithm == "bfs":
            sol = breadth_first_graph_search(problem)
        else:
            sol = depth_first_graph_search(problem)
            
        print(sol)