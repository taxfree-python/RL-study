from enum import Enum

import numpy as np


class State:
    def __init__(self, row=-1, col=-1) -> None:
        self.row = row
        self.col = col

    def __repr__(self) -> str:
        return f"State: [{self.row}, {self.col}]"

    def clone(self) -> "State":
        return State(self.row, self.col)

    def __hash__(self) -> int:
        return hash((self.row, self.col))

    def __eq__(self, other) -> bool:
        return self.row == other.row and self.col == other.col


class Action(Enum):
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2


class Environment:
    def __init__(self, grid, move_prob=0.8) -> None:
        self.grid = grid
        self.agent_state = State()
        self.default_reward = -0.04
        self.move_prob = move_prob
        self.reset()

    @property
    def row_length(self) -> int:
        return len(self.grid)

    @property
    def column_length(self) -> int:
        return len(self.grid[0])

    @property
    def actions(self) -> list[Action]:
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

    @property
    def states(self) -> list[State]:
        states = []
        for row in range(self.row_length):
            for col in range(self.column_length):
                if self.grid[row][col] != 9:
                    states.append(State(row, col))
        return states

    def transit_func(self, state, action) -> dict[State, float]:
        transition_probs = {}
        if not self.can_action_at(state):
            return transition_probs

        opposite_direction = Action(action.value * -1)
        for a in self.actions:
            prob = 0
            if a == action:
                prob = self.move_prob
            elif a != opposite_direction:
                prob = (1 - self.move_prob) / 2

            next_state = self._move(state, a)
            if next_state not in transition_probs:
                transition_probs[next_state] = prob
            else:
                transition_probs[next_state] += prob

        return transition_probs

    def can_action_at(self, state) -> bool:
        if self.grid[state.row][state.col] == 0:
            return True
        else:
            return False

    def _move(self, state, action) -> State:
        if not self.can_action_at(state):
            raise Exception("Can't move from here!")

        next_state = state.clone()

        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.col -= 1
        elif action == Action.RIGHT:
            next_state.col += 1

        if next_state.row < 0:
            next_state.row = 0
        elif next_state.row >= self.row_length:
            next_state.row = self.row_length - 1

        if next_state.col < 0:
            next_state.col = 0
        elif next_state.col >= self.column_length:
            next_state.col = self.column_length - 1

        if self.grid[next_state.row][next_state.col] == 9:
            next_state = state

        return next_state

    def reward_func(self, state) -> tuple[float, bool]:
        reward = self.default_reward
        done = False

        attribute = self.grid[state.row][state.col]
        if attribute == 1:
            reward = 1
            done = True
        elif attribute == -1:
            reward = -1
            done = True

        return reward, done

    def reset(self) -> State:
        self.agent_state = State(self.row_length - 1, 0)
        return self.agent_state

    def step(self, action) -> tuple[State, float, bool]:
        next_state, reward, done = self.transit(self.agent_state, action)
        if next_state is not None:
            self.agent_state = next_state

        return next_state, reward, done

    def transit(self, state, action) -> tuple[State, float, bool]:
        transition_probs = self.transit_func(state, action)

        if len(transition_probs) == 0:
            return None, None, True

        next_states = []
        probs = []
        for s in transition_probs:
            next_states.append(s)
            probs.append(transition_probs[s])

        next_state = np.random.choice(next_states, p=probs)
        reward, done = self.reward_func(next_state)

        return next_state, reward, done
