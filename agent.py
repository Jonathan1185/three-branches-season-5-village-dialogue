"""A role-based villager that follows a terrain-aware daily route."""

from collections import deque

from sandbox.observation_types import ThreeBranchesAction, ThreeBranchesObservation
from sandbox.village import action, day, geometry, layout, me, people, props


class Agent:
    """Visits village landmarks by day, sleeps at home by night, and greets visitors."""

    def reset(self, seed: int, observation: ThreeBranchesObservation) -> None:
        """Give this villager a deterministic role, route, and landmarks."""

        home = me.home(observation)
        door = layout.doorway(observation, home) if home != "none" else None
        home_center = layout.building_center(observation, home) if home != "none" else None
        self.sleep_cell = (
            layout.nearest_walkable(observation, home_center) if home_center is not None else None
        )
        pump = next((prop for prop in props.all(observation) if prop["type"] == "pump"), None)
        benches = [prop for prop in props.all(observation) if prop["type"] == "bench"]

        # Homes provide stable identity without Python's randomized hash function.
        identity = sum(ord(character) for character in home) + seed
        bench = benches[identity % len(benches)] if benches else None
        routines = (
            ("water carrier", (("collect water", pump, "visit"), ("rest after carrying water", bench, "rest"))),
            ("caretaker", (("inspect the pump", pump, "visit"), ("plan at the bench", bench, "rest"))),
            ("neighbor", (("meet neighbors at the bench", bench, "rest"), ("fetch water", pump, "visit"))),
        )
        self.role, routine = routines[identity % len(routines)]
        self.path: list[dict] = []

        if door is not None:
            door_cell = layout.nearest_walkable(observation, door)
            if door_cell is not None:
                self.path.append({"task": "leave home", "cell": door_cell, "kind": "walk"})

        for task, landmark, kind in routine:
            if landmark is not None:
                target = layout.cell_center(observation, landmark["cell"])
                target_cell = layout.nearest_walkable(observation, target)
                if target_cell is not None:
                    self.path.append({"task": task, "cell": target_cell, "kind": kind})

        if door is not None:
            door_cell = layout.nearest_walkable(observation, door)
            if door_cell is not None:
                self.path.append({"task": "return home", "cell": door_cell, "kind": "walk"})

        self.path_index = 0
        self.cell_path: list[dict] = []
        self.route_goal: dict | None = None
        self.task = self.path[0]["task"] if self.path else "wait for work"
        self.greeted_visitors: set[str] = set()
        self.pending_greetings: set[str] = set()

    def _route(self, observation: ThreeBranchesObservation, start: dict, goal: dict) -> list[dict]:
        """Return the shortest legal cardinal route, including start and goal cells."""

        start_key = (int(start["x"]), int(start["y"]))
        goal_key = (int(goal["x"]), int(goal["y"]))
        frontier = deque([start_key])
        previous: dict[tuple[int, int], tuple[int, int] | None] = {start_key: None}
        frame = layout.frame(observation)
        width, height = int(frame["cells_x"]), int(frame["cells_y"])

        while frontier and goal_key not in previous:
            current = frontier.popleft()
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                next_key = current[0] + dx, current[1] + dy
                if (
                    next_key in previous
                    or not 0 <= next_key[0] < width
                    or not 0 <= next_key[1] < height
                ):
                    continue
                current_cell = {"x": current[0], "y": current[1]}
                next_cell = {"x": next_key[0], "y": next_key[1]}
                if layout.can_step(observation, current_cell, next_cell):
                    previous[next_key] = current
                    frontier.append(next_key)

        if goal_key not in previous:
            return []

        route = []
        current: tuple[int, int] | None = goal_key
        while current is not None:
            route.append({"x": current[0], "y": current[1]})
            current = previous[current]
        return list(reversed(route))

    def act(self, observation: ThreeBranchesObservation) -> ThreeBranchesAction:
        """Follow the next legal cell in the current task's route."""

        heading = me.heading(observation)
        here = me.position(observation)
        expression = "wave" if people.seen(observation) else "none"
        audible_ids = {str(person["id"]) for person in people.nearby(observation)}
        for person in people.seen(observation):
            player_id = str(person["id"])
            if (
                people.is_visitor(player_id)
                and player_id in audible_ids
                and player_id not in self.greeted_visitors
            ):
                self.pending_greetings.add(player_id)
        here_cell = layout.cell_at(observation, here)

        if day.phase(observation) == "night":
            self.task = "go home to sleep"
            if self.sleep_cell is None or here_cell is None or here_cell == self.sleep_cell:
                self.task = "sleeping at home"
                return action.stand(heading, "sleep")

            if self.route_goal != self.sleep_cell:
                self.cell_path = self._route(observation, here_cell, self.sleep_cell)
                self.route_goal = self.sleep_cell
            elif self.cell_path and self.cell_path[0] != here_cell:
                if len(self.cell_path) > 1 and self.cell_path[1] == here_cell:
                    self.cell_path.pop(0)
                else:
                    self.cell_path = self._route(observation, here_cell, self.sleep_cell)
            if len(self.cell_path) > 1:
                next_position = layout.cell_center(observation, self.cell_path[1])
                return action.walk(geometry.heading_to(here, next_position), 1.0, expression)

            # If the route home is unexpectedly blocked, remain asleep rather than wandering.
            self.task = "sleeping"
            return action.stand(heading, "sleep")

        if self.path_index < len(self.path) and here_cell is not None:
            stop = self.path[self.path_index]
            self.task = stop["task"]

            if here_cell == stop["cell"]:
                self.path_index += 1
                self.cell_path = []
                self.route_goal = None
                usable = props.usable(observation)
                if stop["kind"] == "rest" and usable is not None and usable["type"] == "bench":
                    return action.stand(heading, "use")
                return action.walk(heading, 0.0, expression)

            # Keep following the cached route; replan only if movement deviates from it.
            if self.route_goal != stop["cell"]:
                self.cell_path = self._route(observation, here_cell, stop["cell"])
                self.route_goal = stop["cell"]
            elif self.cell_path and self.cell_path[0] != here_cell:
                if len(self.cell_path) > 1 and self.cell_path[1] == here_cell:
                    self.cell_path.pop(0)
                else:
                    self.cell_path = self._route(observation, here_cell, stop["cell"])
            elif not self.cell_path:
                self.cell_path = self._route(observation, here_cell, stop["cell"])
            if len(self.cell_path) > 1:
                next_position = layout.cell_center(observation, self.cell_path[1])
                return action.walk(geometry.heading_to(here, next_position), 1.0, expression)

            # An unreachable landmark is skipped rather than trying to cross blocked terrain.
            self.path_index += 1
            self.cell_path = []
            self.route_goal = None
            return action.walk(heading, 0.0, expression)

        if self.path_index >= len(self.path):
            self.task = "finished daily route"
        return action.walk(heading, 0.0, expression)

    # Optional: messaging. On your turn, chat receives messages addressed to your player since
    # its previous turn. Return messages with a recipient and text, or nothing to stay silent.
    # Use None as the recipient to broadcast. Every message is recorded and shown in replays.
    def chat(self, inbox: list[dict]) -> list[dict] | None:
        """Greet each audible visitor once, without replying repeatedly every turn."""

        greetings = [
            {"to": player_id, "text": f"Hello! I am the village {self.role}."}
            for player_id in sorted(self.pending_greetings)
        ]
        self.greeted_visitors.update(self.pending_greetings)
        self.pending_greetings.clear()
        return greetings or None
