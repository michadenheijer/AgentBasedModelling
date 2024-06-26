import mesa.batchrunner
import numpy as np
import pandas as pd

import mesa
import matplotlib.pyplot as plt

LAMBDA_EXCERSISES = 6


class EmployeeAgent(mesa.Agent):
    """Represents employees in the Gym"""

    def __init__(self, unique_id: int, model: mesa.Model) -> None:
        super().__init__(unique_id, model)

    def step(self) -> None:
        # Move randomly within the gym to an empty cell
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        empty_steps = [
            step for step in possible_steps if self.model.grid.is_cell_empty(step)
        ]

        if empty_steps:
            new_position = self.random.choice(empty_steps)
            self.model.grid.move_agent(self, new_position)

        # Check for benches and reset weights
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        for mate in cellmates:
            if isinstance(mate, GymAttendeeAgent):
                mate.reset_weights()
                self.model.incorrectly_placed_weights -= (
                    1  # Decrease incorrectly placed weights
                )


class GymAttendeeAgent(mesa.Agent):
    """Represents gym goers"""

    def __init__(self, unique_id: int, model: mesa.Model) -> None:
        super().__init__(unique_id, model)
        self.equipment_checklist = self.generate_checklist()
        self.base_incorrect_placement_chance = (
            self.model.base_incorrect_placement_chance
        )
        self.environment_effect = self.model.environment_effect

    def reset_probability(self):
        """Computes the probability that a user want to place it back in the correct spot"""

        ## Should be dependent on employee distance
        # and current placing of weights
        # FIXME not yet dependend on the employees
        return self.base_incorrect_placement_chance + self.environment_effect * (
            self.model.incorrectly_placed_weights / self.model.weights
        )

    def generate_checklist(self):
        # Create a checklist of equipment to visit
        # FIXME this we want to be kind of random
        n_exercises = np.random.poisson(LAMBDA_EXCERSISES)
        checklist = []
        for i in range(n_exercises):
            checklist.append(f"weights_{i}")
        return checklist

    def step(self) -> None:
        # Move randomly within the gym to an empty cell
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        empty_steps = [
            step for step in possible_steps if self.model.grid.is_cell_empty(step)
        ]

        if empty_steps:
            new_position = self.random.choice(empty_steps)
            self.model.grid.move_agent(self, new_position)

        # Use equipment and possibly not reset it
        if self.equipment_checklist:
            current_equipment = self.equipment_checklist.pop()
            prob_needed_weight_incorrectly_placed = (
                self.model.incorrectly_placed_weights / self.model.weights
            )

            # Node takes weight, (thus incorrectly placing it)
            # Node may place it back with reset_chance
            if self.random.random() > prob_needed_weight_incorrectly_placed:
                self.model.incorrectly_placed_weights += 1

            if self.random.random() > self.reset_probability():
                self.model.incorrectly_placed_weights -= 1

        else:
            self.leave()

    def leave(self):
        self.model.attendees.remove(self)
        self.model.grid.remove_agent(self)
        self.model.schedule.remove(self)

    def reset_weights(self):
        # Decrease the incorrectly placed weights count
        if self.model.incorrectly_placed_weights > 0:
            self.model.incorrectly_placed_weights -= 1


def compute_incorrectly_placed_weights(model):
    return model.incorrectly_placed_weights


def compute_probability_of_correctly_placing_weights(model):
    total_checks = model.num_attendees * model.checklist_length
    if total_checks == 0:
        return 1  # No equipment used, so "perfect" placement
    return 1 - model.incorrectly_placed_weights / total_checks


def compute_employee_coverage(model):
    # Calculate the average distance from each attendee to the nearest employee
    distances = []
    for agent in model.schedule.agents:
        if isinstance(agent, GymAttendeeAgent):
            min_distance = min(
                [
                    manhattan_distance(agent.pos, emp.pos)
                    for emp in model.schedule.agents
                    if isinstance(emp, EmployeeAgent)
                ]
            )
            distances.append(min_distance)
    return sum(distances) / len(distances) if distances else 0


def manhattan_distance(pos1, pos2):
    """Calculate the Manhattan distance between two points."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def compute_gym_entropy(model):
    """Calculate the gym entropy based on incorrectly placed weights."""
    # total_checks = model.num_attendees * model.checklist_length
    # if total_checks == 0:
    #     return 0  # No equipment used, so no entropy
    return model.incorrectly_placed_weights / model.weights


def compute_number_of_agents(model):
    return len(model.attendees)


class GymModel(mesa.Model):
    """A representation of the Gym"""

    def __init__(
        self,
        num_employees: int,
        num_attendees: int,
        gym_width: float,
        gym_depth: float,
        base_incorrect_placement_chance: float,
        environment_effect: float,
        attendee_lambda: float,
        weights: int,
        benches: int,
    ) -> None:
        super().__init__()
        self.num_employees = num_employees
        self.num_attendees = num_attendees
        self.grid = mesa.space.SingleGrid(gym_width, gym_depth, False)
        self.base_incorrect_placement_chance = base_incorrect_placement_chance
        self.environment_effect = environment_effect
        self.checklist_length = len(GymAttendeeAgent(0, self).generate_checklist())
        self.attendees = []
        self.attendee_lambda = attendee_lambda
        self.weights = weights
        self.benches = benches
        # self.employees = []

        self.time = 0

        self.schedule = mesa.time.RandomActivation(self)
        self.incorrectly_placed_weights = 0

        for i in range(self.num_employees):
            a = EmployeeAgent(i, self)
            self.place_agent_in_empty_cell(a)

        for j in range(self.num_attendees):
            a = GymAttendeeAgent(j + self.num_employees, self)
            self.attendees.append(a)
            self.place_agent_in_empty_cell(a)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "IncWeightPlacements": compute_incorrectly_placed_weights,
                "EmpCoverage": compute_employee_coverage,
                "EmpericalWeightPlacementProbability": compute_probability_of_correctly_placing_weights,
                "GymEntropy": compute_gym_entropy,
                "NrAgents": compute_number_of_agents,
            }
        )

    def place_agent_in_empty_cell(self, agent):
        while True:
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            if self.grid.is_cell_empty((x, y)):
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)
                break

    def step(self):
        """Advance the model by one step"""
        self.time += 1

        new_attendees = np.random.poisson(self.attendee_lambda)
        for _ in range(new_attendees):
            self.new_gym_attendee()

        self.schedule.step()
        self.datacollector.collect(self)

    def new_gym_attendee(self):
        a = GymAttendeeAgent(self.attendees[-1].unique_id + 1, self)
        self.attendees.append(a)
        self.place_agent_in_empty_cell(a)


# Function to plot gym entropy over time
def plot_gym_entropy(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data["GymEntropy"], label="Gym Entropy")
    plt.xlabel("Step")
    plt.ylabel("Gym Entropy")
    plt.title("Gym Entropy Over Time")
    plt.legend()
    plt.show()


def plot_gym_entropy_batch(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data["Step"], data["GymEntropy"])
    plt.xlabel("Step")
    plt.ylabel("Gym Entropy")
    plt.title("Gym Entropy Over Time")
    plt.legend()
    plt.show()


# Example of initializing and running the model for 1000 steps
# (self, num_employees, num_attendees, gym_width, gym_depth, reset_chance)
starter_model = GymModel(
    2,
    10,
    20,
    20,
    base_incorrect_placement_chance=0.2,
    environment_effect=0.5,
    attendee_lambda=4,
    weights=50,
    benches=10,
)
# for i in range(1000):
#     starter_model.step()

results = mesa.batch_run(
    GymModel,
    {
        "num_employees": [2],
        "num_attendees": [10],
        "gym_width": [20],
        "gym_depth": [20],
        "base_incorrect_placement_chance": [0.05, 0.1, 0.2, 0.3, 0.4],
        "environment_effect": [0.1, 0.2, 0.3, 0.4, 0.5],
        "attendee_lambda": [4],
        "weights": [50],
        "benches": [10],
    },
    1,
    iterations=10,
    data_collection_period=1,
    max_steps=1000,
)

results_df = pd.DataFrame(results)
gym_entropy_results = (
    results_df.groupby(["base_incorrect_placement_chance", "environment_effect"])
    .mean()["GymEntropy"]
    .unstack()
    .transpose()
)
gym_entropy_results.plot()
plt.show()
print(gym_entropy_results)
# plot_gym_entropy(results_df.groupby("Step").mean())
# plot_gym_entropy_batch(results_df)

# # Accessing collected data
# data = starter_model.datacollector.get_model_vars_dataframe()
# print(data)

# # Plotting gym entropy over time
# plot_gym_entropy(data)
