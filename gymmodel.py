import mesa.batchrunner
import numpy as np
import pandas as pd

import mesa
import matplotlib.pyplot as plt

import random

LAMBDA_EXCERSISES = 6
_COUNTER = 0


def counter():
    global _COUNTER
    _COUNTER += 1
    return _COUNTER


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

        # If directly next to GymAttendee, they always clean up their station
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        for mate in cellmates:
            if isinstance(mate, GymAttendeeAgent):
                mate.reset_weights()
                self.model.incorrectly_placed_weights -= (
                    1  # Decrease incorrectly placed weights
                )

        if self.model.early_cleaning:
            gym_entropy = self.model.incorrectly_placed_weights / self.model.weights
            if gym_entropy > 0.3 and gym_entropy < 0.5:
                self.model.incorrectly_placed_weights -= 1
                self.model.cleaned_weights += 1


class GymAttendeeAgent(mesa.Agent):
    """Represents gym goers"""

    def __init__(self, unique_id: int, model: mesa.Model, agent_type=None) -> None:
        super().__init__(unique_id, model)
        self.equipment_checklist = self.generate_checklist()
        self.base_inc_utility = self.model.base_inc_utility
        self.environment_effect = self.model.environment_effect
        self.employee_effect = self.model.employee_effect
        self.current_exercise = None
        self.agent_type = (
            agent_type  # None: Default, True: Good Actor, False: Bad actor
        )

    def closest_employee_distance(self):
        distance = float("inf")

        for emp in self.model.employees:
            distance = min(distance, manhattan_distance(self.pos, emp.pos))

        return distance

    def reset_probability(self):
        """Computes the probability that a user want to place it back in the correct spot"""

        ## Should be dependent on employee distance
        # and current placing of weights
        # FIXME not yet dependend on the employees
        if self.agent_type is not None:
            return int(self.agent_type)

        utility = self.base_inc_utility + self.environment_effect * (
            self.model.incorrectly_placed_weights / self.model.weights
        )

        if self.closest_employee_distance() < 5:
            utility += self.employee_effect

        return 1 / (1 + np.exp(utility))

    def generate_checklist(self):
        # Create a checklist of equipment to visit
        n_exercises = np.random.poisson(LAMBDA_EXCERSISES)
        checklist = []

        exercises = random.choice(
            [
                ("bench",),
                ("deadlift",),
                ("free",),
                ("bench", "deadlift"),
                ("bench", "free"),
                ("deadlift", "free"),
                ("bench", "deadlift", "free"),
            ]
        )
        for i in range(n_exercises):
            checklist.append(f"{random.choice(exercises)}")
        return checklist

    def get_next_exercise(self):
        for ex in self.equipment_checklist:
            if self.model.current_status[ex] > 0:
                self.model.current_status[ex] -= 1
                self.current_exercise = ex
                self.equipment_checklist.remove(ex)
                return ex

        return None

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

        if self.current_exercise:
            self.model.current_status[self.current_exercise] += 1
            self.current_exercise = None

        # Use equipment and possibly not reset it
        if self.equipment_checklist:
            self.current_exercise = self.get_next_exercise()

            if self.current_exercise is None:
                return

            prob_needed_weight_incorrectly_placed = (
                self.model.incorrectly_placed_weights / self.model.weights
            )

            # Node takes weight, (thus incorrectly placing it)
            # Node may place it back with reset_chance
            if self.random.random() > prob_needed_weight_incorrectly_placed:
                self.model.incorrectly_placed_weights += 1

            if self.random.random() < self.reset_probability():
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


def compute_cleaned_weights(model):
    return model.cleaned_weights


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


def compute_devices_available(model):
    return (
        model.current_status["bench"]
        + model.current_status["deadlift"]
        + model.current_status["free"]
    )


class GymModel(mesa.Model):
    """A representation of the Gym"""

    def __init__(
        self,
        num_employees: int,
        num_attendees: int,
        gym_width: float,
        gym_depth: float,
        base_inc_utility: float,
        environment_effect: float,
        employee_effect: float,
        attendee_lambda: float,
        weights: int,
        benches: int,
        deadlifts: int,
        free_weights: int,
        init_incorrect_weights: int = 0,
        heterogeneous_frac: float = 0,
        good_frac: float = 0,
        prev_entropy=None,
        early_cleaning=False,
    ) -> None:
        super().__init__()
        self.num_employees = num_employees
        self.num_attendees = num_attendees
        self.grid = mesa.space.SingleGrid(gym_width, gym_depth, False)
        self.base_inc_utility = base_inc_utility
        self.environment_effect = environment_effect
        self.employee_effect = employee_effect
        self.checklist_length = len(GymAttendeeAgent(0, self).generate_checklist())
        self.attendees = []
        self.attendee_lambda = attendee_lambda
        self.weights = weights
        self.current_status = {
            "bench": benches,
            "deadlift": deadlifts,
            "free": free_weights,
        }
        self.heterogeneous_frac = heterogeneous_frac
        self.good_frac = good_frac
        self.employees = []
        self.early_cleaning = early_cleaning
        self.cleaned_weights = 0

        self.time = 0

        if prev_entropy is not None:
            self.good_frac = (
                1 - prev_entropy[self.base_inc_utility][self.environment_effect]
            )

        self.schedule = mesa.time.RandomActivation(self)
        self.incorrectly_placed_weights = init_incorrect_weights

        for i in range(self.num_employees):
            a = EmployeeAgent(i, self)
            self.employees.append(a)
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
                "DevicesAvailable": compute_devices_available,
                "CleanedWeights": compute_cleaned_weights,
            }
        )

    def place_agent_in_empty_cell(self, agent):
        i = 0
        while True:
            i += 1
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            if self.grid.is_cell_empty((x, y)):
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)
                break

            if i > 10000:
                raise Exception("Gym is full")

    def step(self):
        """Advance the model by one step"""
        self.time += 1

        new_attendees = np.random.poisson(self.attendee_lambda)
        for _ in range(new_attendees):
            self.new_gym_attendee()

        self.schedule.step()
        self.datacollector.collect(self)

    def new_gym_attendee(self):
        a = GymAttendeeAgent(counter(), self)

        if self.heterogeneous_frac > random.random():
            a = GymAttendeeAgent(counter(), self, random.random() < self.good_frac)
        self.attendees.append(a)
        self.place_agent_in_empty_cell(a)


# Function to plot gym entropy over time
def plot_gym_entropy(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data["GymEntropy"], label="Gym Entropy")
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Step")
    plt.ylabel("Gym Entropy")
    plt.title("Gym Entropy Over Time")
    plt.legend()
    plt.show()


def plot_gym_devices_available(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data["DevicesAvailable"], label="Devices Available")
    plt.xlabel("Step")
    plt.ylabel("Devices Available")
    plt.title("Devices Available")
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
# starter_model = GymModel(
#     num_employees=1,
#     num_attendees=10,
#     gym_width=20,
#     gym_depth=20,
#     base_inc_utility=-3,
#     environment_effect=8,
#     employee_effect=-2,
#     attendee_lambda=1,
#     weights=50,
#     benches=3,
#     deadlifts=3,
#     free_weights=3,
# )
# for i in range(10000):
#     print(i)
#     starter_model.step()

# # results = mesa.batch_run(
# #     GymModel,
# #     {
# #         "num_employees": [2],
# #         "num_attendees": [10],
# #         "gym_width": [20],
# #         "gym_depth": [20],
# #         "base_incorrect_placement_chance": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
# #         "environment_effect": [
# #             0.1,
# #             0.2,
# #             0.3,
# #             0.4,
# #             0.5,
# #             0.6,
# #             0.7,
# #             0.8,
# #             0.9,
# #             1,
# #             1.1,
# #             1.2,
# #         ],
# #         "attendee_lambda": [4],
# #         "weights": [50],
# #         "benches": [10],
# #         "init_incorrect_weights": [6],
# #     },
# #     1,
# #     iterations=10,
# #     data_collection_period=1,
# #     max_steps=100,
# # )

# # results_df = pd.DataFrame(results)
# # gym_entropy_results = (
# #     results_df[results_df["Step"] > 15]
# #     .groupby(["base_incorrect_placement_chance", "environment_effect"])
# #     .mean()["GymEntropy"]
# #     .unstack()
# #     .transpose()
# # )
# # # gym_entropy_results.plot(
# # #     title="Changing the effect of the Environment clearly has a large impact",
# # #     ylabel="Average Gym Entropy",
# # # )
# # # plt.show()
# # # print(gym_entropy_results)
# # plot_gym_entropy(results_df.groupby("Step").mean())
# # plot_gym_entropy_batch(results_df)

# # # Accessing collected data
# data = starter_model.datacollector.get_model_vars_dataframe()
# print(data)

# # Plotting gym entropy over time
# plot_gym_entropy(data)
# # plot_gym_devices_available(data)
