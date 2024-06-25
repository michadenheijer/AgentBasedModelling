import mesa
import matplotlib.pyplot as plt

class EmployeeAgent(mesa.Agent):
    """Represents employees in the Gym"""
    def __init__(self, unique_id: int, model: mesa.Model) -> None:
        super().__init__(unique_id, model)

    def step(self) -> None:
        # Move randomly within the gym to an empty cell
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False
        )
        empty_steps = [step for step in possible_steps if self.model.grid.is_cell_empty(step)]
        
        if empty_steps:
            new_position = self.random.choice(empty_steps)
            self.model.grid.move_agent(self, new_position)
        
        # Check for benches and reset weights
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        for mate in cellmates:
            if isinstance(mate, GymAttendeeAgent):
                mate.reset_weights()
                self.model.incorrectly_placed_weights -= 1  # Decrease incorrectly placed weights


class GymAttendeeAgent(mesa.Agent):
    """Represents gym goers"""
    def __init__(self, unique_id: int, model: mesa.Model) -> None:
        super().__init__(unique_id, model)
        self.equipment_checklist = self.generate_checklist()
        self.reset_chance = self.model.reset_chance

    def generate_checklist(self):
        # Create a checklist of equipment to visit
        return ["bench", "treadmill", "weights"]

    def step(self) -> None:
        # Move randomly within the gym to an empty cell
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False
        )
        empty_steps = [step for step in possible_steps if self.model.grid.is_cell_empty(step)]
        
        if empty_steps:
            new_position = self.random.choice(empty_steps)
            self.model.grid.move_agent(self, new_position)
        
        # Use equipment and possibly not reset it
        if self.equipment_checklist:
            current_equipment = self.equipment_checklist.pop()
            if self.random.random() > self.reset_chance:
                self.model.incorrectly_placed_weights += 1

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
            min_distance = min([manhattan_distance(agent.pos, emp.pos) for emp in model.schedule.agents if isinstance(emp, EmployeeAgent)])
            distances.append(min_distance)
    return sum(distances) / len(distances) if distances else 0

def manhattan_distance(pos1, pos2):
    """Calculate the Manhattan distance between two points."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def compute_gym_entropy(model):
    """Calculate the gym entropy based on incorrectly placed weights."""
    total_checks = model.num_attendees * model.checklist_length
    if total_checks == 0:
        return 0  # No equipment used, so no entropy
    return model.incorrectly_placed_weights / total_checks

class GymModel(mesa.Model):
    """A representation of the Gym"""
    def __init__(self, num_employees, num_attendees, gym_width, gym_depth, reset_chance) -> None:
        super().__init__()
        self.num_employees = num_employees
        self.num_attendees = num_attendees
        self.grid = mesa.space.SingleGrid(gym_width, gym_depth, False)
        self.reset_chance = reset_chance
        self.checklist_length = len(GymAttendeeAgent(0, self).generate_checklist())

        self.schedule = mesa.time.RandomActivation(self)
        self.incorrectly_placed_weights = 0

        for i in range(self.num_employees):
            a = EmployeeAgent(i, self)
            self.place_agent_in_empty_cell(a)

        for j in range(self.num_attendees):
            a = GymAttendeeAgent(j + self.num_employees, self)
            self.place_agent_in_empty_cell(a)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "IncWeightPlacements": compute_incorrectly_placed_weights, 
                "EmpCoverage": compute_employee_coverage, 
                "EmpericalWeightPlacementProbability": compute_probability_of_correctly_placing_weights,
                "GymEntropy": compute_gym_entropy
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
        self.schedule.step()
        self.datacollector.collect(self)

# Function to plot gym entropy over time
def plot_gym_entropy(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data["GymEntropy"], label="Gym Entropy")
    plt.xlabel("Step")
    plt.ylabel("Gym Entropy")
    plt.title("Gym Entropy Over Time")
    plt.legend()
    plt.show()

# Example of initializing and running the model for 1000 steps
#(self, num_employees, num_attendees, gym_width, gym_depth, reset_chance)
starter_model = GymModel(10, 50, 20, 20, reset_chance=0.9)
for i in range(50):
    starter_model.step()

# Accessing collected data
data = starter_model.datacollector.get_model_vars_dataframe()
print(data)

# Plotting gym entropy over time
plot_gym_entropy(data)
