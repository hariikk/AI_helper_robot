import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import copy

from collections import defaultdict, Counter
from functools import reduce
from operator import eq, neg


from home_layout import generate_house_layout


grid, rooms = generate_house_layout(
    file_name='home.txt')

filepath = 'home.txt'


# print(rooms.items())
# Slice the rooms


rooms_dict = {}

for room_type, (row_slice, col_slice) in rooms.items():
    slices = []

    for y in range(row_slice.start, row_slice.stop):
        for x in range(col_slice.start, col_slice.stop):

            slices.append((y, x))
    # Assign the slices of matrix to each key room_type as dictionary in rooms_slice
    rooms_dict[room_type] = slices

# TODO: One possible options to implement CSP for social robotics

# Uses MRV to choose the variable.

# Assigns a value.

# Uses inference forward checking to prune domains.

# Do Backtrack search.

# Repeat


#  CLass of the main CSP solver

# table arrangement

#    s8 s7
#  s1     s6
#  s2     s5
#    s3 s4


class SeatingCSP():
    def __init__(self, filepath):
        # Variables of the problem are people.
        self.people = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']

        # All people need to be assigned each seat. So the domain will be the number of seats.
        # We will assume the seats are numbered as s1 .... s8 starting from first left seat in counter clockwise.
        self.seats = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']

        # Domain will be a dict where keys will be person and values will be all 8 seats initially.
        self.domains = {person: self.seats.copy() for person in self.people}

        # fix P1 to seat s1 to iniiate the assignment.
        self.domains['P1'] = ['s1']

        # Constraints from the porblem can be defined as either 2 persons will sit adjacent or far apart.
        # Making a binary constraint for this purpose.
        self.constraints = {
            'P1': [('P6', self.should_be_adjacent)],
            'P3': [('P5', self.should_be_far)],
            'P2': [('P8', self.should_be_adjacent)],
            'P4': [('P7', self.should_be_adjacent)],
            'P6': [('P1', self.should_be_adjacent)],
            'P8': [('P2', self.should_be_adjacent)],
            'P5': [('P3', self.should_be_far)],
            'P7': [('P4', self.should_be_adjacent)],
        }

        self.grid = self.read_layout(filepath)
        self.row, self.column = self.grid.shape

    # Read the Layout from the txt file

    def read_layout(self, file_path='home.txt'):
        file_path = file_path
        home = []
        with open(file_path, 'r') as file:

            for line in file:
                row = line.strip().split(' ')
                home.append(row)
        home = np.array(home)
        return home

    def should_be_adjacent(self, seat1, seat2):
        idx1 = self.seats.index(seat1)
        idx2 = self.seats.index(seat2)
        distance = abs(idx1 - idx2)
        return distance == 1 or distance == 7

    def should_be_far(self, seat1, seat2):
        idx1 = self.seats.index(seat1)
        idx2 = self.seats.index(seat2)
        distance = min(abs(idx1 - idx2), 8 - abs(idx1 - idx2))
        return distance >= 3

    def is_consistent(self, person, seat, assignment):

        if seat in assignment.values():
            return False

        # Check binary constraints with already assigned people
        if person in self.constraints:
            for other_person, constraint in self.constraints[person]:
                if other_person in assignment:
                    other_seat = assignment[other_person]
                    if not constraint(seat, other_seat):
                        return False

        return True  # No conflicts

    # Minimum Remaining Values (MRV) chooses the next unassigned variable

    def select_unassigned_variable(self, assignment):
        # Maintian un assigned list
        unassigned = []

        for p in self.people:
            if p not in assignment:
                unassigned.append(p)

        # Choose the person with the fewest seat options
        select_person = None
        min_domain_size = float('inf')

        for person in unassigned:
            domain_size = len(self.domains[person])
            if domain_size < min_domain_size:
                min_domain_size = domain_size
                select_person = person

        return select_person

    # Implement forward checking to prune the tree
    def forward_check(self, person, seat, assignment, domains):
        new_domains = copy.deepcopy(domains)

        if person in self.constraints:
            for neighbor, constraint in self.constraints[person]:
                if neighbor not in assignment:
                    valid_seats = []
                    for neighbor_seat in new_domains[neighbor]:
                        if constraint(seat, neighbor_seat):
                            valid_seats.append(neighbor_seat)
                    if not valid_seats:
                        return None
                    new_domains[neighbor] = valid_seats
        return new_domains

    def backtrack(self, assignment, domains) -> list:
        # initializing the assingment dict
        # assignment = {}
        # Check for goal state
        if len(assignment) == len(self.people):
            return [assignment.copy()]
        # Choose variable (person) using MRV
        person = self.select_unassigned_variable(assignment)

        # Empty list to store all possible solutions later.
        all_solutions = []
        available_seats = list(self.domains[person])

        # For each seat in their domain:
        for seat in available_seats:

            # Check consistency
            if self.is_consistent(person, seat, assignment):

                # Assign a seat
                assignment[person] = seat
                # We could do Forward checking here but skipping at the moment
                # TODO: implement forward checking to prune the tree.
                # Forward checking
                new_domains = self.forward_check(
                    person, seat, assignment, domains)

                #  Recursively do the backtracking until it exits the loop.
                # solutions = self.backtrack(assignment)
                # all_solutions.extend(solutions)

                if new_domains is not None:
                    #  Recursively do the backtracking until it exits the loop.
                    solutions = self.backtrack(assignment, new_domains)
                    # If successful then append the result into the list of dict
                    all_solutions.extend(solutions)

                # Else undo the assingment if needed. It's the ultimate purpose of backtracking search
                del assignment[person]

        # If successful then return the result
        return all_solutions

    def solve(self):
        solutions = self.backtrack({}, copy.deepcopy(self.domains))
        return solutions

    def get_chair_pos(self):
        # We store the position of chairs as a list.
        chairs = {}
        count = 1
        for i in range(self.row):
            for j in range(self.column):
                if self.grid[i][j] == 'H':
                    label = f'H{count}'
                    chairs[label] = (i, j)
                    count += 1
        print(f"Found {len(chairs)} chairs:", chairs)
        return chairs

    def select_solution(self):
        solutions = self.solve()

        # choosing the 2nd solution which is aligning with all the constraints
        solution = solutions[1]
        print(f"Solution No 2: {solution}")

        chairs = self.get_chair_pos()
        # print(f"Chair positions : {chairs}")

        # Map the people with position of chairs
        map_person_chair_pos = {}
        map_person_chair = {}

        # Mapping seat labels 's1' to 's8' to corresponding 'H1' to 'H8'
        seat_to_chair = {f's{i+1}': pos for i,
                         pos in enumerate(chairs.keys())}

        # mapping the position of the chairs as tuples.
        for person, seat in solution.items():
            map_person_chair_pos[person] = seat_to_chair[seat]

        # mapping the position of the chairs as H1...H8
        for person, seat in solution.items():
            map_person_chair[person.lstrip('P')] = seat_to_chair[seat]
        print(f"map person to chair position: {map_person_chair_pos}")
        print(f"map person to chair position: {map_person_chair}")

        return map_person_chair

    def print(self):
        # print(f"DOMAINS : {self.domains}")
        print("All possible solutions")


############################################
# Usage
if __name__ == "__main__":
    filepath = 'home.txt'

    csp = SeatingCSP('home.txt')
    solutions = csp.solve()

    print(f"Total solutions found: {len(solutions)}\n")
    for i, sol in enumerate(solutions, 1):
        print(f"Solution No {i}: {sol}")

    solution = csp.select_solution()


'''

Language Groups:

P1, P2, P3 speak English only
P4, P5 speak Spanish only
P6, P7, P8 are bilingual
At least one bilingual person must sit between language groups

Conversation Preferences:

P1 wants to discuss business with P6 and needs to sit together
P3 and P5 are in a heated argument so must be far apart at the table
P2 and P8 are close friends so want to sit together
P4 and P7 are sharing a presentation so need to sit adjacent

'''
# Solution [0] = P1, P2, P8, P3, P4, P7, P5, P6
# Solution [1] = P1, P2, P8, P3, P7, P4, P5, P6
# Solution No 2: {'P1': 's1', 'P2': 's2', 'P3': 's4', 'P4': 's6', 'P5': 's7', 'P6': 's8', 'P7': 's5', 'P8': 's3'} -- we can select this.
