import numpy as np
import heapq
import string
from home_layout import generate_house_layout
import math


class A_star_people_placement:
    def __init__(self, filepath):
        """
        Function used to process the grid, specify robot position, people position, and chair position
        Also we are defining the static obstacles here which is nothing but the predefined 
        obstacles that are present on the map and defined in the legend provided
        This is static because we have also other obstacles like other people, but this list changes
        dynamically.
        """
        self.grid = self.read_and_load_grid(filepath)
        self.row, self.column = self.grid.shape
        self.robot_pos = self.get_robot_pos()
        self.people_pos = self.get_people_pos()
        self.chair_pos = self.get_chair_pos()
        self.static_obstacles = {'W', 'A', 'F', 'k', 'G', 'T', 'S', 'c', 'b'}

    def read_and_load_grid(self, filepath):
        """
        Function to load the grid from text file as a numpy array
        Grid would be a 2D numpy array
        """
        with open(filepath, 'r') as r:
            lines = r.read().splitlines()
        return np.array([line.split() for line in lines])

    def get_robot_pos(self):
        """
        Just running through the grid and performing the character matching to see if we found the robot
        Once robot is found we return the tuple containing position of robot
        """
        for i in range(self.row - 1):
            for j in range(self.column - 1):
                if self.grid[i][j] == 'r' and self.grid[i][j + 1] == 'r' and \
                   self.grid[i + 1][j] == 'r' and self.grid[i + 1][j + 1] == 'r':
                    return (i, j)

    def get_people_pos(self):
        """
        Similar to above function run through grid match for digits and return the people co-ordinates
        """
        people = {}
        for i in range(self.row):
            for j in range(self.column):
                if self.grid[i][j] in string.digits:
                    people[self.grid[i][j]] = (i, j)
        return people

    def get_chair_pos(self):
        """
        Simialr to above function just run through the grid and return the 
        co-ordinates of chairs
        Here we are storing the co-ordinates of the chair into a dictionary based on where it is in the grid
        top-left is H1, top right is H2 and so on.... in clockwise direction
        """
        chair_positions = []
        chairs = {}
        for i in range(self.row):
            for j in range(self.column):
                if self.grid[i][j] == 'H':
                    chair_positions.append((i, j))

        min_row = min(pos[0] for pos in chair_positions)
        max_row = max(pos[0] for pos in chair_positions)
        min_col = min(pos[1] for pos in chair_positions)
        max_col = max(pos[1] for pos in chair_positions)

        top = sorted([pos for pos in chair_positions if pos[0] == min_row], key=lambda x: x[1])
        bottom = sorted([pos for pos in chair_positions if pos[0] == max_row], key=lambda x: x[1], reverse=True)
        left = sorted([pos for pos in chair_positions if pos[1] == min_col and pos not in top + bottom], key=lambda x: x[0], reverse=True)
        right = sorted([pos for pos in chair_positions if pos[1] == max_col and pos not in top + bottom], key=lambda x: x[0])

        final_order = left + bottom + right + top
        chair_labels = [f'H{i+1}' for i in range(8)]

        for i in range(len(chair_labels)):
            chairs[chair_labels[i]] = final_order[i]

        print(f"Chairs dict: {chairs}")
        return chairs

    def check_if_position_valid(self, row, column, blocked):
        """
        Check at any given location if robot which is 2x2 can move into it.
        """
        if row < 0 or column < 0 or row + 1 >= self.row or column + 1 >= self.column:
            return False
        for rw in range(2):
            for cl in range(2):
                sub_grid = self.grid[row + rw][column + cl]
                if sub_grid in blocked or (row + rw, column + cl) in blocked:
                    return False
        return True

    def get_next_cell(self, pos, blocked):
        """
        Considering 2x2 size of robot return valid A* cell assignment
        """
        row, column = pos
        directions = [(1, 0), (-1, 0), (0, -1), (0, 1)]
        neighbour = []
        for rw, cl in directions:
            new_rw = row + rw
            new_cl = column + cl
            if self.check_if_position_valid(new_rw, new_cl, blocked):
                neighbour.append((new_rw, new_cl))
        return neighbour

    def heuristic(self, a, b):
        """
        Using manhattan distance here as a heuristic so limiting diagonal movements
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def apply_astar(self, start, goal, blocked):
        """
        Applying A* algo here using f-score and g-scores

        """
        open_set = []
        heapq.heappush(open_set, (0 + self.heuristic(start, goal), 0, start))
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, cost, current = heapq.heappop(open_set)
            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_next_cell(current, blocked):
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, tentative_g, neighbor))

        return None

    def reconstruct_path(self, came_from, current):
        """
        Backtracking through A* path to get the full path
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def get_adjacent_valid_positions(self, chair, blocked):
        """
        Here we are checking for adjacent cells of chair because H is a 1x1 while
        robot is a 2x2, so stopping next to the chair and dropping of the persons
        """
        r, c = chair
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        positions = []
        for dr, dc in directions:
            top_left = (r + dr, c + dc)
            if self.check_if_position_valid(top_left[0], top_left[1], blocked):
                positions.append(top_left)
        return positions

    def find_best_assignments(self):
        """
        Standalone function uses greedy search to assign the chairs to people
        Here using A* to find closest person to robot at start, then to closest goal
        And this process repeats until we have the costs of each person to each goal and
        then we choose the least cost ones at the end.
        """
        robot = self.robot_pos
        assignments = {}
        temp_people = self.people_pos.copy()
        temp_chairs = self.chair_pos.copy()

        while temp_people:
            best_person = None
            best_chair = None
            best_path_len = float('inf')

            for pid, p_pos in temp_people.items():
                blocked = self.static_obstacles.union({p for p in string.digits if p != pid})
                blocked_positions = {loc for pid2, loc in temp_people.items() if pid2 != pid}
                path_to_person = self.apply_astar(robot, p_pos, blocked.union(blocked_positions))

                if not path_to_person:
                    continue

                for label, chair in temp_chairs.items():
                    blocked = self.static_obstacles.union(set(temp_people.values())) - {'H'}
                    adjacent_positions = self.get_adjacent_valid_positions(chair, blocked)
                    for adj_pos in adjacent_positions:
                        path_to_chair = self.apply_astar(p_pos, adj_pos, blocked)
                        if path_to_chair:
                            total_len = len(path_to_person) + len(path_to_chair)
                            if total_len < best_path_len:
                                best_path_len = total_len
                                best_person = pid
                                best_chair = label

            if not best_person or not best_chair:
                print("No more valid assignments can be made.")
                break

            assignments[best_person] = best_chair
            del temp_people[best_person]
            del temp_chairs[best_chair]
            robot = self.people_pos[best_person]

        return assignments

    def seat_people_a_star(self, assignment_dict):
        """
        Helper function to apply a-star to seat people finally once assignments are done
        This function returns cost to deliver each person to the seat and finally the total cost
        """
        robot = self.robot_pos
        temp_people = self.people_pos.copy()
        total_cost = 0

        for pid, chair_label in assignment_dict.items():
            if pid not in temp_people:
                print(f"Skipping person {pid}: already seated or missing.")
                continue

            person_pos = temp_people[pid]
            chair_pos = self.chair_pos[chair_label]

            other_people = {v for k, v in temp_people.items() if k != pid}
            blocked = self.static_obstacles.union(other_people)

            path_to_person = self.apply_astar(robot, person_pos, blocked)
            if not path_to_person:
                print(f"Failed to reach person {pid} at {person_pos}")
                continue

            robot = path_to_person[-1]
            total_cost += len(path_to_person)
            print(f"For person {pid}, path to person length: {len(path_to_person)}")

            min_len = float('inf')
            best_chair_path = None
            blocked_chair = self.static_obstacles.union(set(temp_people.values()))

            for adj in self.get_adjacent_valid_positions(chair_pos, blocked_chair):
                path = self.apply_astar(robot, adj, blocked_chair)
                if path and len(path) < min_len:
                    best_chair_path = path
                    min_len = len(path)

            if not best_chair_path:
                print(f"Failed to deliver person {pid} to chair {chair_label}")
                continue

            robot = best_chair_path[-1]
            total_cost += len(best_chair_path)
            print(f"Delivered person {pid} to {chair_label} with total cost: {len(path_to_person) + len(best_chair_path)}")

            # Update the cell so that cell is traversable
            chair_r, chair_c = self.chair_pos[chair_label]
            self.grid[chair_r][chair_c] = '0'

            del temp_people[pid]

        print(f"\nTotal operation cost: {total_cost}")
        return total_cost


if __name__ == '__main__':
    grid, rooms = generate_house_layout(file_name='home.txt')
    planner = A_star_people_placement('home.txt')
    assignments = planner.find_best_assignments()
    total_cost = planner.seat_people_a_star(assignments)
    print("Total cost:", total_cost)
