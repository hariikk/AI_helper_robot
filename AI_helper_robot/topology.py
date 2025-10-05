import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class HouseTopology:
    def __init__(self, layout_file, grid, rooms):
        self.layout_file = layout_file
        self.grid = grid
        self.rooms = rooms
        self.rooms_dict = self.slice_rooms()
        self.graph = nx.Graph()
        self.robot_room = None

    def read_layout(self):
        home = []
        with open(self.layout_file, 'r') as file:
            for line in file:
                row = line.strip().split(' ')
                home.append(row)
        self.grid = np.array(home)
        return self.grid

    def slice_rooms(self):
        rooms_dict = {}
        for room_type, (row_slice, col_slice) in self.rooms.items():
            slices = []
            for y in range(row_slice.start, row_slice.stop):
                for x in range(col_slice.start, col_slice.stop):
                    slices.append((y, x))
            rooms_dict[room_type] = slices
        return rooms_dict

    def get_room_of_position(self, position):
        for room, positions in self.rooms_dict.items():
            if position in positions:
                return room
        return None

    def topology_graph(self):
        graph_edges = set()
        robot_room = None

        directions = [((0, -2), (0, 2)), ((-2, 0), (2, 0))]  # horizontal and vertical opposites

        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                if self.grid[x][y] == 'd':
                    for (dx1, dy1), (dx2, dy2) in directions:
                        p1 = (x + dx1, y + dy1)
                        p2 = (x + dx2, y + dy2)

                        room1 = self.get_room_of_position(p1)
                        room2 = self.get_room_of_position(p2)

                        if room1 and room2 and room1 != room2:
                            graph_edges.add(tuple(sorted((room1, room2))))

                elif self.grid[x][y] == 'r':
                    for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                        mx, my = x + dx, y + dy
                        room = self.get_room_of_position((mx, my))
                        if room:
                            robot_room = room

        self.robot_room = robot_room
        self.graph.add_node("r")
        self.graph.add_nodes_from(self.rooms_dict.keys())
        self.graph.add_edges_from(graph_edges)
        if robot_room:
            self.graph.add_edge("r", robot_room)


    def plot_graph(self):
        plt.figure(figsize=(8, 6))
        nx.draw(self.graph, with_labels=True, node_size=2000,
                node_color="lightblue", font_size=10)
        plt.title("Topological Map of House Rooms")
        plt.show()

        
# from home_layout import generate_house_layout

from utils.home_layout import generate_house_layout

# grid, rooms = generate_house_layout(file_name = 'home.txt')
grid, rooms = generate_house_layout("home.txt")
house = HouseTopology("home.txt", grid, rooms)
house.topology_graph()
house.plot_graph()
