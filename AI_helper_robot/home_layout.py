import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

def generate_house_layout(file_name = 'grid.txt'):


    def generate_random_layout(grid_size=100, margin=10, file_name = 'grid.txt'):
        """Generate a random layout with exactly 5 rooms filling the entire available space"""
        grid = np.full((grid_size, grid_size), '.', dtype=str)
        
        # Add outer walls
        grid[0, :] = 'W'
        grid[-1, :] = 'W'
        grid[:, 0] = 'W'
        grid[:, -1] = 'W'
        
        # Calculate usable space
        usable_start = margin
        usable_end = grid_size - margin
        total_width = usable_end - usable_start
        total_height = usable_end - usable_start
        
        # Generate random split points within acceptable ranges
        vertical_split = np.random.randint(
            usable_start + total_width//3,
            usable_start + 2*total_width//3
        )
        
        horizontal_split = np.random.randint(
            usable_start + total_height//3,
            usable_start + 2*total_height//3
        )
        
        # Randomly decide balcony width
        balcony_width = np.random.randint(
            total_width//4,
            total_width//3
        )
        
        # Create room areas
        room_areas = {
            'L': (slice(usable_start, horizontal_split), 
                  slice(usable_start, vertical_split)),  # Living Room
            'D': (slice(usable_start, horizontal_split),
                  slice(vertical_split, usable_end - balcony_width)),  # Dining Room
            'K': (slice(horizontal_split, usable_end),
                  slice(vertical_split, usable_end - balcony_width)),  # Kitchen
            'B': (slice(horizontal_split, usable_end),
                  slice(usable_start, vertical_split)),  # Bedroom
            'C': (slice(usable_start, horizontal_split),
                  slice(usable_end - balcony_width, usable_end))  # Balcony
        }
        
        # Add rooms and walls
        rooms = {}
        for room_type, (rows, cols) in room_areas.items():
            rooms[room_type] = (rows, cols)
            grid[rows, cols] = room_type
            
            # Add walls
            grid[rows.start-1:rows.stop+1, cols.start-1] = 'W'  # Left wall
            grid[rows.start-1:rows.stop+1, cols.stop] = 'W'     # Right wall
            grid[rows.start-1, cols.start-1:cols.stop+1] = 'W'  # Top wall
            grid[rows.stop, cols.start-1:cols.stop+1] = 'W'     # Bottom wall
        
        def add_door(room1_slice, room2_slice, vertical=True):
            if vertical:
                common_wall = min(room1_slice[1].stop, room2_slice[1].start)
                start = max(room1_slice[0].start, room2_slice[0].start) + 3
                end = min(room1_slice[0].stop, room2_slice[0].stop) - 3
                if end > start:
                    door_pos = np.random.randint(start, end)
                    grid[door_pos:door_pos+3, common_wall-1:common_wall+2] = 'd'
            else:
                common_wall = min(room1_slice[0].stop, room2_slice[0].start)
                start = max(room1_slice[1].start, room2_slice[1].start) + 3
                end = min(room1_slice[1].stop, room2_slice[1].stop) - 3
                if end > start:
                    door_pos = np.random.randint(start, end)
                    grid[common_wall-1:common_wall+2, door_pos:door_pos+3] = 'd'
        
        # Add doors between rooms
        add_door(rooms['L'], rooms['D'], vertical=True)
        add_door(rooms['L'], rooms['B'], vertical=False)
        add_door(rooms['D'], rooms['K'], vertical=False)
        add_door(rooms['B'], rooms['K'], vertical=True)
        add_door(rooms['D'], rooms['C'], vertical=True)
        
        # Add main entrance
        entrance_y = rooms['L'][0].start + np.random.randint(3, rooms['L'][0].stop - rooms['L'][0].start - 3)
        grid[entrance_y:entrance_y+3, usable_start-1:usable_start+2] = 'd'
        
        return grid, rooms

    def place_random_furniture(grid, rooms):
        """Place furniture randomly in rooms"""
        occupied_positions = set()
        
        def get_valid_position(room_slice, size=(2,2), max_attempts=100):
            rows, cols = room_slice
            attempts = max_attempts
            
            while attempts > 0:
                row_range = rows.stop - rows.start - size[0] - 2
                col_range = cols.stop - cols.start - size[1] - 2
                
                if row_range <= 0 or col_range <= 0:
                    return None
                    
                row = np.random.randint(rows.start+1, rows.start + row_range)
                col = np.random.randint(cols.start+1, cols.start + col_range)
                
                valid = True
                for r in range(row-1, row+size[0]+1):
                    for c in range(col-1, col+size[1]+1):
                        if (r, c) in occupied_positions or grid[r, c] in ['W', 'd']:
                            valid = False
                            break
                
                if valid:
                    for r in range(row, row+size[0]):
                        for c in range(col, col+size[1]):
                            occupied_positions.add((r, c))
                    return row, col
                
                attempts -= 1
            return None

        # Place dining table and chairs
        dining_room = rooms['D']
        table_pos = get_valid_position(dining_room, size=(2, 2))
        
        if table_pos:
            # Place table
            grid[table_pos[0]:table_pos[0]+2, table_pos[1]:table_pos[1]+2] = 'T'
            
            # Define chair positions (1x1 chairs)
            chair_positions = []
            
            # Only add chairs if there's space (not adjacent to walls)
            # Top chairs
            if table_pos[0] > dining_room[0].start + 1:
                chair_positions.extend([
                    (table_pos[0]-1, table_pos[1]),    # Top left
                    (table_pos[0]-1, table_pos[1]+1)   # Top right
                ])
            
            # Bottom chairs
            if table_pos[0] + 2 < dining_room[0].stop - 1:
                chair_positions.extend([
                    (table_pos[0]+2, table_pos[1]),    # Bottom left
                    (table_pos[0]+2, table_pos[1]+1)   # Bottom right
                ])
            
            # Left chairs
            if table_pos[1] > dining_room[1].start + 1:
                chair_positions.extend([
                    (table_pos[0], table_pos[1]-1),    # Left top
                    (table_pos[0]+1, table_pos[1]-1)   # Left bottom
                ])
            
            # Right chairs
            if table_pos[1] + 2 < dining_room[1].stop - 1:
                chair_positions.extend([
                    (table_pos[0], table_pos[1]+2),    # Right top
                    (table_pos[0]+1, table_pos[1]+2)   # Right bottom
                ])
            
            # Place chairs (up to 8)
            chairs_placed = 0
            for chair_pos in chair_positions:
                if grid[chair_pos] not in ['W', 'd'] and chairs_placed < 8:
                    grid[chair_pos] = 'H'
                    occupied_positions.add(chair_pos)
                    chairs_placed += 1

        # Place other furniture
        for room_type, room_slice in rooms.items():
            if room_type == 'L':  # Living Room
                pos = get_valid_position(room_slice, (4, 3))
                if pos:
                    grid[pos[0]:pos[0]+4, pos[1]:pos[1]+3] = 'S'  # Sofa
                    
                pos = get_valid_position(room_slice, (2, 3))
                if pos:
                    grid[pos[0]:pos[0]+2, pos[1]:pos[1]+3] = 'c'  # Coffee table
                    
            elif room_type == 'B':  # Bedroom
                pos = get_valid_position(room_slice, (4, 4))
                if pos:
                    grid[pos[0]:pos[0]+4, pos[1]:pos[1]+4] = 'b'  # Bed
                    
                pos = get_valid_position(room_slice, (4, 3))
                if pos:
                    grid[pos[0]:pos[0]+4, pos[1]:pos[1]+3] = 'A'  # Wardrobe
                    
            elif room_type == 'K':  # Kitchen
                pos = get_valid_position(room_slice, (4, 2))
                if pos:
                    grid[pos[0]:pos[0]+4, pos[1]:pos[1]+2] = 'k'  # Counter
                    
                pos = get_valid_position(room_slice, (3, 3))
                if pos:
                    grid[pos[0]:pos[0]+3, pos[1]:pos[1]+3] = 'F'  # Fridge
                    
            elif room_type == 'C':  # Balcony
                for _ in range(3):
                    pos = get_valid_position(room_slice, (2, 2))
                    if pos:
                        grid[pos[0]:pos[0]+2, pos[1]:pos[1]+2] = 'G'  # Garden chair
        
        return occupied_positions

    # Generate layout and place furniture
    grid, rooms = generate_random_layout(file_name = file_name)
    occupied_positions = place_random_furniture(grid, rooms)

    # Place robot near entrance (2x2)
    entrance_positions = [(i, j) for i in range(1, grid.shape[0]-2) 
                         for j in range(1, grid.shape[1]-2)
                         if grid[i, j] == 'd' and grid[i-1, j] == 'W']
    if entrance_positions:
        robot_pos = entrance_positions[0]
        grid[robot_pos[0]:robot_pos[0]+2, robot_pos[1]:robot_pos[1]+2] = 'r'
        for r in range(robot_pos[0], robot_pos[0]+2):
            for c in range(robot_pos[1], robot_pos[1]+2):
                occupied_positions.add((r, c))

    # Place people randomly (1x1 each)
    valid_positions = []
    for room_slice in rooms.values():
        rows, cols = room_slice
        for i in range(rows.start, rows.stop):
            for j in range(cols.start, cols.stop):
                if (i, j) not in occupied_positions and grid[i, j] not in ['W', 'd']:
                    valid_positions.append((i, j))

    # Place 8 people
    num_people = 8
    people_positions = []
    if len(valid_positions) >= num_people:
        for person_num in range(num_people):
            if valid_positions:
                idx = np.random.randint(0, len(valid_positions))
                pos = valid_positions.pop(idx)
                grid[pos] = str(person_num + 1)
                people_positions.append(pos)
                valid_positions = [p for p in valid_positions 
                                 if abs(p[0] - pos[0]) > 0 or abs(p[1] - pos[1]) > 0]

    # Save grid to file
    np.savetxt(file_name, grid, fmt='%s')

    # Visualization setup
    char_to_int = {
        '.': 0,    # Empty space
        'W': 1,    # Wall
        'L': 2,    # Living Room
        'D': 3,    # Dining Room
        'K': 4,    # Kitchen
        'B': 5,    # Bedroom
        'C': 6,    # Balcony
        'd': 7,    # Door
        'r': 8,    # Robot
        'T': 9,    # Dining Table
        'H': 10,   # Dining Chair
        'S': 12,   # Sofa
        'c': 13,   # Coffee Table
        'b': 14,   # Bed
        'A': 15,   # Wardrobe
        'k': 16,   # Kitchen Counter
        'F': 17,   # Fridge
        'G': 18    # Garden Chair
    }

    # Add people numbers to char_to_int mapping
    for i in range(1, num_people + 1):
        char_to_int[str(i)] = 11  # All people share the same color

    colors = [
        'white',         # Empty space
        'black',         # Wall
        'lightblue',     # Living Room
        'honeydew',    # Dining Room
        'lightpink',     # Kitchen
        'lightyellow',          # Bedroom
        'khaki',         # Balcony
        'burlywood',        # Door
        'red',           # Robot
        'orange',        # Dining Table
        'lime',          # Dining Chair
        'black',       # Person
        'cyan',          # Sofa
        'grey',          # Coffee Table
        'purple',        # Bed
        'deeppink',      # Wardrobe
        'plum', # Kitchen Counter
        'dodgerblue',    # Fridge
        'springgreen'    # Garden Chair
    ]

    # Convert to numeric grid
    numeric_grid = np.zeros_like(grid, dtype=int)
    for char, num in char_to_int.items():
        numeric_grid[grid == char] = num

    # Plot
    plt.figure(figsize=(15, 15))
    cmap = ListedColormap(colors)
    plt.imshow(numeric_grid, cmap=cmap)

    # Add room labels
    room_names = {
        'L': "Living\nRoom",
        'D': "Dining\nRoom",
        'K': "Kitchen",
        'B': "Bedroom",
        'C': "Balcony"
    }

    for room_type, (rows, cols) in rooms.items():
        center_x = (cols.start + cols.stop) // 2
        center_y = (rows.start + rows.stop) // 2
        plt.text(center_x, center_y, room_names[room_type],
                 color='black', ha='center', va='center',
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    # Add borders for dining chairs
    chair_positions = np.where(grid == 'H')
    for y, x in zip(chair_positions[0], chair_positions[1]):
        plt.plot([x-0.5, x+0.5, x+0.5, x-0.5, x-0.5],
                 [y-0.5, y-0.5, y+0.5, y+0.5, y-0.5],
                 'black', linewidth=2)

    # Add person numbers
    for i, pos in enumerate(people_positions):
        plt.text(pos[1], pos[0], f'{i+1}',
                 color='white', ha='center', va='center',
                 fontsize=8, fontweight='bold')

    # Add legend
    legend_elements = [Patch(facecolor=color, label=label) 
                      for color, label in zip(colors, [
        'Corridor', 'Wall', 'Living Room', 'Dining Room', 'Kitchen',
        'Bedroom', 'Balcony', 'Door', 'Robot', 'Dining Table',
        'Dining Chair', 'Person', 'Sofa', 'Coffee Table', 'Bed',
        'Wardrobe', 'Kitchen Counter', 'Fridge', 'Garden Chair'
    ])]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.title('Random House Floor Plan with Robot, Furniture and People')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.grid(True, which='major', color='gray', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.show()

    return grid, rooms
