from matplotlib.animation import FuncAnimation

def setup_house_visualization(grid, rooms):
    """
    Setup the initial house visualization with all static elements.
    
    Args:
        grid: House layout grid
        rooms: Dictionary of room areas
    
    Returns:
        fig, ax, numeric_grid: Figure, Axes and numeric grid for animation
    """
    # Setup visualization parameters
    char_to_int = {
        '.': 0,    # Empty space
        'W': 1,    # Wall
        'L': 2,    # Living Room
        'D': 3,    # Dining Room
        'K': 4,    # Kitchen
        'B': 5,    # Bedroom
        'C': 6,    # Balcony
        'd': 7,    # Door
        'T': 8,    # Dining Table
        'H': 9,    # Dining Chair
        'P': 10,   # Person
        'S': 11,   # Sofa
        'c': 12,   # Coffee Table
        'b': 13,   # Bed
        'A': 14,   # Wardrobe
        'k': 15,   # Kitchen Counter
        'F': 16,   # Fridge
        'G': 17,   # Garden Chair
        'R': 18,   # Robot
        'X': 19    # Path Trace
    }

    colors = [
        'white',       # Empty
        'black',       # Wall
        'lightblue',   # Living
        'honeydew',    # Dining
        'lightpink',   # Kitchen
        'lightyellow', # Bedroom
        'khaki',       # Balcony
        'burlywood',   # Door
        'orange',      # Table
        'lime',        # Chair
        'blue',        # Person
        'cyan',        # Sofa
        'grey',        # Coffee Table
        'purple',      # Bed
        'deeppink',    # Wardrobe
        'plum',        # Counter
        'dodgerblue',  # Fridge
        'springgreen', # Garden
        'red',         # Robot
        'yellow'       # Path Trace
    ]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Convert to numeric grid
    numeric_grid = np.zeros_like(grid, dtype=int)
    for char, num in char_to_int.items():
        numeric_grid[grid == char] = num
        
    # Convert robot 'r' to empty space for animation
    numeric_grid[grid == 'r'] = char_to_int['.']
    
    # Add room labels
    room_names = {
        'L': "Living\nRoom",
        'D': "Dining\nRoom",
        'K': "Kitchen",
        'B': "Bedroom",
        'C': "Balcony"
    }

    def update(frame_num, path=None, person_following=False):
        """
        Update function for animation.
        
        Args:
            frame_num: Current frame number
            path: List of (x,y) coordinates for robot's path
            person_following: Whether a person is following the robot
        """
        # Clear previous frame
        ax.clear()
        
        # Create current frame's grid
        current_grid = numeric_grid.copy()
        
        if path and frame_num < len(path):
            # Mark path trace
            for i in range(frame_num):
                prev_pos = path[i]
                current_grid[prev_pos] = char_to_int['X']
            
            # Mark current robot position
            current_pos = path[frame_num]
            current_grid[current_pos] = char_to_int['R']
            
            # If person is following, mark their position
            if person_following and frame_num > len(path)//2:
                if frame_num > 0:
                    prev_pos = path[frame_num-1]
                    current_grid[prev_pos] = char_to_int['P']
        
        # Plot the grid
        ax.imshow(current_grid, cmap=ListedColormap(colors))
        
        # Add room labels
        for room_type, (rows, cols) in rooms.items():
            center_x = (cols.start + cols.stop) // 2
            center_y = (rows.start + rows.stop) // 2
            ax.text(center_x, center_y, room_names[room_type],
                    color='black', ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
        # Add legend
        legend_elements = [
            Patch(facecolor='red', label='Robot'),
            Patch(facecolor='blue', label='Person'),
            Patch(facecolor='yellow', label='Path Trace')
        ]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        ax.set_title(f'Robot Navigation - Step {frame_num}')
        ax.grid(True)
    
    return fig, ax, update, numeric_grid

def animate_robot_movement(grid, rooms, path, person_following=False, speed=200):
    """
    Create animation of robot movement.
    
    Args:
        grid: House layout grid
        rooms: Dictionary of room areas
        path: List of (x,y) coordinates for robot's path
        person_following: Whether a person is following the robot
        speed: Animation speed in milliseconds
    
    Returns:
        animation object
    """
    
    # Setup visualization
    fig, ax, update_func, numeric_grid = setup_house_visualization(grid, rooms)
    
    # Create animation
    anim = FuncAnimation(
        fig,
        lambda frame_num: update_func(frame_num, path, person_following),
        frames=len(path),
        interval=speed,
        repeat=False
    )
    
    plt.close(fig)  # Prevent display of static figure
    return anim


