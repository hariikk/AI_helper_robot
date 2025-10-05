import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from a_star_planner import A_star_people_placement
from PIL import Image

# Initialize the planner and collect all simulation steps
planner = A_star_people_placement('home.txt')
steps = list(planner.assign_position_to_people_stepwise())

# Setup the plot
dpi = 100
fig_width, fig_height = 12, 10  # inches
fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
ax = fig.add_subplot(111)
frames = []

def draw_grid(grid, ax):
    ax.clear()
    rows, cols = grid.shape
    for i in range(rows):
        for j in range(cols):
            val = grid[i][j]
            color = 'white'

            if val == 'r':
                color = 'red'
            elif val in '123456789':
                color = 'orange'
            elif val.startswith('H'):
                color = 'green'
            elif val in {'W', 'A', 'F', 'k', 'G', 'T', 'H', 'S', 'c', 'b'}:
                color = 'black'
            elif val == '.':
                color = 'white'

            ax.add_patch(plt.Rectangle((j, -i), 1, 1, facecolor=color, edgecolor='gray'))
            ax.text(j + 0.5, -i + 0.5, val, ha='center', va='center', fontsize=6)

    ax.set_xlim(0, cols)
    ax.set_ylim(-rows, 0)
    ax.set_aspect('equal')
    ax.axis('off')

def capture_frame(fig):
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = Image.frombytes('RGB', (width, height), fig.canvas.tostring_rgb())
    return np.array(image)

# Run simulation and record frames
for step in steps:
    event = step[0]

    if event in ('move_to_person', 'move_to_chair'):
        draw_grid(planner.grid.copy(), ax)
        frame = capture_frame(fig)
        frames.append([plt.imshow(frame)])

    elif event == 'delivered':
        print(f"Delivered person {step[1]} to chair {step[2]}")

    elif event == 'done':
        print("Finished all deliveries")
        print("Final Assignments:")
        for person, chair in step[1].items():
            print(f"Person {person} → {chair}")

# Export video
print("Saving animation as robot_delivery.mp4...")
ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True)
ani.save('robot_delivery.mp4', writer='ffmpeg', fps=5)
print("Done: video saved as robot_delivery.mp4")
