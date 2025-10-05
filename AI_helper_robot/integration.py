from seating_csp import SeatingCSP
from collections import deque
from robo_placement_class import A_star_people_placement

'''
Choose valid assignment to move forward from CSP.

Map seat labels to physical locations in the grid,

Find each guest’s starting position and move robot to that position with A*

Use A* to guide each one to their assigned seat.

'''
# {P1 : (x,y), P2: (x,y)...} this is how we return the assingment from CSP as a dict


# Integration of Seating CSP + A* + Topological Map
if __name__ == "__main__":
    filepath = 'home.txt'

    # Use CSP class to get one assignment
    csp = SeatingCSP(filepath)

    # Directly get the solution with mapped tuples of position of chairs to each person. Returns a dict
    map_person_chair = csp.select_solution()

    # Use A star class to do path planning
    a_star = A_star_people_placement(filepath)
    a_star.seat_people_a_star(map_person_chair)


# TODO: Print topological map as below example
