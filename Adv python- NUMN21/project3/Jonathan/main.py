from solver import DirichletNeumannSolver
from project3.Final.apartment import Room, Apartment
from project3.Final.global_params import T_HEATER, T_WALL, T_WINDOW
import time
start_time = time.time()


# TODO


# Create rooms and apartment

rooms = []

bounds1 = {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}
bc1 = {
            "left": [
                {"y_min": 0.0, "y_max": 1.0, "type": "Heater", "value": T_HEATER}
            ],
            "right": [
                {"y_min": 0.0, "y_max": 1.0, "type": "Interface"}
            ],
            "top": [
                {"x_min": 0.0, "x_max": 1.0, "type": "Wall", "value": T_WALL}
            ],
            "bottom": [
                {"x_min": 0.0, "x_max": 1.0, "type": "Wall", "value": T_WALL}
            ],
        }
rooms.append(Room(1,bounds1, "right", bc1))

bounds2 = {"x_min": 1.0, "x_max": 2.0, "y_min": 0.0, "y_max": 2.0}
bc2 = {
            "left": [
                {"y_min": 0.0, "y_max": 1.0, "type": "Interface"},
                {"y_min": 1.0, "y_max": 2.0, "type": "Wall", "value": T_WALL},
            ],
            "right": [
                {"y_min": 0.0, "y_max": 1.0, "type": "Wall", "value": T_WALL},
                {"y_min": 1.0, "y_max": 2.0, "type": "Interface"},
            ],
            "top": [
                {"x_min": 1.0, "x_max": 2.0, "type": "Heater", "value": T_HEATER}
            ],
            "bottom": [
                {"x_min": 1.0, "x_max": 2.0, "type": "Window", "value": T_WINDOW}
            ],
        }
rooms.append(Room(2,bounds2, None, bc2)) #None bc Dirichlet

bounds3 = {"x_min": 2.0, "x_max": 3.0, "y_min": 1.0, "y_max": 2.0}
bc3 = {
            "left": [
                {"y_min": 1.0, "y_max": 2.0, "type": "Interface"}
            ],
            "right": [
                {"y_min": 1.0, "y_max": 2.0, "type": "Heater", "value": T_HEATER}
            ],
            "top": [
                {"x_min": 2.0, "x_max": 3.0, "type": "Wall", "value": T_WALL}
            ],
            "bottom": [
                {"x_min": 2.0, "x_max": 3.0, "type": "Wall", "value": T_WALL}
            ],
        }
rooms.append(Room(3,bounds3, "left", bc3))

bounds4 = {"x_min": 2.0, "x_max": 2.5, "y_min": 0.5, "y_max": 1.0}
bc4 = {
            "left": [
                {"y_min": 0.5, "y_max": 1.0, "type": "Interface"}
            ],
            "right": [
                {"y_min": 0.5, "y_max": 1.0, "type": "Wall", "value": T_WALL}
            ],
            "top": [
                {"x_min": 2.0, "x_max": 2.5, "type": "Wall", "value": T_WALL}
            ],
            "bottom": [
                {"x_min": 2.0, "x_max": 2.5, "type": "Heater", "value": T_HEATER}
            ],
        }
rooms.append(Room(4,bounds4, "left", bc4))


apartment = Apartment(rooms)

# Create solver
solver = DirichletNeumannSolver(apartment)

# Run iteration loop
solver.run()
print("--- %s seconds ---" % (time.time() - start_time))
U_global = solver.assemble_global_solution()
solver.plot_solution(U_global)

# Access solution per room
solution = solver.subdomain_data