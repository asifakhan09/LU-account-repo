import numpy as np

TOL = 1e-12
T_WALL, T_WINDOW, T_HEATER = 15.0, 5.0, 40.0

class Room:
    """
    Represents a room with geometric bounds and boundary conditions.

    A Room knows its:
      - Size(x/y min and max)
      - Type of interface (Neumann-left, Neumann-right, or Dirichlet)
      - Boundary condition segments along each side.
    """

    def __init__(self, id: str, bounds: dict[str,float], interface: str,  bc: dict[str,list], h): 
        """
        Initialize a room.

        Params:
            id: str
                Identifier for the room.
            bounds: dict[str, float]
                Dictionary with keys 'x_min', 'x_max', 'y_min', 'y_max' defining room geometry.
            interface: str
                Location of interface ("left", "right", or None).
            bc: dict[str, list]
                Dictionary of boundary segments per side ('left', 'right', 'top', 'bottom'),
                where each segment contains keys:
                - 'x_min' or 'y_min'
                - 'x_max' or 'y_max'
                - 'type' in {"Wall", "Window", "Heater", "Interface"}
                - 'value' (for fixed BCs, optional)
            h: float
                Grid spacing for discretization.
        """
        self.id = id
        self.bounds = bounds
        self.interface = interface
        self.bc = bc
        self.h = h


    def get_geometry(self):
        """
        Compute geometric proprerties of the room.

        Returns:
            nx_s,ny_s: int
              The number of grid points in x and y
            i_off, j_off: int
                Global index offsets for mapping local coordinates to global
        """
        Lx_s = self.bounds['x_max'] - self.bounds['x_min']
        Ly_s = self.bounds['y_max'] - self.bounds['y_min']
        nx_s = int(round(Lx_s / self.h)) + 1
        ny_s = int(round(Ly_s / self.h)) + 1
        i_off = int(round(self.bounds['x_min'] / self.h))
        j_off = int(round(self.bounds['y_min'] / self.h))
        return nx_s, ny_s, i_off, j_off

    
    def get_type(self):
        """ Get the type of the room based on its interface."""

        if self.interface == "right":
            return "Neumann_right"
        elif self.interface == "left":
            return "Neumann_left"
        else:
            return "Dirichlet"
        

    def contains(self, x: float, y: float):
        """
        Check if a global (x, y) point lies inside the room bounds.

        Params:
            x: float
                x-coordinate.
            y: float
                y-coordinate.

        Returns:
            bool
                True if point is inside the room (with tolerance), False otherwise.
        """
        return (self.bounds['x_min'] - TOL <= x <= self.bounds['x_max'] + TOL and
                self.bounds['y_min'] - TOL <= y <= self.bounds['y_max'] + TOL)
    
    
    def get_bc_at(self, x: float, y: float):
        """
        Return the boundary condition type and value at point (x, y) if it's on the boundary.

        Params:
        x: float
            x-coordinate.
        y: float
            y-coordinate.

        Returns:
            (bc_type, value): tuple
                - bc_type: {"Fixed", "Interface", "None"}
                - value: float or None
        """
        matches = []
        # Left
        if abs(x - self.bounds["x_min"]) < TOL:
            for seg in self.bc["left"]:
                if seg["y_min"] - TOL <= y <= seg["y_max"] + TOL:
                    matches.append(seg["type"])
        # Right
        if abs(x - self.bounds["x_max"]) < TOL:
            for seg in self.bc["right"]:
                if seg["y_min"] - TOL <= y <= seg["y_max"] + TOL:
                    matches.append(seg["type"])
        # Bottom
        if abs(y - self.bounds["y_min"]) < TOL:
            for seg in self.bc["bottom"]:
                if seg["x_min"] - TOL <= x <= seg["x_max"] + TOL:
                    matches.append(seg["type"])
        # Top
        if abs(y - self.bounds["y_max"]) < TOL:
            for seg in self.bc["top"]:
                if seg["x_min"] - TOL <= x <= seg["x_max"] + TOL:
                    matches.append(seg["type"])

        if not matches:
            return ("None", None)
        if "Heater" in matches:
            return ("Fixed", T_HEATER)
        if "Window" in matches:
            return ("Fixed", T_WINDOW)
        if "Wall" in matches:
            return ("Fixed", T_WALL)
        if "Interface" in matches:
            return ("Interface", None)
        return ("None", None)



class Apartment:

    """
    Represents the full apartment geometry consisting of multiple rooms.

    Has methods for:
      - Locating rooms containing a given point
      - Accessing boundary conditions
      - Building the set of global interface points shared between rooms.
    """

    def __init__(self,rooms: list[Room], h):
        """
        Initialize the apartment.

        Params:
            rooms: list[Room]
                List of room objects composing the apartment.
            h: float
                Grid spacing used for discretization.
        """
        self.h = h
        self.rooms = rooms
        self.interface_points = {} 
        self.interface_flux = {}


    def get_room_at_point(self, x: float, y: float):
        """
        Find the room that contains the point (x, y).

        Params:
            x: float
                x-coordinate.
            y: float
                y-coordinate.

        Returns:
            Room or None
                Room object if point is inside one of the rooms, otherwise None.
        """
        for r in self.rooms:
            if r.contains(x, y):
                return r
        return None
    

    def get_bc_at(self, x: float, y: float):
        """
        Get boundary condition at global coordinates (x, y).

        Params:
            x: float
                x-coordinate.
            y: float
                y-coordinate.

        Returns:
            (bc_type, value): tuple, or None.
                As returned by Room.get_bc_at().
        """
        for r in self.rooms:
            if r.contains(x, y):
                return r.get_bc_at(x, y)
        return ("None", None)
    

    def build_interface_points(self):
        """
        Build the dictionary of interface points shared between rooms.

        Scans all rooms and their boundary segments. For each boundary segment
        with type "Interface", computes the discrete grid indices of the interface nodes
        and adds them to `self.interface_points` with an initial value equal to T_WALL.
        """
        for r in self.rooms:
            for side, segments in r.bc.items():
                for seg in segments:
                    if seg["type"] == "Interface":
                        if side in ["left", "right"]:
                            x = r.bounds['x_min'] if side == "left" else r.bounds['x_max']
                            ys = np.arange(seg["y_min"] + self.h, seg["y_max"], self.h)
                            for y in ys:
                                i = int(round(x / self.h))
                                j = int(round(y / self.h))
                                self.interface_points[(i, j)] = T_WALL
                        else:
                            y = r.bounds["y_min"] if side == "bottom" else r.bounds["y_max"]
                            xs = np.arange(seg["x_min"] + self.h, seg["x_max"], self.h)
                            for x in xs:
                                i = int(round(x / self.h))
                                j = int(round(y / self.h))
                                self.interface_points[(i, j)] = T_WALL
    

    def get_all_geometry(self):
        """
        Get geometry information for all rooms.

        Returns:
            dict
                Dictionary mapping room_id -> (nx, ny, i_off, j_off),
                as returned by `Room.get_geometry()`.
        """
        return {r.id: r.get_geometry(self.h) for r in self.rooms}

