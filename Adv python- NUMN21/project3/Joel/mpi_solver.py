
"""

How to run: mpirun -np 4 python3 Joel/joel_solver_mpi_4rooms.py -nr n_rooms
- Always run with 4 processors (if three rooms then the last processor is unused)
- it is required to set the number of rooms when running (no default)
- other parameters are optional, see the ArgumentParser arguments
- "python3 Joel/mpi_solver.py --help" prints a list of the parameters

"""

import sys 
sys.path.append(".")
import numpy as np
from matplotlib import pyplot as plt
from marcus.room_class import Room, BoxBoundary, Dirichlet, Neumann
from mpi4py import MPI
from argparse import ArgumentParser, BooleanOptionalAction


class AppTemp:

    def __init__(self, n_rooms, res, w, n_iter, gm):
        assert res % 2 == 0, "Resolution not divisible by 2"
        self.n_rooms = n_rooms
        self.res = res
        self.dx = 1/res
        self.w = w
        self.n_iter = n_iter
        self.gm = gm

        self.T_WALL = 15.0
        self.T_HEATER = 40.0
        self.T_WINDOW = 5.0

        self.comm = MPI.Comm.Clone(MPI.COMM_WORLD)
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        assert self.size == 4, "Wrong number of processors" 

        self._create_appartment()

    def _create_appartment(self):

        if self.rank == 0: # omega 2, root
            self.omega = Room(2,1,self.res,ghost_mode=self.gm)

            # Initial values for artificial Dirichlet conditions
            self.gamma1_right = self.T_WALL*np.ones(self.res)
            self.gamma2_left = self.T_WALL*np.ones(self.res)
            if self.n_rooms == 4:
                self.gamma3_left = self.T_WALL*np.ones(int(self.res/2))

            # Boundary conditions
            if self.n_rooms == 3:
                Right = Dirichlet(np.concatenate((self.gamma2_left,self.T_WALL*np.ones(self.res))))
            else:
                Right = Dirichlet(np.concatenate((self.gamma2_left,self.gamma3_left,self.T_WALL*np.ones(int(self.res/2)))))
            self.bc = BoxBoundary(
                T = Dirichlet(self.T_HEATER*np.ones(self.res)),   
                B = Dirichlet(self.T_WINDOW*np.ones(self.res)),
                L = Dirichlet(np.concatenate((self.T_WALL*np.ones(self.res),self.gamma1_right))),
                R = Right
            )

        elif self.rank == 1: # omega 1
            self.omega = Room(1,1,self.res,ghost_mode=self.gm)
            self.bc = BoxBoundary(
                T = Dirichlet(self.T_WALL*np.ones(self.res)), 
                B = Dirichlet(self.T_WALL*np.ones(self.res)),
                L = Dirichlet(self.T_HEATER*np.ones(self.res)),        
                R = Neumann(0) # No calculations with this value
            )

            # For receiving solution from rank 0
            self.buffer = np.zeros((2*self.res,self.res))

        elif self.rank == 2: # omega 3
            self.omega = Room(1,1,self.res,ghost_mode=self.gm)
            self.bc = BoxBoundary(
                T = Dirichlet(self.T_WALL*np.ones(self.res)), 
                B = Dirichlet(self.T_WALL*np.ones(self.res)),
                L = Neumann(0), # No calculations with this value
                R = Dirichlet(self.T_HEATER*np.ones(self.res))
            ) 

            # For receiving solution from rank 0
            self.buffer = np.zeros((2*self.res,self.res))

        elif self.rank == 3: # omega 4
            self.omega = Room(0.5,0.5,self.res,ghost_mode=self.gm)
            self.bc = BoxBoundary(
                T = Dirichlet(self.T_WALL*np.ones(int(self.res/2))), 
                B = Dirichlet(self.T_HEATER*np.ones(int(self.res/2))),
                L = Neumann(0), # No calculations with this value
                R = Dirichlet(self.T_WALL*np.ones(int(self.res/2)))
            ) 
            # For receiving solution from rank 0
            self.buffer = np.zeros((2*self.res,self.res))

    def _update_bc(self, new_sol, source = 0):
        # Updates each domain's boundary conditions based on its received solution

        if self.rank == 0:
            if source == 1: # gamma 1
                self.gamma1_right = new_sol[:,-1]
                self.bc.L = Dirichlet(np.concatenate((self.T_WALL*np.ones(self.res),self.gamma1_right)))
            elif source == 2: # gamma 2
                self.gamma2_left = new_sol[:,0]
                if self.n_rooms == 3:
                    self.bc.R = Dirichlet(np.concatenate((self.gamma2_left,self.T_WALL*np.ones(self.res))))
                else:
                    self.bc.R = Dirichlet(np.concatenate((self.gamma2_left,self.gamma3_left,self.T_WALL*np.ones(int(self.res/2)))))
            elif source == 3: # gamma 3
                self.gamma3_left = new_sol[:,0]
                self.bc.R = Dirichlet(np.concatenate((self.gamma2_left,self.gamma3_left,self.T_WALL*np.ones(int(self.res/2)))))
        elif self.rank == 1:
            self.gamma1_left = (new_sol[self.res:2*self.res,1] - new_sol[self.res:2*self.res,0]) / self.dx
            self.bc.R = Neumann(self.gamma1_left)
        elif self.rank == 2:
            self.gamma2_right = (new_sol[0:self.res,self.res-2] - new_sol[0:self.res,self.res-1]) / self.dx
            #np.flip(self.gamma2_right)
            self.bc.L = Neumann(self.gamma2_right)
        elif self.rank == 3:
            self.gamma3_right = (new_sol[self.res:int(3*self.res/2),self.res-2] - new_sol[self.res:int(3*self.res/2),self.res-1]) / self.dx
            self.bc.L = Neumann(self.gamma3_right)

    def run(self, plot = True): 

        for k in range(self.n_iter):

            if k > 0 and self.rank in range(0,self.n_rooms): # for relaxation
                uold = u.copy()

            if self.rank == 0: # root
                if k > 0: # update bc
                    for i in range(1,self.n_rooms): # omega 2 borders all other rooms
                        # Size of buffer needs to match the sender's solution:
                        if i in [1,2]:
                            buffer = np.zeros((self.res,self.res))
                        elif i == 3:
                            buffer = np.zeros((int(self.res/2),int(self.res/2)))
                        self.comm.Recv(buffer, source = i)
                        self._update_bc(buffer.copy(), source = i)
                u = self.omega.solve(self.bc)
                for i in range(1,self.n_rooms):
                    self.comm.Send(u.copy(), dest = i)

            elif self.rank in range(1,self.n_rooms): # workers
                self.comm.Recv(self.buffer, source = 0)
                self._update_bc(self.buffer.copy())
                u = self.omega.solve(self.bc)
                self.comm.Send(u.copy(), dest = 0)
                            
            if k > 0 and self.rank in range(0,self.n_rooms): # relaxation
                u = self.w*u + (1-self.w)*uold
                #normdiff = np.linalg.norm(u-uold,2) / np.linalg.norm(u,2)
                #print(f"k = {k}, rank = {self.rank}: ||u - uold||/||u|| = {normdiff:.4e}")

        if plot:
            # Compiling and plotting results on root
            # The results from the workers are sent in the final iteration of the for-loop above.
            if self.rank == 0:
                Lx, Ly = 3.0, 2.0
                nx = int(Lx * self.res) 
                ny = int(Ly * self.res) 
                U = np.full((ny, nx), -10)
                # Inserting each domain's solution in U:
                U[:,self.res:2*self.res] = np.flip(u,0) # omega 2
                for i in range(1,self.n_rooms):
                    if i == 1: # omega 1
                        buffer = np.zeros((self.res,self.res))
                        self.comm.Recv(buffer, source = i)
                        U[0:self.res,0:self.res] = np.flip(buffer.copy(),0)
                    elif i == 2: # omega 3
                        buffer = np.zeros((self.res,self.res))
                        self.comm.Recv(buffer, source = i)
                        U[self.res:2*self.res,2*self.res:3*self.res] = np.flip(buffer.copy(),0)
                    elif i == 3: # omega 4
                        buffer = np.zeros((int(self.res/2),int(self.res/2)))
                        self.comm.Recv(buffer, source = i)
                        U[int(self.res/2):self.res,2*self.res:int(5*self.res/2)] = np.flip(buffer.copy(),0)

                ###### Plotting #########

                plt.figure(figsize=(9, 6))
                #Create a mask for outside, "hides" all values in U that are -10.0 
                masked_array = np.ma.array(U, mask=(U == -10.0)) 

                plt.imshow(masked_array, 
                        origin="lower", 
                        extent=[0, Lx, 0, Ly], 
                        cmap="hot", 
                        vmin=self.T_WINDOW, 
                        vmax=self.T_HEATER)
                plt.colorbar(label = r"Temperature $(\mathbf{^{\circ}C})$")

                #Add black lines for room boundaries
                plt.axvline(x=1.0, ymin=0.0, ymax=1.0, color='black', linestyle='--')
                plt.axvline(x=2.0, ymin=0.0, ymax=1.0, color='black', linestyle='--')
                plt.axhline(y=1.0, xmin=2.0/Lx, xmax=3.0/Lx, color='black', linestyle='--') 
                plt.axhline(y=1.0, xmin=0.0, xmax=1.0/Lx, color='black', linestyle='--') 
                if self.n_rooms == 4:
                    plt.axhline(y=0.5, xmin=2.0/Lx, xmax=2.5/Lx, color='black', linestyle='--') 
                    plt.axvline(x=2.5, ymin=0.5/Ly, ymax=1.0/Ly, color='black', linestyle='--')

                plt.title(f"Temperature Distribution in {self.n_rooms} Room Apartment")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.show()
        
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-nr" , "--n_rooms", type = int, required = True, choices = [3,4], help = "Number of rooms, must be 3 or 4") 
    parser.add_argument("-r", "--resolution", type = int, default = 40, help = "Grid resolution, 1/dx")
    parser.add_argument("-w","--omega", type = float, default = 0.8, help = "Relaxation factor")
    parser.add_argument("-ni", "--n_iterations", type = int, default = 10, help = "Number of Dirichlet/Neumann iterations")
    parser.add_argument("-g", "--ghost_mode", action = BooleanOptionalAction, default = False, help = "Decide if ghost mode is used for bondary conditions") 
    # Ghost mode: -g -> True, --no-g -> False, no argument -> False 
    
    args = parser.parse_args()

    app = AppTemp(n_rooms = args.n_rooms, res = args.resolution, w = args.omega, n_iter = args.n_iterations, gm = args.ghost_mode)
    app.run()




r"""
    
    - PROJECT 2!!!

    #Failed attempt to allow multiple values of parameter:
    #parser.add_argument("-r", "--resolution", type = int, nargs='*', default = 40, help = "Grid resolution, 1/dx") # doesn't work...

    - higher resolution -> more iterations needed to get accurate solution at boundaries
    - time it, in particular look for bottlenecks, doesn't feel faster than Jonathan's non-mpi solver...
    - why is flip needed when assembling solution? lower -> upper
    - border between omega 3 and 4?
    - plot in its own method?

    - Robert computer skills course
"""