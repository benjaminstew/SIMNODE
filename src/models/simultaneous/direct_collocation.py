import numpy as np 
import pyomo.environ as pyo

class BarycentricInterpolation:
    """
    Implementation of barycentric interpolation of Lagrange polynomials, 
    when they are used as a basis for an approximation of the solution to an ODE.
    Uses formulas from https://people.maths.ox.ac.uk/trefethen/barycentric.pdf 
    to compute the barycentric weights, interpolant matrix and collocation differentiation matrix. 
    """
    def __init__(self, start_time, end_time, num_colloc_nodes, transcription_method=None, num_res_eval_nodes=None):
        self.start_time = start_time
        self.end_time = end_time
        self.num_colloc_nodes = num_colloc_nodes
        self.colloc_grid = self.generate_chebyshev_nodes_second_kind() 
        self.bary_ws = self.compute_chebyshev_second_kind_barycentric_weights()
        self.D_colloc = self.compute_colloc_derivative_matrix()

        if transcription_method == 'irrdc': 
            if num_res_eval_nodes is not None:
                self.num_res_eval_nodes = num_res_eval_nodes
                self.res_eval_grid = self.generate_equidistant_nodes()
                self.node_clash_indices, self.L_res = self.compute_interpolant_eval_matrix()
                self.D_res = self.compute_res_derivative_matrix(self.node_clash_indices)
                self.quadrature_ws = self.compute_equidistant_quadrature_weights()
            else:
                raise ValueError("Must set the number of residual evaluation nodes.")

    def generate_equidistant_nodes(self):
        '''Returns a sorted array of self.num_res_eval_nodes equidistant nodes in [self.start_time, self.end_time].'''
        return np.linspace(self.start_time, self.end_time, self.num_res_eval_nodes, dtype=np.float32)

    def generate_chebyshev_nodes_second_kind(self):
        '''
        Returns a sorted array of self.num_colloc_nodes Chebyshev nodes 
        of the second kind in the interval [self.start_time, self.end_time].
        '''
        i = np.arange(0, self.num_colloc_nodes, step=1, dtype=np.float32) # n equispaced nodes 0 -> n-1
        nodes = np.cos(i * np.pi / (self.num_colloc_nodes - 1)) # descending order 1 -> -1 as i ascends 0 -> n-1 
        nodes = 0.5 * (self.end_time - self.start_time) * nodes + 0.5 * (self.start_time + self.end_time) # scale nodes from [-1, 1] to [st, et]
        
        return np.sort(nodes, axis=0) # sort in ascending order a -> b 

    def compute_chebyshev_second_kind_barycentric_weights(self):
        '''Returns barycentric weights for self.colloc_grid.'''
        ws = np.ones(self.num_colloc_nodes, dtype=np.float32)
        for i in range(self.num_colloc_nodes):
            if i % 2 != 0:
                ws[i] *= -1
            if i == 0 or i == (self.num_colloc_nodes-1):
                ws[i] *= 0.5

        return ws

    def compute_colloc_derivative_matrix(self):
        '''Returns the differentiation matrix of the lagrange polynomial interpolant at self.colloc_grid.'''
        D = np.zeros((self.num_colloc_nodes, self.num_colloc_nodes), dtype=np.float32)

        # Compute off-diagonals
        for i in range(self.num_colloc_nodes):
            for j in range(self.num_colloc_nodes):
                if i != j:
                    D[i, j] = (self.bary_ws[j] / self.bary_ws[i]) * (1 / (self.colloc_grid[i] - self.colloc_grid[j]))

        # Compute diagonals as negative sum of off-diagonals of same row (requires all off-diagonals to be filled first)
        for i in range(self.num_colloc_nodes):
            D[i, i] = -np.sum(D[i, :])

        return D
    
    def compute_interpolant_eval_matrix(self):
        '''Returns a matrix of lagrange polynomial interpolant values at self.res_eval_grid.'''
        L = np.zeros((self.num_res_eval_nodes, self.num_colloc_nodes), dtype=np.float32)
        node_clash_indices = {}

        for k in range(self.num_res_eval_nodes):
            clash = np.where(np.isclose(self.res_eval_grid[k], self.colloc_grid, atol=1e-6))[0]

            if clash.size > 0:
                j = clash[0]
                node_clash_indices[k] = j
                # create a one-hot vector in kth row of L so that Y*[j, :] = Y_res[k, :] for clash at (k, j)
                L[k, j] = 1
                L[k, :j] = 0  
                L[k, (j+1):] = 0
                continue

            else:
                denominator = sum(
                        self.bary_ws[i] / (self.res_eval_grid[k] - self.colloc_grid[i])
                        for i in range(self.num_colloc_nodes)
                )
                for j in range(self.num_colloc_nodes):
                    numerator = self.bary_ws[j] / (self.res_eval_grid[k] - self.colloc_grid[j])
                    L[k, j] = numerator / denominator

        print(f"Number of node clashes between res eval grid and colloc grid: {len(node_clash_indices)} of {self.num_res_eval_nodes} res eval nodes.")

        return node_clash_indices, L

    def compute_res_derivative_matrix(self, node_clash_indices):
        '''Returns the differentiation matrix of the lagrange polynomial interpolant at self.res_eval_grid.'''
        D = np.zeros((self.num_res_eval_nodes, self.num_colloc_nodes), dtype=np.float32) 

        for k in range(self.num_res_eval_nodes):

            if k in node_clash_indices:
                    j = node_clash_indices[k]
                    # use barycentric formula so D_res[k, :] = D_colloc[j, :] for clash at (k, j)
                    for i in range(self.num_colloc_nodes):
                        if i != j:
                            D[k, i] = (self.bary_ws[i] / self.bary_ws[j]) * (1 / (self.colloc_grid[j] - self.colloc_grid[i]))       
                    D[k, j] = -np.sum(D[k, :])

            else:
                sigma = sum(
                    1 / (self.res_eval_grid[k] - self.colloc_grid[i])
                    for i in range(self.num_colloc_nodes)
                )
                for j in range(self.num_colloc_nodes):
                    D[k, j] = self.L_res[k, j] * (sigma - (1 / (self.res_eval_grid[k] - self.colloc_grid[j])))
 
        return D
    
    def compute_equidistant_quadrature_weights(self):
        '''Returns quadrature weight for self.res_eval_grid.'''
        h = (self.end_time - self.start_time) / self.num_res_eval_nodes
        ws = np.full(self.num_res_eval_nodes, h, dtype=np.float32)
        ws[0] *= 0.5
        ws[-1] *= 0.5
        
        return ws


        


