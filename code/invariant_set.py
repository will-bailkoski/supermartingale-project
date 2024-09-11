import numpy as np
import gurobipy as gp
from gurobipy import GRB


# This invariant set method simply creates a hypersphere around a point. Not very accurate, but simple
class InvariantBall:
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius

        if not isinstance(radius, (int, float)) or radius <= 0:
            raise ValueError("Radius must be a positive number")

    def contains_point(self, point):
        point = np.array(point)

        if point.shape != self.center.shape:
            raise ValueError(
                f"Point must have the same dimensions as the center. Point: {point.shape} Center: {self.center.shape}")

        distance = np.linalg.norm(point - self.center)
        return distance <= self.radius


    def check_invariant_ball(self, n, C, B, r, domain_bounds):
        # Returns True if this is verified to be a correct invariant set
        model = gp.Model()

        # Variables
        x = model.addVars(n, lb=domain_bounds[0], ub=domain_bounds[1], name="X")

        # Invariant ball constraint
        squared_distance = gp.quicksum((x[i] - self.center[i]) ** 2 for i in range(n))
        model.addConstr(squared_distance + 1e-6 <= self.radius ** 2, "InvariantBall")

        # Transition kernel P(x)
        Cx = [gp.quicksum(C[i][j] * x[j] for j in range(n)) for i in range(n)]
        Bphi = [None] * n
        P_x = [None] * n
        phi_x = model.addVars(n, vtype=GRB.BINARY)
        M = 1e6

        for i in range(n):
            model.addConstr(x[i] + M * phi_x[i] >= 0, name=f"Constr1_{i}")
            model.addConstr(x[i] <= M * (1 - phi_x[i]), name=f"Constr2_{i}")
            Bphi[i] = gp.quicksum(B[i][j] * phi_x[j] for j in range(n))
            P_x[i] = Cx[i] + r[i] - Bphi[i]

        # Define transformed points and their constraints
        transforms = [
            [P_x[i] * 1.1 for i in range(n)],
            [P_x[i] * 0.9 for i in range(n)],
            [P_x[i] * (1.1 if i == 0 else 0.9) for i in range(n)],
            [P_x[i] * (0.9 if i == 0 else 1.1) for i in range(n)],
        ]

        for idx, transform in enumerate(transforms):
            dist = gp.quicksum((transform[i] - self.center[i]) ** 2 for i in range(n))
            model.addConstr(dist >= self.radius ** 2 + 1e-6, name=f"Transform_{idx}")

        # Set a trivial objective since we are only interested in feasibility
        model.setObjective(0)

        # Optimize the model
        model.optimize()

        # Check if a solution was found
        if model.status == GRB.OPTIMAL:
            counterexample = [x[i].X for i in range(n)]
            return False, counterexample
        else:
            return True, None


# This is the method used in the literature. Produces a robust invariant set by generating a polyhedron's boundary
def construct_invariant_set(resolution, n, C, Psi):

    # Compute x_0 (initial random vector)
    # x_0 = 2 * np.random.rand(n, 1) - np.ones((n, 1)) - np.random.rand(n, 1)

    # Start with A as the identity matrix
    A = np.eye(n)
    b = np.zeros((n, 1))

    for k in range(1, resolution + 1):
        A = np.vstack((np.eye(n), np.dot(A, C)))
        b_add = np.dot((np.eye(n) - np.linalg.matrix_power(C, k)), np.linalg.inv(np.eye(n) - C)).dot(Psi)
        b = np.vstack((b, b_add))

    # b = b + np.dot(A, x_0)

    # print(b <= np.dot(A, [[-4], [-4]]))
    return A, b

def check_invariant_set(n, C, B, r, A, b, domain_bounds):
    # Returns True if this is verified to be a correct invariant set
    model = gp.Model()

    # Variables
    x = model.addVars(n, lb=domain_bounds[0], ub=domain_bounds[1], name="X")

    # Invariant constraint
    for i in range(len(A)):
        model.addConstr(gp.quicksum(A[i][j] * x[j] for j in range(n)) <= b[i],
                        name=f"InSetConstraint_{i}")

    # Transition kernel P(x)
    Cx = [gp.quicksum(C[i][j] * x[j] for j in range(n)) for i in range(n)]
    Bphi = [None] * n
    P_x = [None] * n
    phi_x = model.addVars(n, vtype=GRB.BINARY)
    M = 1e6

    for i in range(n):
        model.addConstr(x[i] + M * phi_x[i] >= 0, name=f"Constr1_{i}")
        model.addConstr(x[i] <= M * (1 - phi_x[i]), name=f"Constr2_{i}")
        Bphi[i] = gp.quicksum(B[i][j] * phi_x[j] for j in range(n))
        P_x[i] = Cx[i] + r[i] - Bphi[i]

    # Define transformed points and their constraints
    transforms = [
        [P_x[i] * 1.1 for i in range(n)],
        [P_x[i] * 0.9 for i in range(n)],
        [P_x[i] * (1.1 if i == 0 else 0.9) for i in range(n)],
        [P_x[i] * (0.9 if i == 0 else 1.1) for i in range(n)],
    ]

    # find point outside of invariant set
    for idx, transform in enumerate(transforms):
        model.addConstrs(
            (gp.quicksum(A[i][j] * transform[j] for j in range(n)) - 1e-6 >= b[i]
             for i in range(len(b))),
            name=f"InvariantCheck_{idx}"
        )


    # Set a trivial objective since we are only interested in feasibility
    model.setObjective(0)

    # Optimize the model
    model.optimize()

    # Check if a solution was found
    if model.status == GRB.OPTIMAL:
        counterexample = [x[i].X for i in range(n)]
        return False, counterexample
    else:
        return True, None

def find_point_in_invariant_set(n, A, b):
    model = gp.Model()

    # Variables
    x = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="X")

    # Invariant constraint
    for i in range(len(A)):
        model.addConstr(gp.quicksum(A[i][j] * x[j] for j in range(n)) <= b[i],
                        name=f"InSetConstraint_{i}")


    if model.status == GRB.OPTIMAL:
        counterexample = [x[i].X for i in range(n)]
        return counterexample
    else:
        return None
