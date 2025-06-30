import numpy as np
from scipy.optimize import minimize, differential_evolution, dual_annealing

# Define the objective function to maximize (negating it to minimize)
def objective(x):
    # Coefficients for the linear terms
    linear_terms = np.array([0.0207, 0.1236, 0.3288, 0.28, 0.1963, 0.4, 0.2157, 0.0404, 0.0404])
    
    # The sum of the linear terms and the quadratic penalty term
    linear_sum = np.dot(linear_terms, x)
    penalty = -0.012 * np.sum(x**2)  # Penalty term
    return -(linear_sum + penalty)  # Negate to maximize

# First approach: Use a global optimizer with integer rounding
def solve_with_integer_rounding():
    # Define the constraint: x1 + x2 + ... + x9 <= 100
    def constraint(x):
        return 100 - np.sum(x)
    
    # Bounds for the variables
    bounds = [(0, 100) for _ in range(9)]
    
    # Define the constraints dictionary
    constraints = ({'type': 'ineq', 'fun': constraint})
    
    # Perform the optimization with continuous variables
    result = minimize(objective, np.ones(9), bounds=bounds, constraints=constraints, method='SLSQP')
    
    # Round to integer values and then refine
    x_int = np.round(result.x).astype(int)
    
    # Ensure sum constraint is still satisfied after rounding
    sum_x = np.sum(x_int)
    if sum_x > 100:
        # Adjust by reducing values if we exceed the constraint
        indices = np.argsort(x_int)
        reduction = sum_x - 100
        for i in range(reduction):
            idx = indices[-(i % len(indices)) - 1]
            if x_int[idx] > 0:
                x_int[idx] -= 1
    
    return x_int, -objective(x_int)  # Return the solution and the maximized objective

# Second approach: Use a global optimizer that can handle discrete variables
def solve_with_global_optimizer():
    # Define bounds for integer variables (0 to 100)
    bounds = [(0, 100) for _ in range(9)]
    
    # Use differential evolution with a custom callback to enforce integer values
    def modified_objective(x):
        # Round to integers
        x_int = np.round(x).astype(int)
        
        # Check sum constraint
        if np.sum(x_int) > 100:
            return 1e10  # Large penalty for violation
        
        return objective(x_int)
    
    # Use dual annealing which works well with discrete variables
    result = dual_annealing(modified_objective, bounds, maxiter=1000)
    
    # Get integer solution
    x_int = np.round(result.x).astype(int)
    
    return x_int, -objective(x_int)

# Third approach: Direct search over integer space
def solve_with_direct_search():
    best_x = None
    best_value = float('-inf')
    
    # Start with an initial guess
    x = np.zeros(9, dtype=int)
    
    # Simple greedy algorithm
    remaining = 100
    coefficients = np.array([0.0207, 0.1236, 0.3288, 0.28, 0.1963, 0.4, 0.2157, 0.0404, 0.0404])
    
    # Calculate value per unit for each variable
    value_per_unit = coefficients.copy()
    
    # Sort indices by value per unit (highest first)
    sorted_indices = np.argsort(-value_per_unit)
    
    # Allocate resources greedily
    for idx in sorted_indices:
        # For each variable, find optimal integer allocation considering quadratic penalty
        best_allocation = 0
        best_improvement = 0
        
        for allocation in range(1, remaining + 1):
            temp_x = x.copy()
            temp_x[idx] = allocation
            new_value = -objective(temp_x)
            old_value = -objective(x)
            improvement = new_value - old_value
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_allocation = allocation
            elif improvement < 0:  # Stop if value starts decreasing
                break
        
        x[idx] = best_allocation
        remaining -= best_allocation
        
        if remaining == 0:
            break
    
    return x, -objective(x)

# Run the algorithms and compare results
x_rounded, value_rounded = solve_with_integer_rounding()
x_global, value_global = solve_with_global_optimizer()
x_direct, value_direct = solve_with_direct_search()

# Choose the best solution
solutions = [
    (x_rounded, value_rounded),
    (x_global, value_global),
    (x_direct, value_direct)
]
best_solution = max(solutions, key=lambda s: s[1])
x_best, max_value = best_solution

print("Best integer solution:")
print("Variables:", x_best)
print("Objective value:", max_value)
print("Sum of variables:", np.sum(x_best))