import numpy as np
import sympy as sp
import time



# Create a NumPy meshgrid
x_values = np.linspace(-5, 5, 10**3)
y_values = np.linspace(-5, 5, 10**3)
X, Y = np.meshgrid(x_values, y_values)

# Define your SymPy field
x, y = sp.symbols('x y')
field = sp.sin(x) * sp.cos(y) / (1 + x**2 + y**2)
field_func = sp.lambdify((x, y), field, 'numpy')

# Use lambdify to convert the SymPy field into a function that can be evaluated over the meshgrid

# Evaluate the function over the meshgrid
start = time.time()
Z1 = field_func(X, Y)
end = time.time()
print("Time taken with sympy: ", end - start)

# Use numpy.vectorize to apply the SymPy function to each point in the meshgrid
@np.vectorize
def func(x, y):
    return np.sin(x) * np.cos(y) / (1 + x**2 + y**2)

start = time.time()
Z2 = func(X, Y)
end = time.time()
print("Time taken without sympy: ", end - start)

print(np.linalg.norm( (Z1-Z2).reshape(-1, ) ))