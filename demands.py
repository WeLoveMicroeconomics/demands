import streamlit as st
from sympy import symbols, Eq, solve, diff, latex, Piecewise, Min
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy.core.sympify import SympifyError
from sympy.utilities.lambdify import lambdify
import numpy as np
import matplotlib.pyplot as plt

# Setup symbolic variables
x, y = symbols('x y', real=True, nonnegative=True)
px, py, m = symbols('px py m', positive=True)

st.title("Parametric Constrained Maximization (Enhanced)")

func_input = st.text_input(
    "Enter function to maximize (use x, y, ^ for powers, sqrt(), min()):",
    value="x^2 + sqrt(y)"
)

transformations = (standard_transformations + (implicit_multiplication_application,))
try:
    func_expr = parse_expr(func_input.replace('^', '**'), transformations=transformations, local_dict={'x': x, 'y': y, 'min': Min})
except SympifyError as e:
    st.error(f"Error parsing function: {e}")
    st.stop()

st.markdown(f"Parsed function: ${latex(func_expr)}$")

# Determine strategy
def is_linear(expr):
    """Check if expr is strictly of the form a*x + b*y"""
    coeff_x = expr.coeff(x)
    coeff_y = expr.coeff(y)
    residual = expr - coeff_x * x - coeff_y * y
    return residual == 0 and (coeff_x != 0 or coeff_y != 0)

use_min = func_expr.has(Min)
use_linear = is_linear(func_expr)

if use_min:
    # Extract and equate min arguments
    min_args = func_expr.args if func_expr.func == Min else [arg for arg in func_expr.args if isinstance(arg, Min)][0].args
    f1, f2 = min_args
    eq1 = Eq(px * x + py * y, m)
    eq2 = Eq(f1, f2)
    sol = solve([eq1, eq2], (x, y), dict=True)
    if not sol:
        st.error("No solution found for min(...) case.")
        st.stop()
    sol = sol[0]
    x_sol = sol[x]
    y_sol = sol[y]
    x_func = lambdify((px, py, m), x_sol, "numpy")
    y_func = lambdify((px, py, m), y_sol, "numpy")

elif use_linear:
    # Extract coefficients
    a = func_expr.coeff(x)
    b = func_expr.coeff(y)

    def x_func(px_val, py_val, m_val):
        return np.where((a / px_val) >= (b / py_val), m_val / px_val, 0.0)

    def y_func(px_val, py_val, m_val):
        return np.where((a / px_val) >= (b / py_val), 0.0, m_val / py_val)

    # Symbolic expressions for display purposes
    x_sol = Piecewise((m / px, (a / px) >= (b / py)), (0, True))
    y_sol = Piecewise((0, (a / px) >= (b / py)), (m / py, True))
    
else:
    # General nonlinear case - Lagrangian Method
    lam = symbols('lambda', real=True)
    L = func_expr - lam * (px * x + py * y - m)
    eq1 = Eq(diff(L, x), 0)
    eq2 = Eq(diff(L, y), 0)
    eq3 = Eq(px * x + py * y, m)

    solutions = solve([eq1, eq2, eq3], (x, y, lam), dict=True)

    if not solutions:
        st.error("No symbolic solution found. Try a different function or parameters.")
        st.stop()

    # Filter feasible solutions safely
    feasible = []
    for sol in solutions:
        x_val = sol[x]
        y_val = sol[y]

        if (
            (x_val.is_real is True or x_val.is_real is None)
            and (y_val.is_real is True or y_val.is_real is None)
            and (x_val.is_nonnegative is True or x_val.is_nonnegative is None)
            and (y_val.is_nonnegative is True or y_val.is_nonnegative is None)
        ):
            feasible.append(sol)

    if not feasible:
        st.error("No feasible (real, nonnegative) solutions found.")
        st.stop()

    # Pick the solution that maximizes the objective
    best_sol = max(feasible, key=lambda s: func_expr.subs(s))

    x_sol = best_sol[x]
    y_sol = best_sol[y]

    x_func = lambdify((px, py, m), x_sol, "numpy")
    y_func = lambdify((px, py, m), y_sol, "numpy")

st.markdown("### Parametric solution:")
st.latex(f"x = {latex(x_sol)}")
st.latex(f"y = {latex(y_sol)}")

param_to_vary = st.selectbox("Select parameter to vary:", ("px", "py", "m"))

fixed_params = {}
for param in ('px', 'py', 'm'):
    if param != param_to_vary:
        val = st.number_input(f"Value for {param}:", value=1.0)
        fixed_params[param] = val

var_to_plot = st.selectbox("Select variable to plot:", ("x", "y"))
param_vals = np.linspace(0.1, 10, 200)

vals = []
for v in param_vals:
    args = []
    for p in ('px', 'py', 'm'):
        args.append(v if p == param_to_vary else fixed_params[p])
    try:
        val = x_func(*args) if var_to_plot == "x" else y_func(*args)
        val = np.nan if isinstance(val, complex) or val < 0 else val
    except Exception:
        val = np.nan
    vals.append(val)

# Flip axes: x on horizontal, parameter on vertical
fig, ax = plt.subplots()
ax.plot(vals, param_vals, label=var_to_plot)
ax.set_xlabel(var_to_plot)
ax.set_ylabel(param_to_vary)
ax.set_title(f"{param_to_vary} vs {var_to_plot}")
ax.legend()
ax.grid(True)

st.pyplot(fig)
