import streamlit as st
from sympy import (
    symbols, Eq, solve, diff, latex, Piecewise, Min, simplify
)
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application
)
from sympy.core.sympify import SympifyError
from sympy.utilities.lambdify import lambdify
import numpy as np
import matplotlib.pyplot as plt

# --- Symbolic variables ---
x, y = symbols('x y', real=True, nonnegative=True)
px, py, m, lam = symbols('px py m lambda', positive=True)

st.title("Parametric Constrained Maximization (Symbolic Piecewise)")

func_input = st.text_input(
    "Enter function to maximize (use x, y, ^ for powers, sqrt(), min()):",
    value="sqrt(x) + y"
)

transformations = standard_transformations + (implicit_multiplication_application,)
try:
    func_expr = parse_expr(
        func_input.replace('^', '**'),
        transformations=transformations,
        local_dict={'x': x, 'y': y, 'min': Min}
    )
except SympifyError as e:
    st.error(f"Error parsing function: {e}")
    st.stop()

st.markdown(f"Parsed function: ${latex(func_expr)}$")


# --- Solve symbolically with Lagrange ---
L = func_expr - lam * (px*x + py*y - m)
eq1 = Eq(diff(L, x), 0)
eq2 = Eq(diff(L, y), 0)
eq_budget = Eq(px*x + py*y - m, 0)

# Solve for x, y in terms of λ first
try:
    sol_lambda = solve([eq1, eq2], (x, y), dict=True)
except Exception:
    sol_lambda = []

candidates = []

# Convert λ-solution to candidate expressions
for s in sol_lambda:
    xs = s.get(x)
    ys = s.get(y)
    if xs is None or ys is None:
        continue
    # Only keep solutions that do not self-reference x or y
    if set(xs.free_symbols) & {x, y} or set(ys.free_symbols) & {x, y}:
        continue
    candidates.append((simplify(xs), simplify(ys)))

# Always include corner candidates
candidates.append((0, m/py))
candidates.append((m/px, 0))

# Build Piecewise for y: use interior if y>=0 else y=0
# Then x depends accordingly
final_candidates = []
for xs, ys in candidates:
    y_piece = Piecewise((ys, ys >= 0), (0, True))
    x_piece = Piecewise((xs, ys >= 0), (m/px, True))
    final_candidates.append((x_piece, y_piece))

# Pick the first candidate for symbolic display
x_sol, y_sol = final_candidates[0]

# Lambdify for numeric evaluation
x_func = lambdify((px, py, m), x_sol, "numpy")
y_func = lambdify((px, py, m), y_sol, "numpy")

# --- Display symbolic solution ---
st.markdown("### Symbolic solution (Piecewise)")
st.latex(f"x = {latex(x_sol)}")
st.latex(f"y = {latex(y_sol)}")

# --- Plotting ---
param_to_vary = st.selectbox("Select parameter to vary:", ("px", "py", "m"))
fixed_params = {p: st.number_input(f"Value for {p}:", value=1.0) for p in ('px','py','m') if p != param_to_vary}
var_to_plot = st.selectbox("Select variable to plot:", ("x","y"))
param_vals = np.linspace(0.1, 10, 200)

vals = []
for v in param_vals:
    args = [v if p==param_to_vary else fixed_params[p] for p in ('px','py','m')]
    try:
        val = x_func(*args) if var_to_plot=='x' else y_func(*args)
        val = np.nan if np.isscalar(val) and (val < 0 or np.iscomplex(val)) else val
    except:
        val = np.nan
    vals.append(val)

fig, ax = plt.subplots()
ax.plot(vals, param_vals)
ax.set_xlabel(var_to_plot)
ax.set_ylabel(param_to_vary)
ax.set_title(f"{param_to_vary} vs {var_to_plot}")
ax.grid(True)
st.pyplot(fig)
