import streamlit as st
from sympy import symbols, Eq, solve, diff, latex, Piecewise, simplify
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy.core.sympify import SympifyError
from sympy.utilities.lambdify import lambdify
import numpy as np
import matplotlib.pyplot as plt

# --- Symbolic variables ---
x, y = symbols('x y', real=True, nonnegative=True)
px, py, m, lam = symbols('px py m lambda', positive=True)

st.title("Parametric Constrained Maximization with Piecewise Clipping")

func_input = st.text_input(
    "Enter function to maximize (use x, y, ^ for powers, sqrt()):",
    value="x^2 * y"
)

transformations = standard_transformations + (implicit_multiplication_application,)
try:
    func_expr = parse_expr(func_input.replace('^', '**'), transformations=transformations, local_dict={'x': x, 'y': y})
except SympifyError as e:
    st.error(f"Error parsing function: {e}")
    st.stop()

st.markdown(f"Parsed function: ${latex(func_expr)}$")

# --- Lagrangian ---
L = func_expr - lam*(px*x + py*y - m)
eq_x = Eq(diff(L, x), 0)
eq_y = Eq(diff(L, y), 0)
eq_budget = Eq(px*x + py*y - m, 0)

# Solve symbolically
solutions = solve([eq_x, eq_y, eq_budget], (x, y, lam), dict=True)
if not solutions:
    st.error("No symbolic solution found.")
    st.stop()

sol = solutions[0]
x_sol = simplify(sol[x])
y_sol = simplify(sol[y])

# --- Detect negative-prone solutions and clip ---
def make_piecewise(var_sol):
    expr = simplify(var_sol)
    try:
        # Clip at zero if it depends on parameters
        if expr.has(px, py, m):
            return Piecewise((expr, expr >= 0), (0, True))
        else:
            return expr
    except Exception:
        return expr

y_piece = make_piecewise(y_sol)

# Update x accordingly: if y=0 then x = m/px
x_piece = Piecewise((x_sol, y_piece >= 0), (m/px, True))

# Lambdify
x_func = lambdify((px, py, m), x_piece, 'numpy')
y_func = lambdify((px, py, m), y_piece, 'numpy')

st.markdown("### Parametric solution with clipping:")
st.latex(f"x = {latex(x_piece)}")
st.latex(f"y = {latex(y_piece)}")

# --- Numeric evaluation ---
st.markdown("### Evaluate numeric solution")
px_val = st.number_input("px:", value=1.0)
py_val = st.number_input("py:", value=1.0)
m_val = st.number_input("m:", value=1.0)

try:
    x_num = float(x_func(px_val, py_val, m_val))
    y_num = float(y_func(px_val, py_val, m_val))
    f_num = float(func_expr.subs({x: x_num, y: y_num}))
    st.latex(f"x^* = {x_num:.12g}, \\quad y^* = {y_num:.12g}")
    st.latex(f"f^* = {f_num:.12g}")
except Exception as e:
    st.error(f"Error evaluating numeric solution: {e}")

# --- Sweep plot ---
param_to_vary = st.selectbox("Select parameter to vary:", ("px", "py", "m"))
fixed_params = {p: st.number_input(f"Value for {p}:", value=1.0) for p in ('px','py','m') if p != param_to_vary}
var_to_plot = st.selectbox("Select variable to plot:", ("x","y"))
param_vals = np.linspace(0.1, 10, 200)

vals = []
for v in param_vals:
    args = [v if p==param_to_vary else fixed_params[p] for p in ('px','py','m')]
    try:
        val = x_func(*args) if var_to_plot=='x' else y_func(*args)
        vals.append(val)
    except:
        vals.append(np.nan)

fig, ax = plt.subplots()
ax.plot(vals, param_vals)
ax.set_xlabel(var_to_plot)
ax.set_ylabel(param_to_vary)
ax.set_title(f"{param_to_vary} vs {var_to_plot}")
ax.grid(True)
st.pyplot(fig)
