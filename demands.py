import streamlit as st
from sympy import (
    symbols, Eq, solve, diff, latex, Piecewise, Min, Poly, simplify
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

st.title("Parametric Constrained Maximization (Fixed candidate-based solver)")

func_input = st.text_input(
    "Enter function to maximize (use x, y, ^ for powers, sqrt(), min()):",
    value="x^2 + sqrt(y)"
)

transformations = (standard_transformations + (implicit_multiplication_application,))
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


# --- Utility: robust linear detection ---
def is_linear(expr):
    try:
        if expr.has(Min):
            return False
        if expr.is_polynomial(x, y):
            p = Poly(expr, x, y)
            return p.total_degree() <= 1
        return False
    except Exception:
        return False


use_min = func_expr.has(Min)
use_linear = is_linear(func_expr)


if use_min:
    min_node = func_expr
    if func_expr.func != Min:
        from sympy import preorder_traversal
        min_node = next((n for n in preorder_traversal(func_expr) if n.func == Min), None)
        if min_node is None:
            st.error("Couldn't interpret min(...) expression.")
            st.stop()
    f1, f2 = min_node.args
    eq1 = Eq(px * x + py * y, m)
    eq2 = Eq(f1, f2)
    sols = solve([eq1, eq2], (x, y), dict=True)
    if not sols:
        st.error("No solution found for min(...) case.")
        st.stop()
    sol = sols[0]
    x_sol = simplify(sol[x])
    y_sol = simplify(sol[y])
    x_func = lambdify((px, py, m), x_sol, "numpy")
    y_func = lambdify((px, py, m), y_sol, "numpy")

elif use_linear:
    a = func_expr.coeff(x)
    b = func_expr.coeff(y)

    def x_func(px_val, py_val, m_val):
        return np.where((a / px_val) >= (b / py_val), m_val / px_val, 0.0)

    def y_func(px_val, py_val, m_val):
        return np.where((a / px_val) >= (b / py_val), 0.0, m_val / py_val)

    x_sol = Piecewise((m / px, (a / px) >= (b / py)), (0, True))
    y_sol = Piecewise((0, (a / px) >= (b / py)), (m / py, True))

else:
    # --- Candidate-based nonlinear solver ---
    fx = diff(func_expr, x)
    fy = diff(func_expr, y)
    eqA = Eq(fx * py - fy * px, 0)
    eqB = Eq(px * x + py * y - m, 0)

    # Solve interior candidates
    solutions = []
    try:
        solutions = solve([eqA, eqB], (x, y), dict=True)
    except Exception:
        solutions = []

    candidates = []
    for sol in solutions:
        xs = simplify(sol.get(x, None))
        ys = simplify(sol.get(y, None))
        if xs is None or ys is None:
            continue
        if (set(xs.free_symbols) & {x, y}) or (set(ys.free_symbols) & {x, y}):
            continue
        candidates.append((xs, ys))

    # Add corners
    candidates.append((simplify(0), simplify(m / py)))
    candidates.append((simplify(m / px), simplify(0)))

    # De-duplicate
    unique = []
    seen = set()
    for xs, ys in candidates:
        key = (str(xs), str(ys))
        if key not in seen:
            seen.add(key)
            unique.append((xs, ys))
    candidates = unique

    if not candidates:
        st.error("No candidate solutions found.")
        st.stop()

    # Numeric evaluation to select best candidate
    def evaluate_candidate(xs, ys, pxv, pyv, mv):
        try:
            val = float(lambdify((px, py, m), func_expr.subs({x: xs, y: ys}), "numpy")(pxv, pyv, mv))
            if val < 0 or np.isnan(val) or np.iscomplex(val):
                return -np.inf
            return val
        except Exception:
            return -np.inf

    def select_best(pxv, pyv, mv):
        best_val = -np.inf
        best_xy = (np.nan, np.nan)
        for xs, ys in candidates:
            val = evaluate_candidate(xs, ys, pxv, pyv, mv)
            if val > best_val:
                best_val = val
                best_xy = (xs, ys)
        return best_xy

    # Lambdify functions that select the best candidate for numeric inputs
    def x_func(pxv, pyv, mv):
        xs, _ = select_best(pxv, pyv, mv)
        return float(lambdify((px, py, m), xs, "numpy")(pxv, pyv, mv))

    def y_func(pxv, pyv, mv):
        _, ys = select_best(pxv, pyv, mv)
        return float(lambdify((px, py, m), ys, "numpy")(pxv, pyv, mv))

    # For symbolic display, show first candidate (approximation)
    x_sol, y_sol = candidates[0]


# --- Display symbolic solution ---
st.markdown("### Parametric solution:")
st.latex(f"x = {latex(x_sol)}")
st.latex(f"y = {latex(y_sol)}")
try:
    f_at_solution = simplify(func_expr.subs({x: x_sol, y: y_sol}))
    if not (set(f_at_solution.free_symbols) & {x, y}):
        st.latex(f"f^* = {latex(f_at_solution)}")
except Exception:
    pass

# --- UI for varying parameters and plotting ---
param_to_vary = st.selectbox("Select parameter to vary:", ("px", "py", "m"))
fixed_params = {}
for param in ('px','py','m'):
    if param != param_to_vary:
        fixed_params[param] = st.number_input(f"Value for {param}:", value=1.0)

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
