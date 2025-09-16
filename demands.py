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
px, py, m = symbols('px py m', positive=True)

st.title("Parametric Constrained Maximization (Fixed non-linear solver)")

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
    """
    Return True if expr is (effectively) a linear polynomial in x,y:
      a*x + b*y + c  (total degree <= 1).
    For non-polynomial expressions (sqrt, etc.) returns False.
    """
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
    # --- handle min(...) specially the way you had it ---
    # Extract min arguments robustly (Min may be nested)
    min_node = func_expr
    if func_expr.func != Min:
        # try to find a Min instance inside
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
    # --- linear corner solution (same logic you had) ---
    a = func_expr.coeff(x)
    b = func_expr.coeff(y)

    def x_func(px_val, py_val, m_val):
        return np.where((a / px_val) >= (b / py_val), m_val / px_val, 0.0)

    def y_func(px_val, py_val, m_val):
        return np.where((a / px_val) >= (b / py_val), 0.0, m_val / py_val)

    # Symbolic expressions for display
    x_sol = Piecewise((m / px, (a / px) >= (b / py)), (0, True))
    y_sol = Piecewise((0, (a / px) >= (b / py)), (m / py, True))

else:
    # --- robust nonlinear solver (fixed) ---
    # Use first-order conditions: diff(f,x)/px == diff(f,y)/py  and budget
    fx = diff(func_expr, x)
    fy = diff(func_expr, y)

    # eqA encodes fx/px = fy/py  <=>  fx*py - fy*px = 0
    eqA = Eq(fx * py - fy * px, 0)
    eqB = Eq(px * x + py * y - m, 0)

    # Solve for (x, y) directly (eliminates lambda)
    solutions = []
    try:
        solutions = solve([eqA, eqB], (x, y), dict=True)
    except Exception:
        solutions = []

    # Build candidate list: interior solutions from 'solutions' plus boundary candidates
    candidates = []

    # Helper: drop solutions that are self-referential (contain x or y on RHS)
    for sol in solutions:
        xs = simplify(sol.get(x, None))
        ys = simplify(sol.get(y, None))
        if xs is None or ys is None:
            continue
        # discard if RHS expresses x or y in terms of themselves
        if (set(xs.free_symbols) & {x, y}) or (set(ys.free_symbols) & {x, y}):
            # self-referential; skip (these are the offending cases you saw)
            continue
        candidates.append((xs, ys))

    # Add corner candidates explicitly
    candidates.append((simplify(0), simplify(m / py)))   # x = 0, y = m/py
    candidates.append((simplify(m / px), simplify(0)))   # y = 0, x = m/px

    # De-duplicate candidates (string-based quick dedupe)
    unique = []
    seen = set()
    for xs, ys in candidates:
        key = (str(simplify(xs)), str(simplify(ys)))
        if key not in seen:
            seen.add(key)
            unique.append((xs, ys))
    candidates = unique

    # If we have no symbolic interior candidates, we still keep corners
    if not candidates:
        st.error("No candidate solutions found.")
        st.stop()

    # Evaluate objective numerically for each candidate at a reference positive point
    # (used to pick which candidate actually gives the highest objective)
    px_ref, py_ref, m_ref = 1.0, 1.0, 1.0

    def evaluate_candidate_obj(xs, ys, pxv=px_ref, pyv=py_ref, mv=m_ref):
        """
        Substitute candidate (xs, ys) into the objective and evaluate numerically.
        Returns -inf on failure.
        """
        try:
            obj_expr = func_expr.subs({x: xs, y: ys})
            # If obj_expr still contains x or y, reject
            if set(obj_expr.free_symbols) & {x, y}:
                return -np.inf
            obj_fun = lambdify((px, py, m), obj_expr, "numpy")
            val = obj_fun(pxv, pyv, mv)
            # turn numpy scalar/array into float
            if isinstance(val, np.ndarray):
                # pick first element if array-like
                val = float(np.asarray(val).reshape(-1)[0])
            else:
                val = float(val)
            if np.isnan(val) or np.iscomplex(val):
                return -np.inf
            return val
        except Exception:
            return -np.inf

    best_val = -np.inf
    best_candidate = None
    for xs, ys in candidates:
        val = evaluate_candidate_obj(xs, ys)
        if val > best_val:
            best_val = val
            best_candidate = (xs, ys)

    if best_candidate is None:
        st.error("Failed to select a feasible maximizer.")
        st.stop()

    x_sol, y_sol = (simplify(best_candidate[0]), simplify(best_candidate[1]))

    # Create numpy-callable functions
    x_func = lambdify((px, py, m), x_sol, "numpy")
    y_func = lambdify((px, py, m), y_sol, "numpy")


# --- Display symbolic solution ---
st.markdown("### Parametric solution:")
st.latex(f"x = {latex(x_sol)}")
st.latex(f"y = {latex(y_sol)}")

# Also show the objective value expression at the chosen candidate if available
try:
    f_at_solution = simplify(func_expr.subs({x: x_sol, y: y_sol}))
    if not (set(f_at_solution.free_symbols) & {x, y}):
        st.latex(f"f^* = {latex(f_at_solution)}")
except Exception:
    pass


# --- UI for varying parameters and plotting ---
param_to_vary = st.selectbox("Select parameter to vary:", ("px", "py", "m"))

fixed_params = {}
for param in ('px', 'py', 'm'):
    if param != param_to_vary:
        val = st.number_input(f"Value for {param}:", value=1.0)
        fixed_params[param] = float(val)

var_to_plot = st.selectbox("Select variable to plot:", ("x", "y"))
param_vals = np.linspace(0.1, 10, 200)

vals = []
for v in param_vals:
    args = []
    for p in ('px', 'py', 'm'):
        args.append(v if p == param_to_vary else fixed_params[p])
    try:
        # call appropriate function; these accept numpy arrays if lambdify supports it
        val = x_func(*args) if var_to_plot == "x" else y_func(*args)
        # numeric cleaning
        if isinstance(val, np.ndarray):
            # flatten
            val = np.asarray(val).reshape(-1)
            # if broadcasting happened, take elementwise; keep first element if shape mismatch
            if val.size == 0:
                val = np.nan
            else:
                val = val[0] if val.size == 1 else val
        # ban complex and negative where not allowed
        if np.isscalar(val):
            if np.iscomplex(val) or val < 0:
                val = np.nan
    except Exception:
        val = np.nan
    vals.append(val)

# Flip axes: x on horizontal, parameter on vertical (preserve your previous layout)
fig, ax = plt.subplots()
ax.plot(vals, param_vals, label=var_to_plot)
ax.set_xlabel(var_to_plot)
ax.set_ylabel(param_to_vary)
ax.set_title(f"{param_to_vary} vs {var_to_plot}")
ax.legend()
ax.grid(True)

st.pyplot(fig)
