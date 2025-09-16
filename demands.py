import streamlit as st
from sympy import (
    symbols, Eq, solve, diff, latex, Piecewise, Min, simplify, preorder_traversal
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

st.title("Parametric Constrained Maximization (robust numeric candidate selection)")

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


# --- Helper: improved linear detection ---
def is_linear(expr):
    try:
        if expr.has(Min):
            return False
        # Only treat polynomial total-degree <= 1 as linear here
        return expr.is_polynomial(x, y) and (expr.as_poly(x, y).total_degree() <= 1)
    except Exception:
        return False


use_min = func_expr.has(Min)
use_linear = is_linear(func_expr)


if use_min:
    # --- handle Min(...) robustly ---
    min_node = func_expr if func_expr.func == Min else next(
        (n for n in preorder_traversal(func_expr) if n.func == Min), None
    )
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
    x_sym = simplify(sol[x])
    y_sym = simplify(sol[y])
    x_func = lambdify((px, py, m), x_sym, "numpy")
    y_func = lambdify((px, py, m), y_sym, "numpy")
    symbolic_candidates = [(x_sym, y_sym)]

elif use_linear:
    # --- linear corner logic ---
    a = func_expr.coeff(x)
    b = func_expr.coeff(y)

    def x_func(px_val, py_val, m_val):
        return np.where((a / px_val) >= (b / py_val), m_val / px_val, 0.0)

    def y_func(px_val, py_val, m_val):
        return np.where((a / px_val) >= (b / py_val), 0.0, m_val / py_val)

    x_sym = Piecewise((m / px, (a / px) >= (b / py)), (0, True))
    y_sym = Piecewise((0, (a / px) >= (b / py)), (m / py, True))
    symbolic_candidates = [(x_sym, y_sym)]

else:
    # --- Nonlinear: build symbolic candidates (interior + corners) ---
    fx = diff(func_expr, x)
    fy = diff(func_expr, y)

    # First-order condition ratio: fx/px = fy/py  -> fx*py - fy*px = 0
    eqA = Eq(fx * py - fy * px, 0)
    eqB = Eq(px * x + py * y - m, 0)

    try:
        sols = solve([eqA, eqB], (x, y), dict=True)
    except Exception:
        sols = []

    candidates = []
    # collect interior symbolic solutions if RHS doesn't contain x or y
    for sol in sols:
        xs = simplify(sol.get(x, None))
        ys = simplify(sol.get(y, None))
        if xs is None or ys is None:
            continue
        # skip self-referential solutions where xs or ys still contain x or y
        if (set(xs.free_symbols) & {x, y}) or (set(ys.free_symbols) & {x, y}):
            continue
        candidates.append((xs, ys))

    # always include boundary corners
    candidates.append((simplify(0), simplify(m / py)))   # x=0, y=m/py
    candidates.append((simplify(m / px), simplify(0)))   # x=m/px, y=0

    # de-duplicate
    unique = []
    seen = set()
    for xs, ys in candidates:
        key = (str(simplify(xs)), str(simplify(ys)))
        if key not in seen:
            seen.add(key)
            unique.append((xs, ys))
    candidates = unique
    symbolic_candidates = candidates

    # Precompute lambdified callables for each candidate (x,y) and their objective
    cand_x_call = []
    cand_y_call = []
    cand_obj_call = []
    for xs, ys in candidates:
        # objective expression with substitution
        obj_expr = func_expr.subs({x: xs, y: ys})
        # If obj_expr still depends on x or y, mark as invalid (skip numerics)
        if set(obj_expr.free_symbols) & {x, y}:
            # we won't use this candidate numerically
            cand_x_call.append(None)
            cand_y_call.append(None)
            cand_obj_call.append(None)
            continue
        try:
            cand_x_call.append(lambdify((px, py, m), xs, "numpy"))
            cand_y_call.append(lambdify((px, py, m), ys, "numpy"))
            cand_obj_call.append(lambdify((px, py, m), obj_expr, "numpy"))
        except Exception:
            cand_x_call.append(None)
            cand_y_call.append(None)
            cand_obj_call.append(None)

    # numeric selector: choose best feasible candidate for given numeric px,py,m
    def choose_candidate(px_val, py_val, m_val):
        # accept scalar values only; callers will vectorize if passed arrays
        best_val = -np.inf
        best_xy = (np.nan, np.nan)
        for i, (xs_call, ys_call, obj_call) in enumerate(zip(cand_x_call, cand_y_call, cand_obj_call)):
            if xs_call is None or ys_call is None or obj_call is None:
                continue
            try:
                xv = xs_call(px_val, py_val, m_val)
                yv = ys_call(px_val, py_val, m_val)
                ov = obj_call(px_val, py_val, m_val)
                # unify numpy / python scalar
                xv = np.asarray(xv).reshape(-1)[0].item() if np.asarray(xv).size > 0 else float(xv)
                yv = np.asarray(yv).reshape(-1)[0].item() if np.asarray(yv).size > 0 else float(yv)
                ov = np.asarray(ov).reshape(-1)[0].item() if np.asarray(ov).size > 0 else float(ov)
            except Exception:
                continue
            # feasibility: nonnegative and real
            if not np.isfinite(xv) or not np.isfinite(yv):
                continue
            # small tolerance for numerical noise
            tol = 1e-9
            if (xv >= -tol) and (yv >= -tol):
                # clip small negative round-off to zero
                xv = max(xv, 0.0)
                yv = max(yv, 0.0)
                if ov > best_val:
                    best_val = ov
                    best_xy = (xv, yv)
        # If no candidate selected (rare if px,py,m positive), fallback to corners explicitly
        if not np.isfinite(best_xy[0]) or not np.isfinite(best_xy[1]):
            # fallback to x=m/px, y=0 or x=0,y=m/py whichever feasible and larger
            try:
                xc = float(m_val / px_val)
                yc = 0.0
                valc = float(func_expr.subs({x: xc, y: yc}))
            except Exception:
                valc = -np.inf
            try:
                xc2 = 0.0
                yc2 = float(m_val / py_val)
                valc2 = float(func_expr.subs({x: xc2, y: yc2}))
            except Exception:
                valc2 = -np.inf
            if valc >= valc2:
                return (xc, yc)
            else:
                return (xc2, yc2)
        return best_xy

    # vectorized x_func and y_func that accept scalar or numpy arrays
    def x_func(px_val, py_val, m_val):
        arr_px = np.asarray(px_val)
        arr_py = np.asarray(py_val)
        arr_m = np.asarray(m_val)
        # broadcasting rules: flatten parameter arrays to 1d of same length
        if arr_px.shape == arr_py.shape == arr_m.shape and arr_px.ndim >= 1:
            n = arr_px.size
            out = np.empty(n, dtype=float)
            for i in range(n):
                xi, _ = choose_candidate(float(arr_px.reshape(-1)[i]), float(arr_py.reshape(-1)[i]), float(arr_m.reshape(-1)[i]))
                out[i] = xi
            return out.reshape(arr_px.shape)
        else:
            # scalar or mixed shapes: treat as scalar
            xi, _ = choose_candidate(float(np.asarray(px_val)), float(np.asarray(py_val)), float(np.asarray(m_val)))
            return xi

    def y_func(px_val, py_val, m_val):
        arr_px = np.asarray(px_val)
        arr_py = np.asarray(py_val)
        arr_m = np.asarray(m_val)
        if arr_px.shape == arr_py.shape == arr_m.shape and arr_px.ndim >= 1:
            n = arr_px.size
            out = np.empty(n, dtype=float)
            for i in range(n):
                _, yi = choose_candidate(float(arr_px.reshape(-1)[i]), float(arr_py.reshape(-1)[i]), float(arr_m.reshape(-1)[i]))
                out[i] = yi
            return out.reshape(arr_px.shape)
        else:
            _, yi = choose_candidate(float(np.asarray(px_val)), float(np.asarray(py_val)), float(np.asarray(m_val)))
            return yi


# --- Display symbolic candidate list for transparency ---
st.markdown("### Symbolic candidate solutions (interior + corners)")
for i, (xs_sym, ys_sym) in enumerate(symbolic_candidates):
    st.latex(f"\\text{{cand}}_{{{i}}}:\\quad x = {latex(simplify(xs_sym))},\\; y = {latex(simplify(ys_sym))}")

st.markdown("_(Numeric selection chooses, for each numeric (px,py,m), the feasible candidate with highest objective)_")

# Let user test a specific numeric triple to see which solution is chosen
st.markdown("### Evaluate chosen solution at specific parameters")
px_test = st.number_input("px (test)", value=1.0)
py_test = st.number_input("py (test)", value=1.0)
m_test = st.number_input("m (test)", value=1.0)

# compute numeric chosen solution for the test triple
try:
    x_chosen = float(x_func(px_test, py_test, m_test))
    y_chosen = float(y_func(px_test, py_test, m_test))
    st.write(f"Chosen numeric solution at (px={px_test}, py={py_test}, m={m_test}):")
    st.latex(f"x^* = {x_chosen:.12g},\\quad y^* = {y_chosen:.12g}")
    try:
        f_val =_
