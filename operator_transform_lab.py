import math
import numpy as np
import sympy as sp
import streamlit as st


# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="Operator Transform Laboratory - Laplace Section",
    page_icon="∂",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    .katex-display {
        margin: 0.7rem 0 0.7rem 0 !important;
    }
    h1, h2, h3 {
        letter-spacing: 0.2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Symbols
# ============================================================
s = sp.Symbol("s", positive=True, real=True)
t = sp.Symbol("t", positive=True, real=True)
a = sp.Symbol("a", real=True)
b = sp.Symbol("b", real=True)
k = sp.Symbol("k", positive=True, real=True)
nu = sp.Symbol("nu", real=True)
n = sp.Symbol("n", integer=True, nonnegative=True)

# ============================================================
# Helpers
# ============================================================
def safe_latex(expr):
    try:
        return sp.latex(sp.simplify(expr))
    except Exception:
        return str(expr)

# ============================================================
# Section 2.1: Operator-based reconstruction
# ============================================================
def render_operator_reconstruction():
    st.header("2.1 Operator-based reconstruction of the Laplace transform")

    st.markdown(r"""
    **Definition 2.1.** Let $\mathcal{A}$ denote the class of functions $f$ that are analytic 
    in a neighborhood of $t = 0$ and admit a Maclaurin expansion
    """)
    st.latex(r"f(t) = \sum_{n=0}^\infty a_n t^n.")

    st.markdown(r"""
    The operator-based transform $T$ of $f$ is defined by
    """)
    st.latex(r"T\{f\}(s) := \sum_{n=0}^\infty a_n (-D_s)^n \left( \frac{1}{s} \right),")
    st.markdown(r"whenever the above operator series is convergent.")

    st.markdown(r"""
    **Theorem 2.2** (Equivalence with the classical Laplace transform). 
    For every function $f \in \mathcal{A}$ and for every $s$ such that the operator series 
    defining $T\{f\}(s)$ is convergent, one has
    """)
    st.latex(r"T\{f\}(s) = \int_0^\infty e^{-st}f(t) \, dt,")
    st.markdown(r"whenever the integral exists.")

    st.markdown(r"**Proof.** Starting from the Maclaurin expansion")
    st.latex(r"f(t) = \sum_{n=0}^\infty a_n t^n,")

    st.markdown(r"""
    the differential planting mechanism acts on the rational kernel $1/s$ via repeated 
    differentiation, yielding the fundamental identity
    """)
    st.latex(r"D_s^n \left( \frac{1}{s} \right) = (-1)^n \frac{n!}{s^{n+1}}.")

    st.markdown(r"Substituting this expression into the operator series gives")
    st.latex(r"T\{f\}(s) = \sum_{n=0}^\infty a_n \frac{n!}{s^{n+1}}.")

    st.markdown(r"""
    To reveal the integral structure hidden in this purely differential expression, 
    we invoke the Gamma-function identity
    """)
    st.latex(r"\frac{1}{s^{n+1}} = \frac{1}{n!} \int_0^\infty t^n e^{-st} \, dt, \quad s > 0.")

    st.markdown(r"""
    Inserting this representation and interchanging summation and integration 
    (justified by absolute convergence) yields
    """)
    st.latex(r"""
    T\{f\}(s) = \int_0^\infty e^{-st} \left( \sum_{n=0}^\infty a_n t^n \right) dt 
    = \int_0^\infty e^{-st} f(t) dt.
    """)

    st.markdown(r"This is precisely the classical Laplace transform of $f$, completing the proof.")

    st.info(r"""
    **Remark 1** (Convergence domain). The definition of $T$ does not impose an explicit 
    a priori condition on $s$. The admissible values of $s$ are determined implicitly by the 
    convergence of the operator series. In concrete examples, this convergence domain can be 
    computed explicitly and coincides with the classical region of convergence of the Laplace 
    transform (e.g. trigonometric, exponential, and Bessel-type functions).
    """)

    st.info(r"""
    **Remark 2** (Emergent integral structure). Within the present framework, the Laplace 
    transform is not postulated a priori as an integral operator. Rather, it emerges naturally 
    from the internal differential planting mechanism acting on the base kernel $1/s$. 
    The exponential kernel $e^{-st}$ appears only after invoking the Gamma-function identity, 
    revealing the integral representation as a consequence of an underlying purely differential 
    structure. This does not invalidate the classical integral definition, but rather explains 
    its structural origin.
    """)

# ============================================================
# Section 2.2: Rank bookkeeping
# ============================================================
def render_rank_bookkeeping():
    st.header("2.2 Rank bookkeeping used in the case studies")

    st.markdown(r"""
    The notion of rank introduced here serves purely as a bookkeeping device.  
    Thus, the derivative rank is directly inherited from the power of the variable in the 
    Maclaurin expansion of the input function.  
    We use the differential operator
    """)
    st.latex(r"D := \frac{d}{ds},")
    st.markdown(r"acting on the base kernel $1/s$.")

    st.markdown(r"""
    In the Maclaurin-planting construction, each power $t^n$ selects a derivative rank $D^n$ 
    to be applied to $1/s$. This notion of rank is the only bookkeeping device required in 
    the applications below.
    """)

    st.subheader("Basic kernel identity")
    st.markdown(r"For every $n \in \mathbb{N}_0$ and $s > 0$,")
    st.latex(r"""
    D^n \left( \frac{1}{s} \right) = (-1)^n \frac{n!}{s^{n+1}}, 
    \quad \text{hence} \quad 
    (-D)^n \left( \frac{1}{s} \right) = \frac{n!}{s^{n+1}}.
    """)

    st.subheader("Rank parity (sign absorption)")
    st.markdown(r"""
    The factor $(-1)^n$ arising from $D^n(1/s)$ is absorbed by writing the planted operator 
    as $(-D)^n$. This convention keeps all subsequent computations sign-clean.
    """)

    st.subheader("Rank lowering by multiplication with $s$")
    st.markdown(r"For $n \geq 1$,")
    st.latex(r"s D^n \left( \frac{1}{s} \right) = -n D^{n-1} \left( \frac{1}{s} \right).")
    st.markdown(r"More generally, for $k \leq n$,")
    st.latex(r"""
    s^k D^n \left( \frac{1}{s} \right) = (-1)^k n^{\underline{k}} D^{n-k} \left( \frac{1}{s} \right), 
    \quad n^{\underline{k}} := n(n-1)\cdots(n-k+1).
    """)

    st.subheader("Rank shift induced by division by $t^m$")
    st.markdown(r"If")
    st.latex(r"""
    f(t) = \sum_{n=0}^\infty c_n t^n, \quad 
    \frac{f(t)}{t^m} = \sum_{n=m}^\infty c_n t^{n-m},
    """)
    st.markdown(r"then the planted rank is lowered accordingly, and the operator-based transform satisfies")
    st.latex(r"""
    T\left(\frac{f(t)}{t^m}\right)(s) = \sum_{n=m}^\infty c_n (-D)^{n-m} \left(\frac{1}{s}\right),
    """)
    st.markdown(r"whenever the resulting series is admissible.")

# ============================================================
# Section 2.3: Kernel shifting
# ============================================================
def render_kernel_shifting():
    st.header("2.3 Kernel shifting and rank preservation")

    st.markdown(r"""
    A key structural feature of the planted-operator framework is the kernel shift
    """)
    st.latex(r"\frac{1}{s} \to \frac{1}{s-a},")
    st.markdown(r"which encodes exponential tilting in the $t$-domain.")

    st.subheader("Shifted operator definition")
    st.markdown(r"Given the Maclaurin expansion")
    st.latex(r"f(t) = \sum_{n=0}^{\infty} a_n t^n,")
    st.markdown(r"we define the shifted planted transform by")
    st.latex(r"""
    T_a\{f\}(s) := \sum_{n=0}^{\infty} a_n (-D)^n \left(\frac{1}{s-a}\right), 
    \quad D = \frac{d}{ds}.
    """)

    st.subheader("Shifted kernel identity")
    st.markdown(r"For every $n \in \mathbb{N}_0$ and $s > a$,")
    st.latex(r"""
    D^n \left(\frac{1}{s-a}\right) = (-1)^n \frac{n!}{(s-a)^{n+1}}, 
    \quad \text{hence} \quad 
    (-D)^n \left(\frac{1}{s-a}\right) = \frac{n!}{(s-a)^{n+1}}.
    """)

    st.subheader("Rank invariance under shifting")
    st.markdown(r"""
    The shift $s \mapsto s-a$ modifies only the kernel location and does not alter the 
    derivative rank selected by each Maclaurin coefficient. Thus, rank bookkeeping is 
    preserved under exponential tilting.
    """)

    st.subheader("Interpretation")
    st.markdown(r"""
    The shift $s \mapsto s-a$ corresponds to multiplication by $e^{at}$ in the $t$-domain, 
    exactly mirroring the classical Laplace shift property, but derived here purely at the 
    operator level.
    """)

# ============================================================
# Section 2.4: Shifted kernels and repeated poles
# ============================================================
def render_shifted_kernels():
    st.header("2.4 Shifted kernels and repeated poles")

    st.subheader("Simple shift")
    st.latex(r"""
    \frac{1}{s^2-1} = \frac{1}{2} \left(\frac{1}{s-1} - \frac{1}{s+1}\right) 
    \implies D^n\left(\frac{1}{s^2-1}\right) = \frac{1}{2} D^n\left(\frac{1}{s-1}\right) 
    - \frac{1}{2} D^n\left(\frac{1}{s+1}\right).
    """)

    st.subheader("Repeated poles")
    st.markdown(r"For integers $m \geq 1$ and $n \geq 0$,")
    st.latex(r"""
    D^n\left(\frac{1}{(s-a)^m}\right) = (-1)^n \frac{\Gamma(m+n)}{\Gamma(m)} (s-a)^{-(m+n)}.
    """)

# ============================================================
# Case Studies Data
# ============================================================
def get_laplace_cases_detailed():
    cases = {}

    # Case A: f(t) = 1
    cases["Case A: f(t) = 1"] = {
        "title": "Case A: f(t) = 1",
        "function": r"f(t) = 1",
        "series": r"1 = \sum_{n \geq 0} a_n t^n \quad \text{with } a_0 = 1,\; a_{n>0} = 0.",
        "coefficients": r"a_0 = 1,\quad a_{n>0} = 0.",
        "rank": r"D^0",
        "plant_sum": [
            r"T\{1\}(s) = D^0\left(\frac{1}{s}\right) = \frac{1}{s}",
        ],
        "closed_form": r"T\{1\}(s) = \frac{1}{s}",
        "condition": r"s > 0",
    }

    # Case B: f(t) = t^n
    cases["Case B: f(t) = t^n"] = {
        "title": "Case B: f(t) = t^n",
        "function": r"f(t) = t^n",
        "series": r"t^n = \sum_{k \geq 0} a_k t^k,\quad a_k = \delta_{k,n}",
        "coefficients": r"a_k = \delta_{k,n} \quad \text{(only the } k = n \text{ term is nonzero)}.",
        "rank": r"D^n",
        "plant_sum": [
            r"T\{t^n\}(s) = (-1)^n D^n\left(\frac{1}{s}\right)",
            r"D^n\left(\frac{1}{s}\right) = (-1)^n \frac{n!}{s^{n+1}}",
            r"T\{t^n\}(s) = (-1)^n \cdot (-1)^n \frac{n!}{s^{n+1}} = \frac{n!}{s^{n+1}}",
        ],
        "closed_form": r"T\{t^n\}(s) = \frac{n!}{s^{n+1}}",
        "condition": r"s > 0",
    }

    # Case C: f(t) = e^(at)
    cases["Case C: f(t) = e^{at}"] = {
        "title": "Case C: f(t) = e^{at}",
        "function": r"f(t) = e^{at}",
        "series": r"e^{at} = \sum_{n \geq 0} \frac{a^n}{n!} t^n",
        "coefficients": r"a_n = \frac{a^n}{n!}",
        "rank": r"D^n",
        "plant_sum": [
            r"T\{e^{at}\}(s) = \sum_{n \geq 0} (-1)^n a_n D^n\left(\frac{1}{s}\right)",
            r"= \sum_{n \geq 0} \frac{a^n}{n!} (-1)^n \cdot (-1)^n \frac{n!}{s^{n+1}}",
            r"= \frac{1}{s} \sum_{n \geq 0} \left(\frac{a}{s}\right)^n",
            r"= \frac{1}{s} \cdot \frac{1}{1 - \frac{a}{s}} = \frac{1}{s - a}",
        ],
        "closed_form": r"T\{e^{at}\}(s) = \frac{1}{s - a}",
        "condition": r"\left|\frac{a}{s}\right| < 1",
    }

    # Case D: f(t) = cos(bt)
    cases["Case D: f(t) = cos(bt)"] = {
        "title": "Case D: f(t) = cos(bt)",
        "function": r"f(t) = \cos(bt)",
        "series": r"\cos(bt) = \sum_{n \geq 0} \frac{(-1)^n b^{2n}}{(2n)!} t^{2n}",
        "coefficients": r"a_{2n} = \frac{(-1)^n b^{2n}}{(2n)!}",
        "rank": r"D^{2n}",
        "plant_sum": [
            r"T\{\cos(bt)\}(s) = \sum_{n \geq 0} a_{2n} (-1)^{2n} D^{2n}\left(\frac{1}{s}\right)",
            r"= \sum_{n \geq 0} \frac{(-1)^n b^{2n}}{(2n)!} \cdot \frac{(2n)!}{s^{2n+1}}",
            r"= \frac{1}{s} \sum_{n \geq 0} \left(-\frac{b^2}{s^2}\right)^n",
            r"= \frac{1}{s} \cdot \frac{1}{1 + \frac{b^2}{s^2}} = \frac{s}{s^2 + b^2}",
        ],
        "closed_form": r"T\{\cos(bt)\}(s) = \frac{s}{s^2 + b^2}",
        "condition": r"\left|\frac{b}{s}\right| < 1",
    }

    # Case E: f(t) = sin(bt)
    cases["Case E: f(t) = sin(bt)"] = {
        "title": "Case E: f(t) = sin(bt)",
        "function": r"f(t) = \sin(bt)",
        "series": r"\sin(bt) = \sum_{n \geq 0} \frac{(-1)^n b^{2n+1}}{(2n+1)!} t^{2n+1}",
        "coefficients": r"a_{2n+1} = \frac{(-1)^n b^{2n+1}}{(2n+1)!}",
        "rank": r"D^{2n+1}",
        "plant_sum": [
            r"T\{\sin(bt)\}(s) = \sum_{n \geq 0} a_{2n+1} (-1)^{2n+1} D^{2n+1}\left(\frac{1}{s}\right)",
            r"= \sum_{n \geq 0} \frac{(-1)^n b^{2n+1}}{(2n+1)!} \cdot \frac{(2n+1)!}{s^{2n+2}}",
            r"= \frac{b}{s^2} \sum_{n \geq 0} \left(-\frac{b^2}{s^2}\right)^n",
            r"= \frac{b}{s^2} \cdot \frac{1}{1 + \frac{b^2}{s^2}} = \frac{b}{s^2 + b^2}",
        ],
        "closed_form": r"T\{\sin(bt)\}(s) = \frac{b}{s^2 + b^2}",
        "condition": r"\left|\frac{b}{s}\right| < 1",
    }

    # Case F: f(t) = t sin(bt)
    cases["Case F: f(t) = t sin(bt)"] = {
        "title": "Case F: f(t) = t sin(bt)",
        "function": r"f(t) = t\sin(bt)",
        "series": r"t\sin(bt) = \sum_{n \geq 0} \frac{(-1)^n b^{2n+1}}{(2n+1)!} t^{2n+2}",
        "coefficients": r"a_{2n+2} = \frac{(-1)^n b^{2n+1}}{(2n+1)!}",
        "rank": r"D^{2n+2}",
        "plant_sum": [
            r"T\{t\sin(bt)\}(s) = \sum_{n \geq 0} a_{2n+2} (-1)^{2n+2} D^{2n+2}\left(\frac{1}{s}\right)",
            r"= \sum_{n \geq 0} \frac{(-1)^n b^{2n+1}}{(2n+1)!} \cdot \frac{(2n+2)!}{s^{2n+3}}",
            r"= \frac{b}{s^3} \sum_{n \geq 0} (2n+2) \left(-\frac{b^2}{s^2}\right)^n",
            r"\text{Let } u = -\frac{b^2}{s^2}: \quad \sum_{n \geq 0} (2n+2) u^n = \frac{2}{(1-u)^2}",
            r"= \frac{b}{s^3} \cdot \frac{2}{\left(1 + \frac{b^2}{s^2}\right)^2} = \frac{2bs}{(s^2 + b^2)^2}",
        ],
        "closed_form": r"T\{t\sin(bt)\}(s) = \frac{2bs}{(s^2 + b^2)^2}",
        "condition": r"s > |b|",
    }

    # Case G: f(t) = t cos(bt)
    cases["Case G: f(t) = t cos(bt)"] = {
        "title": "Case G: f(t) = t cos(bt)",
        "function": r"f(t) = t\cos(bt)",
        "series": r"t\cos(bt) = \sum_{n \geq 0} \frac{(-1)^n b^{2n}}{(2n)!} t^{2n+1}",
        "coefficients": r"a_{2n+1} = \frac{(-1)^n b^{2n}}{(2n)!}",
        "rank": r"D^{2n+1}",
        "plant_sum": [
            r"T\{t\cos(bt)\}(s) = \sum_{n \geq 0} a_{2n+1} (-1)^{2n+1} D^{2n+1}\left(\frac{1}{s}\right)",
            r"= \sum_{n \geq 0} \frac{(-1)^n b^{2n}}{(2n)!} \cdot \frac{(2n+1)!}{s^{2n+2}}",
            r"= \frac{1}{s^2} \sum_{n \geq 0} (2n+1) \left(-\frac{b^2}{s^2}\right)^n",
            r"\text{Let } u = \frac{b^2}{s^2}: \quad \sum_{n \geq 0} (2n+1)(-u)^n = \frac{1-u}{(1+u)^2}",
            r"= \frac{1}{s^2} \cdot \frac{1 - \frac{b^2}{s^2}}{\left(1 + \frac{b^2}{s^2}\right)^2} = \frac{s^2 - b^2}{(s^2 + b^2)^2}",
        ],
        "closed_form": r"T\{t\cos(bt)\}(s) = \frac{s^2 - b^2}{(s^2 + b^2)^2}",
        "condition": r"s > |b|",
    }

    # Case H: cosh and sinh
    cases["Case H: cosh(bt) and sinh(bt)"] = {
        "title": "Case H: cosh(bt) and sinh(bt)",
        "function": r"f(t) = \cosh(bt) \text{ or } \sinh(bt)",
        "series": r"""
        \cosh(bt) = \sum_{n \geq 0} \frac{b^{2n}}{(2n)!} t^{2n}, \quad
        \sinh(bt) = \sum_{n \geq 0} \frac{b^{2n+1}}{(2n+1)!} t^{2n+1}
        """,
        "coefficients": r"a_{2n} = \frac{b^{2n}}{(2n)!}, \quad a_{2n+1} = \frac{b^{2n+1}}{(2n+1)!}",
        "rank": r"D^{2n} \text{ (even)}, \quad D^{2n+1} \text{ (odd)}",
        "plant_sum": [
            r"T\{\cosh(bt)\}(s) = \frac{1}{s} \sum_{n \geq 0} \left(\frac{b^2}{s^2}\right)^n = \frac{1}{s} \cdot \frac{1}{1 - \frac{b^2}{s^2}} = \frac{s}{s^2 - b^2}",
            r"T\{\sinh(bt)\}(s) = \frac{b}{s^2} \sum_{n \geq 0} \left(\frac{b^2}{s^2}\right)^n = \frac{b}{s^2} \cdot \frac{1}{1 - \frac{b^2}{s^2}} = \frac{b}{s^2 - b^2}",
        ],
        "closed_form": r"""
        T\{\cosh(bt)\}(s) = \frac{s}{s^2 - b^2}, \quad
        T\{\sinh(bt)\}(s) = \frac{b}{s^2 - b^2}
        """,
        "condition": r"\left|\frac{b}{s}\right| < 1",
    }

    # Shifted trig/hyperbolic
    cases["Shifted: e^{at}cos(bt), e^{at}sin(bt), e^{at}cosh(bt), e^{at}sinh(bt)"] = {
        "title": "Shifted Cases: e^{at}cos(bt), e^{at}sin(bt), e^{at}cosh(bt), e^{at}sinh(bt)",
        "function": r"f(t) = e^{at}\cos(bt), \; e^{at}\sin(bt), \; e^{at}\cosh(bt), \; e^{at}\sinh(bt)",
        "series": r"\text{Use the shifted kernel } \frac{1}{s-a} \text{ with the same coefficients.}",
        "coefficients": r"\text{Coefficients remain unchanged; only the kernel is shifted.}",
        "rank": r"\text{Rank preserved under kernel shifting.}",
        "plant_sum": [
            r"T\{e^{at}\cos(bt)\}(s) = \frac{s-a}{(s-a)^2 + b^2}",
            r"T\{e^{at}\sin(bt)\}(s) = \frac{b}{(s-a)^2 + b^2}",
            r"T\{e^{at}\cosh(bt)\}(s) = \frac{s-a}{(s-a)^2 - b^2}",
            r"T\{e^{at}\sinh(bt)\}(s) = \frac{b}{(s-a)^2 - b^2}",
        ],
        "closed_form": r"""
        T\{e^{at}\cos(bt)\} = \frac{s-a}{(s-a)^2 + b^2}, \quad
        T\{e^{at}\sin(bt)\} = \frac{b}{(s-a)^2 + b^2}, \\
        T\{e^{at}\cosh(bt)\} = \frac{s-a}{(s-a)^2 - b^2}, \quad
        T\{e^{at}\sinh(bt)\} = \frac{b}{(s-a)^2 - b^2}
        """,
        "condition": r"\text{Appropriate convergence conditions apply.}",
    }

    # Case I: sinc(bt)
    cases["Case I: sinc(bt) = sin(bt)/(bt)"] = {
        "title": "Case I: sinc(bt) = sin(bt)/(bt)",
        "function": r"f(t) = \mathrm{sinc}(bt) = \frac{\sin(bt)}{bt}",
        "series": r"\mathrm{sinc}(bt) = \sum_{n \geq 0} \frac{(-1)^n b^{2n}}{(2n+1)!} t^{2n}",
        "coefficients": r"a_{2n} = \frac{(-1)^n b^{2n}}{(2n+1)!}",
        "rank": r"D^{2n}",
        "plant_sum": [
            r"T\{\mathrm{sinc}(bt)\}(s) = \sum_{n \geq 0} a_{2n} (-1)^{2n} D^{2n}\left(\frac{1}{s}\right)",
            r"= \sum_{n \geq 0} \frac{(-1)^n b^{2n}}{(2n+1)!} \cdot \frac{(2n)!}{s^{2n+1}}",
            r"= \frac{1}{s} \sum_{n \geq 0} \frac{(-1)^n}{2n+1} \left(\frac{b}{s}\right)^{2n}",
            r"\text{Using } \sum_{n \geq 0} \frac{(-1)^n x^{2n+1}}{2n+1} = \arctan(x), \quad x = \frac{b}{s}",
            r"= \frac{1}{b} \arctan\left(\frac{b}{s}\right)",
        ],
        "closed_form": r"T\{\mathrm{sinc}(bt)\}(s) = \frac{1}{b} \arctan\left(\frac{b}{s}\right)",
        "condition": r"\left|\frac{b}{s}\right| < 1",
    }

    # Case J1: (cos(bt)-1)/t
    cases["Case J1: (cos(bt)-1)/t"] = {
        "title": "Case J1: (cos(bt)-1)/t",
        "function": r"f(t) = \frac{\cos(bt) - 1}{t}",
        "series": r"\frac{\cos(bt) - 1}{t} = \sum_{n \geq 1} \frac{(-1)^n b^{2n}}{(2n)!} t^{2n-1}",
        "coefficients": r"a_{2n-1} = \frac{(-1)^n b^{2n}}{(2n)!}",
        "rank": r"D^{2n-1}",
        "plant_sum": [
            r"T\left\{\frac{\cos(bt)-1}{t}\right\}(s) = \sum_{n \geq 1} a_{2n-1} (-1)^{2n-1} D^{2n-1}\left(\frac{1}{s}\right)",
            r"= \sum_{n \geq 1} \frac{(-1)^n b^{2n}}{(2n)!} \cdot \frac{(2n-1)!}{s^{2n}}",
            r"= \sum_{n \geq 1} \frac{(-1)^{n+1} b^{2n}}{2n} \cdot \frac{1}{s^{2n}}",
            r"= \frac{1}{2} \sum_{n \geq 1} \frac{1}{n} \left(-\frac{b^2}{s^2}\right)^n",
            r"\text{Using } \sum_{n \geq 1} \frac{r^n}{n} = -\ln(1-r), \quad |r| < 1",
            r"= -\frac{1}{2} \ln\left(1 + \frac{b^2}{s^2}\right)",
        ],
        "closed_form": r"T\left\{\frac{\cos(bt)-1}{t}\right\}(s) = -\frac{1}{2} \ln\left(1 + \frac{b^2}{s^2}\right)",
        "condition": r"s > |b|",
    }

    # Case J2: (cos(bt)-1)/t^2
    cases["Case J2: (cos(bt)-1)/t^2"] = {
        "title": "Case J2: (cos(bt)-1)/t^2",
        "function": r"f(t) = \frac{\cos(bt) - 1}{t^2}",
        "series": r"\frac{\cos(bt)-1}{t^2} = \sum_{n \geq 1} \frac{(-1)^n b^{2n}}{(2n)!} t^{2n-2}",
        "coefficients": r"a_{2n-2} = \frac{(-1)^n b^{2n}}{(2n)!}",
        "rank": r"D^{2n-2}",
        "plant_sum": [
            r"T\left\{\frac{\cos(bt)-1}{t^2}\right\}(s) = \sum_{n \geq 1} a_{2n-2} (-1)^{2n-2} D^{2n-2}\left(\frac{1}{s}\right)",
            r"= \sum_{n \geq 1} \frac{(-1)^n b^{2n}}{(2n)!} \cdot \frac{(2n-2)!}{s^{2n-1}}",
            r"= \sum_{n \geq 1} \frac{(-1)^n b^{2n}}{2n(2n-1)} \cdot \frac{1}{s^{2n-1}}",
            r"= s \sum_{n \geq 1} \frac{(-1)^n}{2n(2n-1)} \left(\frac{b^2}{s^2}\right)^n",
            r"\text{Using partial fractions and arctan series:}",
            r"= \frac{s}{2} \ln\left(1 + \frac{b^2}{s^2}\right) - b \arctan\left(\frac{b}{s}\right)",
        ],
        "closed_form": r"T\left\{\frac{\cos(bt)-1}{t^2}\right\}(s) = \frac{s}{2} \ln\left(1 + \frac{b^2}{s^2}\right) - b \arctan\left(\frac{b}{s}\right)",
        "condition": r"s > |b|",
    }

    # Case J3: Bessel J_0(kt)
    cases["Case J3: Bessel J_0(kt)"] = {
        "title": "Case J3: Bessel Function J_0(kt)",
        "function": r"f(t) = J_0(kt)",
        "series": r"J_0(kt) = \sum_{n=0}^{\infty} \frac{(-1)^n}{(n!)^2} \left(\frac{kt}{2}\right)^{2n}",
        "coefficients": r"a_{2n} = \frac{(-1)^n}{(n!)^2} \left(\frac{k}{2}\right)^{2n}",
        "rank": r"D^{2n}",
        "plant_sum": [
            r"T\{J_0(kt)\}(s) = \sum_{n=0}^{\infty} a_{2n} (-1)^{2n} D^{2n}\left(\frac{1}{s}\right)",
            r"= \sum_{n=0}^{\infty} \frac{(-1)^n}{(n!)^2} \left(\frac{k}{2}\right)^{2n} \cdot \frac{(2n)!}{s^{2n+1}}",
            r"= \frac{1}{s} \sum_{n=0}^{\infty} \frac{(2n)!}{(n!)^2} \left(-\frac{k^2}{4s^2}\right)^n",
            r"\text{Let } z = -\frac{k^2}{4s^2}.",
            r"\text{Using } \sum_{n=0}^{\infty} \frac{(2n)!}{(n!)^2} z^n = \frac{1}{\sqrt{1-4z}},",
            r"1 - 4z = 1 + \frac{k^2}{s^2} = \frac{s^2 + k^2}{s^2}",
            r"\sqrt{1-4z} = \frac{\sqrt{s^2 + k^2}}{s}",
            r"T\{J_0(kt)\}(s) = \frac{1}{s} \cdot \frac{s}{\sqrt{s^2 + k^2}} = \frac{1}{\sqrt{s^2 + k^2}}",
        ],
        "closed_form": r"T\{J_0(kt)\}(s) = \frac{1}{\sqrt{s^2 + k^2}}",
        "condition": r"s > 0",
    }

    # Case J3: Bessel J_1(kt)
    cases["Case J3: Bessel J_1(kt)"] = {
        "title": "Case J3: Bessel Function J_1(kt)",
        "function": r"f(t) = J_1(kt)",
        "series": r"J_1(kt) = -\frac{1}{k}\frac{d}{dt}J_0(kt) \text{ (derivative identity)}",
        "coefficients": r"\text{Derived from } J_0(kt) \text{ via differentiation.}",
        "rank": r"\text{Derivative relation}",
        "plant_sum": [
            r"J_1(kt) = -\frac{1}{k} \frac{d}{dt}J_0(kt),",
            r"T\{J_1(kt)\}(s) = -\frac{1}{k} T\left\{\frac{d}{dt}J_0(kt)\right\}(s)",
            r"\text{Using derivative rule: } T\{f'(t)\}(s) = sT\{f(t)\}(s) - f(0)",
            r"J_0(0) = 1, \quad T\{J_0(kt)\}(s) = \frac{1}{\sqrt{s^2 + k^2}}",
            r"T\{J_1(kt)\}(s) = -\frac{1}{k} \left[ s \cdot \frac{1}{\sqrt{s^2 + k^2}} - 1 \right]",
            r"= \frac{1}{k} \left(1 - \frac{s}{\sqrt{s^2 + k^2}}\right)",
            r"= \frac{\sqrt{s^2 + k^2} - s}{k\sqrt{s^2 + k^2}}",
        ],
        "closed_form": r"T\{J_1(kt)\}(s) = \frac{\sqrt{s^2 + k^2} - s}{k\sqrt{s^2 + k^2}}",
        "condition": r"s > 0",
    }

    # Case J3: Bessel J_nu(kt) general
    cases["Case J3: Bessel J_ν(kt) General"] = {
        "title": "Case J3: Bessel Function J_ν(kt) (General Order)",
        "function": r"f(t) = J_\nu(kt)",
        "series": r"J_\nu(kt) = \sum_{n=0}^{\infty} \frac{(-1)^n}{n!\,\Gamma(n+\nu+1)} \left(\frac{kt}{2}\right)^{2n+\nu}",
        "coefficients": r"a_{2n+\nu} = \frac{(-1)^n}{n!\,\Gamma(n+\nu+1)} \left(\frac{k}{2}\right)^{2n+\nu}",
        "rank": r"D^{2n+\nu}",
        "plant_sum": [
            r"T\{J_\nu(kt)\}(s) = \sum_{n=0}^{\infty} a_{2n+\nu} (-1)^{2n+\nu} D^{2n+\nu}\left(\frac{1}{s}\right)",
            r"= \left(\frac{k}{2}\right)^\nu \frac{1}{s^{\nu+1}} \sum_{n=0}^{\infty} \frac{(-1)^n \Gamma(2n+\nu+1)}{n!\,\Gamma(n+\nu+1)} \left(\frac{k^2}{4s^2}\right)^n",
            r"\text{After summation, the planted series reproduces the classical Laplace–Bessel closed form.}",
        ],
        "closed_form": r"T\{J_\nu(kt)\}(s) = \frac{\left(\sqrt{s^2+k^2}-s\right)^\nu}{k^\nu\sqrt{s^2+k^2}}",
        "condition": r"s > 0,\; \nu > -1",
    }

    return cases

# ============================================================
# Section 2.6: Integrals derived from the series
# ============================================================
def render_integrals():
    st.header("2.6 Integrals derived from the series")

    st.markdown(r"""
    From Case I (sinc function), we have the integral representation
    """)
    st.latex(r"""
    \int_0^\infty e^{-st} \frac{\sin(bt)}{bt} \, dt = \mathcal{T}\{\mathrm{sinc}(bt)\}(s) = \frac{1}{b} \tan^{-1}\left(\frac{b}{s}\right).
    """)

    st.markdown(r"**Example 1:** Setting $s = 1$, $b = 1$,")
    st.latex(r"""
    \int_0^\infty e^{-t} \frac{\sin t}{t} \, dt = \tan^{-1}(1) = \frac{\pi}{4}.
    """)

    st.markdown(r"**Example 2:** Setting $s = 0$, $b = 1$ (taking the limit),")
    st.latex(r"""
    \int_0^\infty \frac{\sin t}{t} \, dt = \frac{\pi}{2}.
    """)

    st.markdown(r"""
    From Case J1:
    """)
    st.latex(r"""
    \int_0^\infty e^{-st} \frac{\cos(bt) - 1}{t} \, dt = -\frac{1}{2} \ln\left(1 + \frac{b^2}{s^2}\right), \quad s > |b|.
    """)

    st.markdown(r"""
    From Case J2:
    """)
    st.latex(r"""
    \int_0^\infty e^{-st} \frac{\cos(bt) - 1}{t^2} \, dt = \frac{s}{2} \ln\left(1 + \frac{b^2}{s^2}\right) - b \arctan\left(\frac{b}{s}\right), \quad s > |b|.
    """)

# ============================================================
# Table 1: Summary of Results
# ============================================================
def render_table():
    st.header("Table 1: Summary of Operator Laplace Transform Results")

    st.markdown(r"""
    | Function $f(t)$ | Planted series (pure operator) | Closed form |
    |---|---|---|
    | $1$ | $T\{1\}(s) = D^0\left(\frac{1}{s}\right)$ | $\frac{1}{s}$ |
    | $t^n$ | $T\{t^n\}(s) = (-1)^n D^n\left(\frac{1}{s}\right)$ | $\frac{n!}{s^{n+1}}$ |
    | $e^{at}$ | $T\{e^{at}\}(s) = \sum_{n \geq 0} \frac{a^n}{n!}(-1)^n D^n\left(\frac{1}{s}\right)$ | $\frac{1}{s-a}$ |
    | $\cos(bt)$ | $T\{\cos(bt)\}(s) = \sum_{n \geq 0} \frac{(-1)^n b^{2n}}{(2n)!}(-1)^{2n} D^{2n}\left(\frac{1}{s}\right)$ | $\frac{s}{s^2+b^2}$ |
    | $\sin(bt)$ | $T\{\sin(bt)\}(s) = \sum_{n \geq 0} \frac{(-1)^n b^{2n+1}}{(2n+1)!}(-1)^{2n+1} D^{2n+1}\left(\frac{1}{s}\right)$ | $\frac{b}{s^2+b^2}$ |
    | $t\sin(bt)$ | $T\{t\sin(bt)\}(s) = \sum_{n \geq 0} \frac{(-1)^n b^{2n+1}}{(2n+1)!}(-1)^{2n+2} D^{2n+2}\left(\frac{1}{s}\right)$ | $\frac{2bs}{(s^2+b^2)^2}$ |
    | $t\cos(bt)$ | $T\{t\cos(bt)\}(s) = \sum_{n \geq 0} \frac{(-1)^n b^{2n}}{(2n)!}(-1)^{2n+1} D^{2n+1}\left(\frac{1}{s}\right)$ | $\frac{s^2-b^2}{(s^2+b^2)^2}$ |
    | $\cosh(bt)$ | $T\{\cosh(bt)\}(s) = \sum_{n \geq 0} \frac{b^{2n}}{(2n)!}(-1)^{2n} D^{2n}\left(\frac{1}{s}\right)$ | $\frac{s}{s^2-b^2}$ |
    | $\sinh(bt)$ | $T\{\sinh(bt)\}(s) = \sum_{n \geq 0} \frac{b^{2n+1}}{(2n+1)!}(-1)^{2n+1} D^{2n+1}\left(\frac{1}{s}\right)$ | $\frac{b}{s^2-b^2}$ |
    | $\mathrm{sinc}(bt)$ | $T\{\mathrm{sinc}(bt)\}(s) = \sum_{n \geq 0} \frac{(-1)^n b^{2n}}{(2n+1)!}(-1)^{2n} D^{2n}\left(\frac{1}{s}\right)$ | $\frac{1}{b}\arctan\left(\frac{b}{s}\right)$ |
    | $\frac{\cos(bt)-1}{t}$ | $T\left\{\frac{\cos(bt)-1}{t}\right\}(s) = \sum_{n \geq 1} \frac{(-1)^n b^{2n}}{(2n)!}(-1)^{2n-1} D^{2n-1}\left(\frac{1}{s}\right)$ | $-\frac{1}{2}\ln\left(1+\frac{b^2}{s^2}\right)$ |
    | $\frac{\cos(bt)-1}{t^2}$ | $T\left\{\frac{\cos(bt)-1}{t^2}\right\}(s) = \sum_{n \geq 1} \frac{(-1)^n b^{2n}}{(2n)!}(-1)^{2n-2} D^{2n-2}\left(\frac{1}{s}\right)$ | $\frac{s}{2}\ln\left(1+\frac{b^2}{s^2}\right)-b\arctan\left(\frac{b}{s}\right)$ |
    | $J_0(kt)$ | $T\{J_0(kt)\}(s) = \sum_{n=0}^{\infty} \frac{(-1)^n}{(n!)^2}\left(\frac{k}{2}\right)^{2n}(-1)^{2n} D^{2n}\left(\frac{1}{s}\right)$ | $\frac{1}{\sqrt{s^2+k^2}}$ |
    | $J_1(kt)$ | $T\{J_1(kt)\}(s) = -\frac{1}{k} T\left\{\frac{d}{dt}J_0(kt)\right\}(s)$ | $\frac{\sqrt{s^2+k^2}-s}{k\sqrt{s^2+k^2}}$ |
    | $J_\nu(kt)$ | $T\{J_\nu(kt)\}(s) = \sum_{n=0}^{\infty} \frac{(-1)^n}{n!\Gamma(n+\nu+1)}\left(\frac{k}{2}\right)^{2n+\nu}(-1)^{2n+\nu} D^{2n+\nu}\left(\frac{1}{s}\right)$ | $\frac{(\sqrt{s^2+k^2}-s)^\nu}{k^\nu\sqrt{s^2+k^2}}$ |
    """)

# ============================================================
# Main Section: Operator Laplace Transform
# ============================================================
def render_laplace_section():
    st.title("Part I: An Operator-Based Laplace Transform")
    st.subheader("The Birth of Kernel Geometry")

    st.markdown(r"""
    This part presents a new operator-based formulation of the Laplace transform that 
    originates purely from the Maclaurin series representation of a function, without 
    assuming an integral definition *a priori*. The central idea is to embed the series 
    coefficients into a sequence of differential operators acting on a simple rational 
    kernel, allowing the Laplace transform to be reconstructed step by step from an 
    internal differential structure. We show that this operator-based construction 
    reproduces the classical Laplace transform on an appropriate class of functions. 
    Moreover, the proposed framework provides a structural interpretation that links 
    the order of differentiation to the behavior of the underlying kernel. Several 
    consequences and illustrative applications are derived, including trigonometric 
    functions, special functions, and linear differential equations, demonstrating 
    the effectiveness and flexibility of the proposed approach.
    """)

    st.markdown(r"""
    We denote by $\mathcal{L}\{f\}(s)$ the classical Laplace transform
    """)
    st.latex(r"\mathcal{L}\{f\}(s) = \int_0^\infty e^{-st}f(t) \, dt, \quad s > 0,")
    st.markdown(r"whenever the integral exists.")

    # Render all subsections
    render_operator_reconstruction()
    st.divider()

    render_rank_bookkeeping()
    st.divider()

    render_kernel_shifting()
    st.divider()

    render_shifted_kernels()
    st.divider()

    # Case Studies
    st.header("Case Studies")

    st.info(r"""
    **Remark 3.** The constants $a, b, k,$ and similar parameters appearing in the examples 
    (such as $e^{at}, \cos(bt),$ and $J_0(kt)$) arise naturally from the Maclaurin expansion 
    of the input function. Within the present operator-based framework, these constants act 
    as inherited scaling factors that govern the growth or oscillatory behavior of the 
    function and, consequently, determine the effective rank and convergence behavior of 
    the associated operator series.
    """)

    cases = get_laplace_cases_detailed()

    selected_case = st.selectbox(
        "Choose a symbolic Laplace case",
        list(cases.keys()),
        index=0,
        key="laplace_detailed_case",
    )

    case = cases[selected_case]

    st.subheader(case["title"])

    st.markdown("**Function**")
    st.latex(case["function"])

    st.markdown("**Series**")
    st.latex(case["series"])

    st.markdown("**Coefficients**")
    st.latex(case["coefficients"])

    st.markdown("**Rank**")
    st.latex(case["rank"])

    st.markdown("**Plant & Sum**")
    for step in case["plant_sum"]:
        st.latex(step)

    st.markdown("**Closed Form**")
    st.latex(case["closed_form"])

    st.markdown("**Validity / Convergence Condition**")
    st.latex(case["condition"])

    st.divider()

    # Integrals derived from series
    render_integrals()
    st.divider()

    # Table 1
    render_table()

# ============================================================
# Run the app
# ============================================================
if __name__ == "__main__":
    render_laplace_section()
