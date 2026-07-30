import math
import numpy as np
import sympy as sp
import streamlit as st


st.set_page_config(
    page_title="Operator Transform Laboratory",
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
    h1 {
        font-size: 2.2rem !important;
    }
    h2 {
        font-size: 1.8rem !important;
    }
    h3 {
        font-size: 1.4rem !important;
    }
    h4 {
        font-size: 1.2rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Symbols
# ============================================================
s_sym = sp.Symbol("s", positive=True, real=True)
t_sym = sp.Symbol("t", positive=True, real=True)
a_sym = sp.Symbol("a", real=True)
b_sym = sp.Symbol("b", real=True)
k_sym = sp.Symbol("k", positive=True, real=True)
nu_sym = sp.Symbol("nu", real=True)
n_sym = sp.Symbol("n", integer=True, nonnegative=True)

# ============================================================
# GENERAL INTRODUCTION
# ============================================================
st.title("General Introduction")

st.markdown("""
Classical integral transforms are traditionally studied as separate analytical tools, each equipped with its own integral definition, kernel structure, and domain of applicability. The Laplace, Fourier, Mellin, and others transforms are typically introduced independently, with little emphasis on a common structural origin linking these transformations within a unified analytical framework. In this monograph, we present a unified formulation of the classical Laplace, Fourier, Mellin, and Hankel transforms based on a novel operator-based approach. The proposed methodology does not assume any integral definition *a priori*. Instead, it originates from the Maclaurin series representation of the function to be transformed, serving as the foundational analytic structure from which all transform representations are derived. The central idea of the framework is to embed the coefficients of the Maclaurin series into a sequence of differential operators acting on a simple fractional kernel. This kernel is chosen to be common across all transforms and is given in the form $1/s$, where the parameter $s$ is encoded according to the specific transform under consideration in order to recover the corresponding classical results. In this setting, the notion of *rank* emerges naturally, with the rank governing both the type of transform and the resulting transformation behavior of the function. Within this operator-based framework, the classical integral transforms arise as distinct realizations of a single repeated-differentiation mechanism, distinguished only by their rank structure and kernel encoding. This viewpoint allows the Laplace, Fourier, Mellin, and Hankel transforms to be treated within a coherent and unified operator geometry, while preserving their classical forms and properties. The monograph is organized into four main parts. Each part is devoted to a specific transform and is equipped with its own title and introductory section, where the detailed construction and properties of the corresponding operator transform are developed. The exposition begins with the operator-based Laplace transform, followed by the Fourier transform, the Mellin transform, and finally the Hankel transform.
""")

st.divider()

# ============================================================
# PART I
# ============================================================
st.header("Part I: An Operator-Based Laplace Transform: The Birth of Kernel Geometry")

st.markdown("""
This part presents a new operator-based formulation of the Laplace transform that originates purely from the Maclaurin series representation of a function, without assuming an integral definition a priori. The central idea is to embed the series coefficients into a sequence of differential operators acting on a simple rational kernel, allowing the Laplace transform to be reconstructed step by step from an internal differential structure. We show that this operator-based construction reproduces the classical Laplace transform on an appropriate class of functions. Moreover, the proposed framework provides a structural interpretation that links the order of differentiation to the behavior of the underlying kernel. Several consequences and illustrative applications are derived, including trigonometric functions, special functions, and linear differential equations, demonstrating the effectiveness and flexibility of the proposed approach.
""")

st.markdown("""
We denote by $\\mathcal{L}\\{f\\}(s)$ the classical Laplace transform
""")
st.latex(r"\mathcal{L}\{f\}(s) = \int_0^\infty e^{-st}f(t) \, dt, \quad s > 0,")
st.markdown("whenever the integral exists.")

# ============================================================
# SECTION 2.1
# ============================================================
st.header("2.1 Operator-based reconstruction of the Laplace transform")

st.markdown("**Definition 2.1.** Let $\\mathcal{A}$ denote the class of functions $f$ that are analytic in a neighborhood of $t = 0$ and admit a Maclaurin expansion")
st.latex(r"f(t) = \sum_{n=0}^\infty a_n t^n.")
st.markdown("""
The operator-based transform $T$ of $f$ is defined by
""")
st.latex(r"T\{f\}(s) := \sum_{n=0}^\infty a_n (-D_s)^n \left( \frac{1}{s} \right),")
st.markdown("whenever the above operator series is convergent.")

st.markdown("**Theorem 2.2** (Equivalence with the classical Laplace transform). For every function $f \\in \\mathcal{A}$ and for every $s$ such that the operator series defining $T\\{f\\}(s)$ is convergent, one has")
st.latex(r"T\{f\}(s) = \int_0^\infty e^{-st}f(t) \, dt,")
st.markdown("whenever the integral exists.")

st.markdown("**Proof.** Starting from the Maclaurin expansion")
st.latex(r"f(t) = \sum_{n=0}^\infty a_n t^n,")
st.markdown("""
the differential differentiation mechanism acts on the rational kernel $1/s$ via repeated differentiation, yielding the fundamental identity
""")
st.latex(r"D_s^n \left( \frac{1}{s} \right) = (-1)^n \frac{n!}{s^{n+1}}.")
st.markdown("Substituting this expression into the operator series gives")
st.latex(r"T\{f\}(s) = \sum_{n=0}^\infty a_n \frac{n!}{s^{n+1}}.")
st.markdown("""
To reveal the integral structure hidden in this purely differential expression, we invoke the Gamma-function identity
""")
st.latex(r"\frac{1}{s^{n+1}} = \frac{1}{n!} \int_0^\infty t^n e^{-st} \, dt, \quad s > 0.")
st.markdown("""
Inserting this representation and interchanging summation and integration (justified by absolute convergence) yields
""")
st.latex(r"""
T\{f\}(s) = \int_0^\infty e^{-st} \left( \sum_{n=0}^\infty a_n t^n \right) dt = \int_0^\infty e^{-st} f(t) dt.
""")
st.markdown("This is precisely the classical Laplace transform of $f$, completing the proof.")

st.markdown("""
**Remark 1** (Convergence domain). The definition of $T$ does not impose an explicit a priori condition on $s$. The admissible values of $s$ are determined implicitly by the convergence of the operator series. In concrete examples, this convergence domain can be computed explicitly and coincides with the classical region of convergence of the Laplace transform (e.g. trigonometric, exponential, and Bessel-type functions).
""")

st.markdown("""
**Remark 2** (Emergent integral structure). Within the present framework, the Laplace transform is not postulated a priori as an integral operator. Rather, it emerges naturally from the mechanism of repeated differentiation acting on the base kernel $1/s$. The exponential kernel $e^{-st}$ appears only after invoking the Gamma-function identity, revealing the integral representation as a consequence of an underlying purely differential structure. This does not invalidate the classical integral definition, but rather explains its structural origin.
""")

st.divider()

# ============================================================
# SECTION 2.2
# ============================================================
st.header("2.2 Rank bookkeeping used in the case studies")

st.markdown("""
The notion of rank introduced here serves purely as a bookkeeping device.  
Thus, the derivative rank is directly inherited from the power of the variable in the Maclaurin expansion of the input function.  
We use the differential operator
""")
st.latex(r"D := \frac{d}{ds},")
st.markdown("acting on the base kernel $1/s$. In the Maclaurin-differentiation construction, each power $t^n$ selects a derivative rank $D^n$ to be applied to $1/s$. This notion of rank is the only bookkeeping device required in the applications below.")

st.markdown("**Basic kernel identity.** For every $n \\in \\mathbb{N}_0$ and $s > 0$,")
st.latex(r"""
D^n \left( \frac{1}{s} \right) = (-1)^n \frac{n!}{s^{n+1}}, \quad \text{hence} \quad (-D)^n \left( \frac{1}{s} \right) = \frac{n!}{s^{n+1}}.
""")

st.markdown("**Rank parity (sign absorption).** The factor $(-1)^n$ arising from $D^n(1/s)$ is absorbed by writing the generated operator as $(-D)^n$. This convention keeps all subsequent computations sign-clean.")

st.markdown("**Rank lowering by multiplication with $s$.** For $n \\geq 1$,")
st.latex(r"s D^n \left( \frac{1}{s} \right) = -n D^{n-1} \left( \frac{1}{s} \right).")
st.markdown("More generally, for $k \\leq n$,")
st.latex(r"""
s^k D^n \left( \frac{1}{s} \right) = (-1)^k n^{\underline{k}} D^{n-k} \left( \frac{1}{s} \right), \quad n^{\underline{k}} := n(n-1)\cdots(n-k+1).
""")

st.markdown("**Rank shift induced by division by $t^m$.** If")
st.latex(r"""
f(t) = \sum_{n=0}^\infty c_n t^n, \quad \frac{f(t)}{t^m} = \sum_{n=m}^\infty c_n t^{n-m},
""")
st.markdown("then the generated rank is lowered accordingly, and the operator-based transform satisfies")
st.latex(r"""
T\left(\frac{f(t)}{t^m}\right)(s) = \sum_{n=m}^\infty c_n (-D)^{n-m} \left(\frac{1}{s}\right),
""")
st.markdown("whenever the resulting series is admissible.")

st.divider()

# ============================================================
# SECTION 2.3
# ============================================================
st.header("2.3 Kernel shifting and rank preservation")

st.markdown("""
A key structural feature of the generated-operator framework is the kernel shift
""")
st.latex(r"\frac{1}{s} \to \frac{1}{s-a},")
st.markdown("which encodes exponential tilting in the $t$-domain.")

st.markdown("**Shifted operator definition.** Given the Maclaurin expansion")
st.latex(r"f(t) = \sum_{n=0}^{\infty} a_n t^n,")
st.markdown("we define the shifted generated transform by")
st.latex(r"""
T_a\{f\}(s) := \sum_{n=0}^{\infty} a_n (-D)^n \left(\frac{1}{s-a}\right), \quad D = \frac{d}{ds}.
""")

st.markdown("**Shifted kernel identity.** For every $n \\in \\mathbb{N}_0$ and $s > a$,")
st.latex(r"""
D^n \left(\frac{1}{s-a}\right) = (-1)^n \frac{n!}{(s-a)^{n+1}}, \quad \text{hence} \quad (-D)^n \left(\frac{1}{s-a}\right) = \frac{n!}{(s-a)^{n+1}}.
""")

st.markdown("**Rank invariance under shifting.** The shift $s \\mapsto s-a$ modifies only the kernel location and does not alter the derivative rank selected by each Maclaurin coefficient. Thus, rank bookkeeping is preserved under exponential tilting.")

st.markdown("**Interpretation.** The shift $s \\mapsto s-a$ corresponds to multiplication by $e^{at}$ in the $t$-domain, exactly mirroring the classical Laplace shift property, but derived here purely at the operator level.")

st.divider()

# ============================================================
# SECTION 2.4
# ============================================================
st.header("2.4 Shifted kernels and repeated poles")

st.markdown("**Simple shift.**")
st.latex(r"""
\frac{1}{s^2-1} = \frac{1}{2} \left(\frac{1}{s-1} - \frac{1}{s+1}\right) \implies D^n\left(\frac{1}{s^2-1}\right) = \frac{1}{2} D^n\left(\frac{1}{s-1}\right) - \frac{1}{2} D^n\left(\frac{1}{s+1}\right).
""")

st.markdown("**Repeated poles.** For integers $m \\geq 1$ and $n \\geq 0$,")
st.latex(r"""
D^n\left(\frac{1}{(s-a)^m}\right) = (-1)^n \frac{\Gamma(m+n)}{\Gamma(m)} (s-a)^{-(m+n)}.
""")

st.divider()

# ============================================================
# CASE STUDIES - INTERACTIVE
# ============================================================
st.header("Case Studies")

st.markdown("""
**Remark 3.** The constants $a, b, k,$ and similar parameters appearing in the examples (such as $e^{at}, \cos(bt),$ and $J_0(kt)$) arise naturally from the Maclaurin expansion of the input function. Within the present operator-based framework, these constants act as inherited scaling factors that govern the growth or oscillatory behavior of the function and, consequently, determine the effective rank and convergence behavior of the associated operator series.
""")

# ============================================================
# CASE DATA (with full details from the original long code)
# ============================================================
def get_laplace_cases():
    cases = {}

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
        "params": ["s"],
        "compute_result": lambda s_val, **kwargs: 1.0 / s_val if s_val > 0 else None,
    }

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
        "params": ["s", "n"],
        "compute_result": lambda s_val, n_val=0, **kwargs: math.factorial(int(n_val)) / (s_val ** (int(n_val) + 1)) if s_val > 0 else None,
    }

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
        "params": ["s", "a"],
        "compute_result": lambda s_val, a_val=0, **kwargs: 1.0 / (s_val - a_val) if s_val > abs(a_val) else None,
    }

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
        "params": ["s", "b"],
        "compute_result": lambda s_val, b_val=0, **kwargs: s_val / (s_val**2 + b_val**2) if s_val > abs(b_val) else None,
    }

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
        "params": ["s", "b"],
        "compute_result": lambda s_val, b_val=0, **kwargs: b_val / (s_val**2 + b_val**2) if s_val > abs(b_val) else None,
    }

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
        "params": ["s", "b"],
        "compute_result": lambda s_val, b_val=0, **kwargs: (2 * b_val * s_val) / ((s_val**2 + b_val**2)**2) if s_val > abs(b_val) else None,
    }

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
        "params": ["s", "b"],
        "compute_result": lambda s_val, b_val=0, **kwargs: (s_val**2 - b_val**2) / ((s_val**2 + b_val**2)**2) if s_val > abs(b_val) else None,
    }

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
        "params": ["s", "b"],
        "compute_result": lambda s_val, b_val=0, **kwargs: s_val / (s_val**2 - b_val**2) if s_val > abs(b_val) else None,
    }

    cases["Shifted Cases: e^{at}cos(bt), e^{at}sin(bt), e^{at}cosh(bt), e^{at}sinh(bt)"] = {
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
        "params": ["s", "a", "b"],
        "compute_result": lambda s_val, a_val=0, b_val=0, **kwargs: (s_val - a_val) / ((s_val - a_val)**2 + b_val**2) if s_val > abs(a_val) else None,
    }

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
        "params": ["s", "b"],
        "compute_result": lambda s_val, b_val=0, **kwargs: (1.0 / b_val) * math.atan(b_val / s_val) if s_val > abs(b_val) and b_val != 0 else None,
    }

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
        "params": ["s", "b"],
        "compute_result": lambda s_val, b_val=0, **kwargs: -0.5 * math.log(1 + (b_val / s_val)**2) if s_val > abs(b_val) else None,
    }

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
        "params": ["s", "b"],
        "compute_result": lambda s_val, b_val=0, **kwargs: (s_val / 2.0) * math.log(1 + (b_val / s_val)**2) - b_val * math.atan(b_val / s_val) if s_val > abs(b_val) else None,
    }

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
        "params": ["s", "k"],
        "compute_result": lambda s_val, k_val=1, **kwargs: 1.0 / math.sqrt(s_val**2 + k_val**2) if s_val > 0 else None,
    }

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
        "params": ["s", "k"],
        "compute_result": lambda s_val, k_val=1, **kwargs: (math.sqrt(s_val**2 + k_val**2) - s_val) / (k_val * math.sqrt(s_val**2 + k_val**2)) if s_val > 0 and k_val > 0 else None,
    }

    cases["Case J3: Bessel J_ν(kt) General"] = {
        "title": "Case J3: Bessel Function J_ν(kt) (General Order)",
        "function": r"f(t) = J_\nu(kt)",
        "series": r"J_\nu(kt) = \sum_{n=0}^{\infty} \frac{(-1)^n}{n!\,\Gamma(n+\nu+1)} \left(\frac{kt}{2}\right)^{2n+\nu}",
        "coefficients": r"a_{2n+\nu} = \frac{(-1)^n}{n!\,\Gamma(n+\nu+1)} \left(\frac{k}{2}\right)^{2n+\nu}",
        "rank": r"D^{2n+\nu}",
        "plant_sum": [
            r"T\{J_\nu(kt)\}(s) = \sum_{n=0}^{\infty} a_{2n+\nu} (-1)^{2n+\nu} D^{2n+\nu}\left(\frac{1}{s}\right)",
            r"= \left(\frac{k}{2}\right)^\nu \frac{1}{s^{\nu+1}} \sum_{n=0}^{\infty} \frac{(-1)^n \Gamma(2n+\nu+1)}{n!\,\Gamma(n+\nu+1)} \left(\frac{k^2}{4s^2}\right)^n",
            r"\text{After summation, the generated series reproduces the classical Laplace–Bessel closed form.}",
        ],
        "closed_form": r"T\{J_\nu(kt)\}(s) = \frac{(\sqrt{s^2+k^2}-s)^\nu}{k^\nu\sqrt{s^2+k^2}}",
        "condition": r"s > 0,\; \nu > -1",
        "params": ["s", "k", "nu"],
        "compute_result": lambda s_val, k_val=1, nu_val=0, **kwargs: ((math.sqrt(s_val**2 + k_val**2) - s_val)**nu_val) / ((k_val**nu_val) * math.sqrt(s_val**2 + k_val**2)) if s_val > 0 and k_val > 0 and nu_val > -1 else None,
    }

    return cases

# ============================================================
# INTERACTIVE SELECTION
# ============================================================
cases = get_laplace_cases()
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

# ============================================================
# PARAMETER INPUTS AND NUMERICAL EVALUATION
# ============================================================
st.markdown("---")
st.subheader("Numerical Evaluation")

params = case.get("params", [])

if "s" in params:
    s_val = st.number_input(
        "Enter value for $s$",
        value=1.0,
        step=0.1,
        format="%.2f",
        key=f"s_{selected_case[:10]}"
    )
else:
    s_val = None

if "a" in params:
    a_val = st.number_input(
        "Enter value for $a$",
        value=0.5,
        step=0.1,
        format="%.2f",
        key=f"a_{selected_case[:10]}"
    )
else:
    a_val = None

if "b" in params:
    b_val = st.number_input(
        "Enter value for $b$",
        value=1.0,
        step=0.1,
        format="%.2f",
        key=f"b_{selected_case[:10]}"
    )
else:
    b_val = None

if "k" in params:
    k_val = st.number_input(
        "Enter value for $k$",
        value=1.0,
        step=0.1,
        format="%.2f",
        key=f"k_{selected_case[:10]}"
    )
else:
    k_val = None

if "n" in params:
    n_val = st.number_input(
        "Enter value for $n$ (integer)",
        value=0,
        step=1,
        format="%d",
        key=f"n_{selected_case[:10]}"
    )
else:
    n_val = None

if "nu" in params:
    nu_val = st.number_input(
        "Enter value for $\\nu$",
        value=0.0,
        step=0.1,
        format="%.2f",
        key=f"nu_{selected_case[:10]}"
    )
else:
    nu_val = None

# Compute result
if case.get("compute_result"):
    try:
        result = case["compute_result"](
            s_val=s_val if s_val is not None else 1.0,
            a_val=a_val if a_val is not None else 0.0,
            b_val=b_val if b_val is not None else 1.0,
            k_val=k_val if k_val is not None else 1.0,
            n_val=n_val if n_val is not None else 0,
            nu_val=nu_val if nu_val is not None else 0.0,
        )
        if result is not None:
            st.success(f"**Numerical Result:** $T\\{{f\\}}({s_val if s_val else 1.0})$ = {result:.6f}")
        else:
            st.warning("The entered values do not satisfy the convergence condition.")
    except Exception as e:
        st.error(f"Error computing result: {e}")

st.divider()

# ============================================================
# TABLE 1
# ============================================================
st.header("Table 1: Summary of Operator Laplace Transform Results")

st.markdown("""
| Function $f(t)$ | Planted series (pure operator) | Closed form |
|---|---|---|
| $1$ | $T\\{1\\}(s) = D^0\\left(\\frac{1}{s}\\right)$ | $\\frac{1}{s}$ |
| $t^n$ | $T\\{t^n\\}(s) = (-1)^n D^n\\left(\\frac{1}{s}\\right)$ | $\\frac{n!}{s^{n+1}}$ |
| $e^{at}$ | $T\\{e^{at}\\}(s) = \\sum_{n \\geq 0} \\frac{a^n}{n!}(-1)^n D^n\\left(\\frac{1}{s}\\right)$ | $\\frac{1}{s-a}$ |
| $\\cos(bt)$ | $T\\{\\cos(bt)\\}(s) = \\sum_{n \\geq 0} \\frac{(-1)^n b^{2n}}{(2n)!}(-1)^{2n} D^{2n}\\left(\\frac{1}{s}\\right)$ | $\\frac{s}{s^2+b^2}$ |
| $\\sin(bt)$ | $T\\{\\sin(bt)\\}(s) = \\sum_{n \\geq 0} \\frac{(-1)^n b^{2n+1}}{(2n+1)!}(-1)^{2n+1} D^{2n+1}\\left(\\frac{1}{s}\\right)$ | $\\frac{b}{s^2+b^2}$ |
| $t\\sin(bt)$ | $T\\{t\\sin(bt)\\}(s) = \\sum_{n \\geq 0} \\frac{(-1)^n b^{2n+1}}{(2n+1)!}(-1)^{2n+2} D^{2n+2}\\left(\\frac{1}{s}\\right)$ | $\\frac{2bs}{(s^2+b^2)^2}$ |
| $t\\cos(bt)$ | $T\\{t\\cos(bt)\\}(s) = \\sum_{n \\geq 0} \\frac{(-1)^n b^{2n}}{(2n)!}(-1)^{2n+1} D^{2n+1}\\left(\\frac{1}{s}\\right)$ | $\\frac{s^2-b^2}{(s^2+b^2)^2}$ |
| $\\cosh(bt)$ | $T\\{\\cosh(bt)\\}(s) = \\sum_{n \\geq 0} \\frac{b^{2n}}{(2n)!}(-1)^{2n} D^{2n}\\left(\\frac{1}{s}\\right)$ | $\\frac{s}{s^2-b^2}$ |
| $\\sinh(bt)$ | $T\\{\\sinh(bt)\\}(s) = \\sum_{n \\geq 0} \\frac{b^{2n+1}}{(2n+1)!}(-1)^{2n+1} D^{2n+1}\\left(\\frac{1}{s}\\right)$ | $\\frac{b}{s^2-b^2}$ |
| $\\mathrm{sinc}(bt)$ | $T\\{\\mathrm{sinc}(bt)\\}(s) = \\sum_{n \\geq 0} \\frac{(-1)^n b^{2n}}{(2n+1)!}(-1)^{2n} D^{2n}\\left(\\frac{1}{s}\\right)$ | $\\frac{1}{b}\\arctan\\left(\\frac{b}{s}\\right)$ |
| $\\frac{\\cos(bt)-1}{t}$ | $T\\left\\{\\frac{\\cos(bt)-1}{t}\\right\\}(s) = \\sum_{n \\geq 1} \\frac{(-1)^n b^{2n}}{(2n)!}(-1)^{2n-1} D^{2n-1}\\left(\\frac{1}{s}\\right)$ | $-\\frac{1}{2}\\ln\\left(1+\\frac{b^2}{s^2}\\right)$ |
| $\\frac{\\cos(bt)-1}{t^2}$ | $T\\left\\{\\frac{\\cos(bt)-1}{t^2}\\right\\}(s) = \\sum_{n \\geq 1} \\frac{(-1)^n b^{2n}}{(2n)!}(-1)^{2n-2} D^{2n-2}\\left(\\frac{1}{s}\\right)$ | $\\frac{s}{2}\\ln\\left(1+\\frac{b^2}{s^2}\\right)-b\\arctan\\left(\\frac{b}{s}\\right)$ |
| $J_0(kt)$ | $T\\{J_0(kt)\\}(s) = \\sum_{n=0}^{\\infty} \\frac{(-1)^n}{(n!)^2}\\left(\\frac{k}{2}\\right)^{2n}(-1)^{2n} D^{2n}\\left(\\frac{1}{s}\\right)$ | $\\frac{1}{\\sqrt{s^2+k^2}}$ |
| $J_1(kt)$ | $T\\{J_1(kt)\\}(s) = -\\frac{1}{k} T\\left\\{\\frac{d}{dt}J_0(kt)\\right\\}(s)$ | $\\frac{\\sqrt{s^2+k^2}-s}{k\\sqrt{s^2+k^2}}$ |
| $J_\\nu(kt)$ | $T\\{J_\\nu(kt)\\}(s) = \\sum_{n=0}^{\\infty} \\frac{(-1)^n}{n!\\Gamma(n+\\nu+1)}\\left(\\frac{k}{2}\\right)^{2n+\\nu}(-1)^{2n+\\nu} D^{2n+\\nu}\\left(\\frac{1}{s}\\right)$ | $\\frac{(\\sqrt{s^2+k^2}-s)^\\nu}{k^\\nu\\sqrt{s^2+k^2}}$ |
""")
