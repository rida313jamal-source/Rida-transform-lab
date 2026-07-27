import math
import numpy as np
import sympy as sp
import streamlit as st
from scipy.integrate import quad

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
# GENERAL INTRODUCTION
# ============================================================
st.title("General Introduction")

st.markdown("""
Classical integral transforms are traditionally studied as separate analytical tools, each equipped with its own integral definition, kernel structure, and domain of applicability. The Laplace, Fourier, Mellin, and others transforms are typically introduced independently, with little emphasis on a common structural origin linking these transformations within a unified analytical framework. In this monograph, we present a unified formulation of the classical Laplace, Fourier, Mellin, and Hankel transforms based on a novel operator-based approach. The proposed methodology does not assume any integral definition *a priori*. Instead, it originates from the Maclaurin series representation of the function to be transformed, serving as the foundational analytic structure from which all transform representations are derived. The central idea of the framework is to embed the coefficients of the Maclaurin series into a sequence of differential operators acting on a simple fractional kernel. This kernel is chosen to be common across all transforms and is given in the form $1/s$, where the parameter $s$ is encoded according to the specific transform under consideration in order to recover the corresponding classical results. In this setting, the notion of *rank* emerges naturally, with the rank governing both the type of transform and the resulting transformation behavior of the function. Within this operator-based framework, the classical integral transforms arise as distinct realizations of a single planted differential mechanism, distinguished only by their rank structure and kernel encoding. This viewpoint allows the Laplace, Fourier, Mellin, and Hankel transforms to be treated within a coherent and unified operator geometry, while preserving their classical forms and properties. The monograph is organized into four main parts. Each part is devoted to a specific transform and is equipped with its own title and introductory section, where the detailed construction and properties of the corresponding operator transform are developed. The exposition begins with the operator-based Laplace transform, followed by the Fourier transform, the Mellin transform, and finally the Hankel transform.
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
the differential planting mechanism acts on the rational kernel $1/s$ via repeated differentiation, yielding the fundamental identity
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
**Remark 2** (Emergent integral structure). Within the present framework, the Laplace transform is not postulated a priori as an integral operator. Rather, it emerges naturally from the internal differential planting mechanism acting on the base kernel $1/s$. The exponential kernel $e^{-st}$ appears only after invoking the Gamma-function identity, revealing the integral representation as a consequence of an underlying purely differential structure. This does not invalidate the classical integral definition, but rather explains its structural origin.
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
st.markdown("acting on the base kernel $1/s$. In the Maclaurin-planting construction, each power $t^n$ selects a derivative rank $D^n$ to be applied to $1/s$. This notion of rank is the only bookkeeping device required in the applications below.")

st.markdown("**Basic kernel identity.** For every $n \\in \\mathbb{N}_0$ and $s > 0$,")
st.latex(r"""
D^n \left( \frac{1}{s} \right) = (-1)^n \frac{n!}{s^{n+1}}, \quad \text{hence} \quad (-D)^n \left( \frac{1}{s} \right) = \frac{n!}{s^{n+1}}.
""")

st.markdown("**Rank parity (sign absorption).** The factor $(-1)^n$ arising from $D^n(1/s)$ is absorbed by writing the planted operator as $(-D)^n$. This convention keeps all subsequent computations sign-clean.")

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
st.markdown("then the planted rank is lowered accordingly, and the operator-based transform satisfies")
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
A key structural feature of the planted-operator framework is the kernel shift
""")
st.latex(r"\frac{1}{s} \to \frac{1}{s-a},")
st.markdown("which encodes exponential tilting in the $t$-domain.")

st.markdown("**Shifted operator definition.** Given the Maclaurin expansion")
st.latex(r"f(t) = \sum_{n=0}^{\infty} a_n t^n,")
st.markdown("we define the shifted planted transform by")
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
# CASE DATA
# ============================================================
cases = {
    "Case A: f(t) = 1": {
        "title": "Case A: f(t) = 1",
        "series": r"1 = \sum_{n \geq 0} a_n t^n \quad \text{with } a_0 = 1,\; a_{n>0} = 0.",
        "rank": r"D^0",
        "plant_sum": r"T\{1\}(s) = D^0\left(\frac{1}{s}\right) = \frac{1}{s}.",
        "closed": r"T\{1\}(s) = \frac{1}{s}",
        "condition": r"s > 0",
        "has_integral": False,
    },
    "Case B: f(t) = t^n": {
        "title": "Case B: f(t) = t^n",
        "series": r"t^n = \sum_{k \geq 0} a_k t^k,\quad a_k = \delta_{k,n}",
        "rank": r"D^n",
        "plant_sum": r"""
T\{t^n\}(s) = (-1)^n D^n\left(\frac{1}{s}\right) = (-1)^n \frac{(-1)^n n!}{s^{n+1}} = \frac{n!}{s^{n+1}}.
""",
        "closed": r"T\{t^n\}(s) = \frac{n!}{s^{n+1}}",
        "condition": r"s > 0",
        "has_integral": False,
    },
    "Case C: f(t) = e^{at}": {
        "title": "Case C: f(t) = e^{at}",
        "series": r"e^{at} = \sum_{n \geq 0} \frac{a^n}{n!} t^n",
        "rank": r"D^n",
        "plant_sum": r"""
T\{e^{at}\}(s) = \sum_{n \geq 0} (-1)^n a_n D^n\left(\frac{1}{s}\right) = \sum_{n \geq 0} \frac{a^n}{n!} (-1)^n \frac{(-1)^n n!}{s^{n+1}} = \frac{1}{s} \sum_{n \geq 0} \left(\frac{a}{s}\right)^n = \frac{1}{s - a}, \quad \left|\frac{a}{s}\right| < 1.
""",
        "closed": r"T\{e^{at}\}(s) = \frac{1}{s - a}",
        "condition": r"\left|\frac{a}{s}\right| < 1",
        "has_integral": False,
    },
    "Case D: f(t) = cos(bt)": {
        "title": "Case D: f(t) = cos(bt)",
        "series": r"\cos(bt) = \sum_{n \geq 0} \frac{(-1)^n b^{2n}}{(2n)!} t^{2n}",
        "rank": r"D^{2n}",
        "plant_sum": r"""
T\{\cos(bt)\}(s) = \sum_{n \geq 0} a_{2n} (-1)^{2n} D^{2n}\left(\frac{1}{s}\right) = \sum_{n \geq 0} \frac{(-1)^n b^{2n}}{(2n)!} \frac{(2n)!}{s^{2n+1}} = \frac{1}{s} \sum_{n \geq 0} \left(-\frac{b^2}{s^2}\right)^n = \frac{s}{s^2 + b^2}, \quad \left|\frac{b}{s}\right| < 1.
""",
        "closed": r"T\{\cos(bt)\}(s) = \frac{s}{s^2 + b^2}",
        "condition": r"\left|\frac{b}{s}\right| < 1",
        "has_integral": False,
    },
    "Case E: f(t) = sin(bt)": {
        "title": "Case E: f(t) = sin(bt)",
        "series": r"\sin(bt) = \sum_{n \geq 0} \frac{(-1)^n b^{2n+1}}{(2n+1)!} t^{2n+1}",
        "rank": r"D^{2n+1}",
        "plant_sum": r"""
T\{\sin(bt)\}(s) = \sum_{n \geq 0} a_{2n+1} (-1)^{2n+1} D^{2n+1}\left(\frac{1}{s}\right) = \sum_{n \geq 0} \frac{(-1)^n b^{2n+1}}{(2n+1)!} \frac{(2n+1)!}{s^{2n+2}} = \frac{b}{s^2} \sum_{n \geq 0} \left(-\frac{b^2}{s^2}\right)^n = \frac{b}{s^2 + b^2}, \quad \left|\frac{b}{s}\right| < 1.
""",
        "closed": r"T\{\sin(bt)\}(s) = \frac{b}{s^2 + b^2}",
        "condition": r"\left|\frac{b}{s}\right| < 1",
        "has_integral": False,
    },
    "Case F: f(t) = t sin(bt)": {
        "title": "Case F: f(t) = t sin(bt)",
        "series": r"t \sin(bt) = \sum_{n \geq 0} \frac{(-1)^n b^{2n+1}}{(2n+1)!} t^{2n+2}",
        "rank": r"D^{2n+2}",
        "plant_sum": r"""
T\{t \sin(bt)\}(s) = \sum_{n \geq 0} \frac{(-1)^n b^{2n+1}}{(2n+1)!} \frac{(2n+2)!}{s^{2n+3}} = \frac{b}{s^3} \sum_{n \geq 0} (2n+2) \left( -\frac{b^2}{s^2} \right)^n = \frac{2b s}{(s^2 + b^2)^2}, \quad s > |b|.
""",
        "closed": r"T\{t\sin(bt)\}(s) = \frac{2bs}{(s^2 + b^2)^2}",
        "condition": r"s > |b|",
        "has_integral": False,
    },
    "Case G: f(t) = t cos(bt)": {
        "title": "Case G: f(t) = t cos(bt)",
        "series": r"t \cos(bt) = \sum_{n \geq 0} \frac{(-1)^n b^{2n}}{(2n)!} t^{2n+1}",
        "rank": r"D^{2n+1}",
        "plant_sum": r"""
T\{t \cos(bt)\}(s) = \sum_{n \geq 0} \frac{(-1)^n b^{2n}}{(2n)!} \frac{(2n+1)!}{s^{2n+2}} = \frac{1}{s^2} \sum_{n \geq 0} (2n+1) \left( -\frac{b^2}{s^2} \right)^n = \frac{s^2 - b^2}{(s^2 + b^2)^2}, \quad s > |b|.
""",
        "closed": r"T\{t\cos(bt)\}(s) = \frac{s^2 - b^2}{(s^2 + b^2)^2}",
        "condition": r"s > |b|",
        "has_integral": False,
    },
    "Case H: cosh(bt) and sinh(bt)": {
        "title": "Case H: cosh(bt) and sinh(bt)",
        "series": r"\cosh(bt) = \sum_{n \geq 0} \frac{b^{2n}}{(2n)!} t^{2n}, \quad \sinh(bt) = \sum_{n \geq 0} \frac{b^{2n+1}}{(2n+1)!} t^{2n+1}",
        "rank": r"D^{2n} \text{ (even)}, \quad D^{2n+1} \text{ (odd)}",
        "plant_sum": r"""
T\{\cosh(bt)\}(s) = \frac{1}{s} \sum_{n \geq 0} \left( \frac{b^2}{s^2} \right)^n = \frac{s}{s^2 - b^2}, \quad
T\{\sinh(bt)\}(s) = \frac{b}{s^2} \sum_{n \geq 0} \left( \frac{b^2}{s^2} \right)^n = \frac{b}{s^2 - b^2}, \quad \left|\frac{b}{s}\right| < 1.
""",
        "closed": r"T\{\cosh(bt)\}(s) = \frac{s}{s^2 - b^2}, \quad T\{\sinh(bt)\}(s) = \frac{b}{s^2 - b^2}",
        "condition": r"\left|\frac{b}{s}\right| < 1",
        "has_integral": False,
    },
    "Shifted Cases: e^{at}cos(bt), etc.": {
        "title": "Shifted Cases: e^{at}cos(bt), e^{at}sin(bt), e^{at}cosh(bt), e^{at}sinh(bt)",
        "series": r"\text{Use the shifted kernel } \frac{1}{s-a} \text{ with the same coefficients.}",
        "rank": r"\text{Rank preserved under kernel shifting.}",
        "plant_sum": r"""
T\{e^{at}\cos(bt)\} = \frac{s-a}{(s-a)^2 + b^2}, \quad
T\{e^{at}\sin(bt)\} = \frac{b}{(s-a)^2 + b^2}, \\
T\{e^{at}\cosh(bt)\} = \frac{s-a}{(s-a)^2 - b^2}, \quad
T\{e^{at}\sinh(bt)\} = \frac{b}{(s-a)^2 - b^2}.
""",
        "closed": r"T\{e^{at}\cos(bt)\} = \frac{s-a}{(s-a)^2 + b^2}, \quad T\{e^{at}\sin(bt)\} = \frac{b}{(s-a)^2 + b^2}, \quad T\{e^{at}\cosh(bt)\} = \frac{s-a}{(s-a)^2 - b^2}, \quad T\{e^{at}\sinh(bt)\} = \frac{b}{(s-a)^2 - b^2}",
        "condition": r"\text{Appropriate convergence conditions apply.}",
        "has_integral": False,
    },
    "Case I: sinc(bt) = sin(bt)/(bt)": {
        "title": "Case I: sinc(bt) = sin(bt)/(bt)",
        "series": r"\mathrm{sinc}(bt) = \sum_{n \geq 0} \frac{(-1)^n b^{2n}}{(2n+1)!} t^{2n}",
        "rank": r"D^{2n}",
        "plant_sum": r"""
T\{\mathrm{sinc}(bt)\}(s) = \sum_{n \geq 0} \frac{(-1)^n b^{2n}}{(2n+1)!} \frac{(2n)!}{s^{2n+1}} = \frac{1}{s} \sum_{n \geq 0} \frac{(-1)^n}{2n+1} \left( \frac{b}{s} \right)^{2n} = \frac{1}{b} \arctan\left(\frac{b}{s}\right).
""",
        "closed": r"T\{\mathrm{sinc}(bt)\}(s) = \frac{1}{b} \arctan\left(\frac{b}{s}\right)",
        "condition": r"\left|\frac{b}{s}\right| < 1",
        "has_integral": True,
        "integral_label": r"\int_0^\infty e^{-st} \frac{\sin(bt)}{bt} \, dt",
        "integral_result": r"\frac{1}{b} \tan^{-1}\left(\frac{b}{s}\right)",
    },
    "Case J1: (cos(bt)-1)/t": {
        "title": "Case J1: (cos(bt)-1)/t",
        "series": r"\frac{\cos(bt) - 1}{t} = \sum_{n \geq 1} \frac{(-1)^n b^{2n}}{(2n)!} t^{2n-1}",
        "rank": r"D^{2n-1}",
        "plant_sum": r"""
T\left\{\frac{\cos(bt)-1}{t}\right\}(s) = \sum_{n \geq 1} \frac{(-1)^n b^{2n}}{(2n)!} \frac{(2n-1)!}{s^{2n}} = -\frac{1}{2} \ln \left( 1 + \frac{b^2}{s^2} \right), \quad s > |b|.
""",
        "closed": r"T\left\{\frac{\cos(bt)-1}{t}\right\}(s) = -\frac{1}{2} \ln\left(1 + \frac{b^2}{s^2}\right)",
        "condition": r"s > |b|",
        "has_integral": True,
        "integral_label": r"\int_0^\infty e^{-st} \frac{\cos(bt) - 1}{t} \, dt",
        "integral_result": r"-\frac{1}{2} \ln\left(1 + \frac{b^2}{s^2}\right)",
    },
    "Case J2: (cos(bt)-1)/t^2": {
        "title": "Case J2: (cos(bt)-1)/t^2",
        "series": r"\frac{\cos(bt)-1}{t^2} = \sum_{n \geq 1} \frac{(-1)^n b^{2n}}{(2n)!} t^{2n-2}",
        "rank": r"D^{2n-2}",
        "plant_sum": r"""
T\left\{\frac{\cos(bt)-1}{t^2}\right\}(s) = \sum_{n \geq 1} \frac{(-1)^n b^{2n}}{(2n)!} \frac{(2n-2)!}{s^{2n-1}} = \frac{s}{2} \ln \left( 1 + \frac{b^2}{s^2} \right) - b \arctan\left(\frac{b}{s}\right), \quad s > |b|.
""",
        "closed": r"T\left\{\frac{\cos(bt)-1}{t^2}\right\}(s) = \frac{s}{2} \ln\left(1 + \frac{b^2}{s^2}\right) - b \arctan\left(\frac{b}{s}\right)",
        "condition": r"s > |b|",
        "has_integral": True,
        "integral_label": r"\int_0^\infty e^{-st} \frac{\cos(bt) - 1}{t^2} \, dt",
        "integral_result": r"\frac{s}{2} \ln\left(1 + \frac{b^2}{s^2}\right) - b \arctan\left(\frac{b}{s}\right)",
    },
    "Case J3: Bessel J_0(kt)": {
        "title": "Case J3: Bessel Function J_0(kt)",
        "series": r"J_0(kt) = \sum_{n=0}^{\infty} \frac{(-1)^n}{(n!)^2} \left(\frac{kt}{2}\right)^{2n}",
        "rank": r"D^{2n}",
        "plant_sum": r"""
T\{J_0(kt)\}(s) = \frac{1}{s} \sum_{n=0}^{\infty} \frac{(2n)!}{(n!)^2} \left( -\frac{k^2}{4s^2} \right)^n = \frac{1}{\sqrt{s^2 + k^2}}.
""",
        "closed": r"T\{J_0(kt)\}(s) = \frac{1}{\sqrt{s^2 + k^2}}",
        "condition": r"s > 0",
        "has_integral": False,
    },
    "Case J3: Bessel J_1(kt)": {
        "title": "Case J3: Bessel Function J_1(kt)",
        "series": r"J_1(kt) = -\frac{1}{k} \frac{d}{dt}J_0(kt)",
        "rank": r"\text{Derivative relation}",
        "plant_sum": r"""
J_1(kt) = -\frac{1}{k} \frac{d}{dt}J_0(kt) \implies T\{J_1(kt)\}(s) = -\frac{1}{k} \left[ sT\{J_0(kt)\}(s) - J_0(0) \right] = \frac{\sqrt{s^2 + k^2} - s}{k\sqrt{s^2 + k^2}}.
""",
        "closed": r"T\{J_1(kt)\}(s) = \frac{\sqrt{s^2 + k^2} - s}{k\sqrt{s^2 + k^2}}",
        "condition": r"s > 0",
        "has_integral": False,
    },
    "Case J3: Bessel J_ν(kt) General": {
        "title": "Case J3: Bessel Function J_ν(kt) (General Order)",
        "series": r"J_\nu(kt) = \sum_{n=0}^{\infty} \frac{(-1)^n}{n!\,\Gamma(n+\nu+1)} \left(\frac{kt}{2}\right)^{2n+\nu}",
        "rank": r"D^{2n+\nu}",
        "plant_sum": r"""
T\{J_\nu(kt)\}(s) = \left(\frac{k}{2}\right)^\nu \frac{1}{s^{\nu+1}} \sum_{n=0}^{\infty} \frac{(-1)^n \Gamma(2n+\nu+1)}{n!\,\Gamma(n+\nu+1)} \left(\frac{k^2}{4s^2}\right)^n = \frac{(\sqrt{s^2+k^2}-s)^\nu}{k^\nu\sqrt{s^2+k^2}}.
""",
        "closed": r"T\{J_\nu(kt)\}(s) = \frac{(\sqrt{s^2+k^2}-s)^\nu}{k^\nu\sqrt{s^2+k^2}}",
        "condition": r"s > 0,\; \nu > -1",
        "has_integral": False,
    },
}

# ============================================================
# INTERACTIVE SELECTION
# ============================================================
selected_case = st.selectbox(
    "Choose a symbolic Laplace case",
    list(cases.keys()),
    index=0,
    key="laplace_detailed_case",
)

case = cases[selected_case]

st.subheader(case["title"])

st.markdown("**Series**")
st.latex(case["series"])

st.markdown("**Rank**")
st.latex(case["rank"])

st.markdown("**Plant & Sum**")
st.latex(case["plant_sum"])

st.markdown("**Closed Form**")
st.latex(case["closed"])

st.markdown("**Validity / Convergence Condition**")
st.latex(case["condition"])

# ============================================================
# INTEGRAL SECTION (only for cases with has_integral = True)
# ============================================================
if case.get("has_integral", False):
    st.markdown("---")
    st.subheader("Integral Representation")

    st.markdown("The planted operator result corresponds to the following classical integral:")
    st.latex(case["integral_label"] + r" = " + case["integral_result"])

    # Extract parameters from the selected case title
    if "sinc" in selected_case:
        st.markdown("**Numerical Evaluation**")

        col1, col2 = st.columns(2)
        with col1:
            s_val = st.number_input("Enter value for $s$", value=1.0, step=0.1, format="%.2f", key="s_sinc")
        with col2:
            b_val = st.number_input("Enter value for $b$", value=1.0, step=0.1, format="%.2f", key="b_sinc")

        if s_val > 0:
            import math
            result = (1.0 / b_val) * math.atan(b_val / s_val)
            st.latex(r"\int_0^\infty e^{-" + f"{s_val:.2f}" + r"t} \frac{\sin(" + f"{b_val:.2f}" + r"t)}{" + f"{b_val:.2f}" + r"t} \, dt = " + f"{result:.6f}")

            if abs(s_val - 1.0) < 0.01 and abs(b_val - 1.0) < 0.01:
                st.latex(r"\Rightarrow \int_0^\infty e^{-t} \frac{\sin t}{t} \, dt = \tan^{-1}(1) = \frac{\pi}{4} \approx 0.785398")
            if abs(s_val) < 0.01 and abs(b_val - 1.0) < 0.01:
                st.latex(r"\Rightarrow \int_0^\infty \frac{\sin t}{t} \, dt = \frac{\pi}{2} \approx 1.570796")

    elif "J1" in selected_case:
        st.markdown("**Numerical Evaluation**")

        col1, col2 = st.columns(2)
        with col1:
            s_val = st.number_input("Enter value for $s$", value=1.0, step=0.1, format="%.2f", key="s_j1")
        with col2:
            b_val = st.number_input("Enter value for $b$", value=1.0, step=0.1, format="%.2f", key="b_j1")

        if s_val > abs(b_val):
            import math
            result = -0.5 * math.log(1 + (b_val / s_val)**2)
            st.latex(r"\int_0^\infty e^{-" + f"{s_val:.2f}" + r"t} \frac{\cos(" + f"{b_val:.2f}" + r"t) - 1}{t} \, dt = " + f"{result:.6f}")

    elif "J2" in selected_case:
        st.markdown("**Numerical Evaluation**")

        col1, col2 = st.columns(2)
        with col1:
            s_val = st.number_input("Enter value for $s$", value=1.0, step=0.1, format="%.2f", key="s_j2")
        with col2:
            b_val = st.number_input("Enter value for $b$", value=1.0, step=0.1, format="%.2f", key="b_j2")

        if s_val > abs(b_val):
            import math
            result = (s_val / 2.0) * math.log(1 + (b_val / s_val)**2) - b_val * math.atan(b_val / s_val)
            st.latex(r"\int_0^\infty e^{-" + f"{s_val:.2f}" + r"t} \frac{\cos(" + f"{b_val:.2f}" + r"t) - 1}{t^2} \, dt = " + f"{result:.6f}")

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
