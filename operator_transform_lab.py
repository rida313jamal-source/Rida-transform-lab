import math
import numpy as np
import sympy as sp
import streamlit as st
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
    h1, h2, h3 {
        letter-spacing: 0.2px;
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
# PART I: OPERATOR-BASED LAPLACE TRANSFORM
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

st.info("""
**Remark 1** (Convergence domain). The definition of $T$ does not impose an explicit a priori condition on $s$. The admissible values of $s$ are determined implicitly by the convergence of the operator series. In concrete examples, this convergence domain can be computed explicitly and coincides with the classical region of convergence of the Laplace transform (e.g. trigonometric, exponential, and Bessel-type functions).
""")

st.info("""
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
# CASE STUDIES
# ============================================================
st.header("Case Studies")

st.info("""
**Remark 3.** The constants $a, b, k,$ and similar parameters appearing in the examples (such as $e^{at}, \cos(bt),$ and $J_0(kt)$) arise naturally from the Maclaurin expansion of the input function. Within the present operator-based framework, these constants act as inherited scaling factors that govern the growth or oscillatory behavior of the function and, consequently, determine the effective rank and convergence behavior of the associated operator series.
""")

# ============================================================
# CASE A
# ============================================================
st.subheader("Case A: f(t) = 1")

st.markdown("**Series:** $1 = \\sum_{n \\geq 0} a_n t^n$ with $a_0 = 1$, $a_{n>0} = 0$.")
st.markdown("**Rank:** $D^0$.")
st.markdown("**Plant & Sum.**")
st.latex(r"T\{1\}(s) = D^0\left(\frac{1}{s}\right) = \frac{1}{s}.")

# ============================================================
# CASE B
# ============================================================
st.subheader("Case B: f(t) = t^n")

st.markdown("**Series:** Monomial; $a_k = \\delta_{k,n}$ (only the $k = n$ term is nonzero).")
st.markdown("**Rank:** $D^n$.")
st.markdown("**Plant & Sum.**")
st.latex(r"""
T\{t^n\}(s) = (-1)^n D^n\left(\frac{1}{s}\right) = (-1)^n \frac{(-1)^n n!}{s^{n+1}} = \frac{n!}{s^{n+1}}.
""")

# ============================================================
# CASE C
# ============================================================
st.subheader("Case C: f(t) = e^{at}")

st.markdown("**Series:** $e^{at} = \\sum_{n \\geq 0} \\frac{a^n}{n!} t^n$, $a_n = \\frac{a^n}{n!}$.")
st.markdown("**Rank:** $D^n$.")
st.markdown("**Plant & Sum.**")
st.latex(r"""
T\{e^{at}\}(s) = \sum_{n \geq 0} (-1)^n a_n D^n\left(\frac{1}{s}\right) = \sum_{n \geq 0} \frac{a^n}{n!} (-1)^n \frac{(-1)^n n!}{s^{n+1}} = \frac{1}{s} \sum_{n \geq 0} \left(\frac{a}{s}\right)^n = \frac{1}{s - a}, \quad \left|\frac{a}{s}\right| < 1.
""")

# ============================================================
# CASE D
# ============================================================
st.subheader("Case D: f(t) = cos(bt)")

st.markdown("**Series:** $\\cos(bt) = \\sum_{n \\geq 0} \\frac{(-1)^n b^{2n}}{(2n)!} t^{2n}$.")
st.markdown("**Coefficients:** $a_{2n} = \\frac{(-1)^n b^{2n}}{(2n)!}$.")
st.markdown("**Rank:** $D^{2n}$.")
st.markdown("**Plant & Sum.**")
st.latex(r"""
T\{\cos(bt)\}(s) = \sum_{n \geq 0} a_{2n} (-1)^{2n} D^{2n}\left(\frac{1}{s}\right) = \sum_{n \geq 0} \frac{(-1)^n b^{2n}}{(2n)!} \frac{(2n)!}{s^{2n+1}} = \frac{1}{s} \sum_{n \geq 0} \left(-\frac{b^2}{s^2}\right)^n = \frac{s}{s^2 + b^2}, \quad \left|\frac{b}{s}\right| < 1.
""")

# ============================================================
# CASE E
# ============================================================
st.subheader("Case E: f(t) = sin(bt)")

st.markdown("**Series:** $\\sin(bt) = \\sum_{n \\geq 0} \\frac{(-1)^n b^{2n+1}}{(2n+1)!} t^{2n+1}$.")
st.markdown("**Coefficients:** $a_{2n+1} = \\frac{(-1)^n b^{2n+1}}{(2n+1)!}$.")
st.markdown("**Rank:** $D^{2n+1}$.")
st.markdown("**Plant & Sum.**")
st.latex(r"""
T\{\sin(bt)\}(s) = \sum_{n \geq 0} a_{2n+1} (-1)^{2n+1} D^{2n+1}\left(\frac{1}{s}\right) = \sum_{n \geq 0} \frac{(-1)^n b^{2n+1}}{(2n+1)!} \frac{(2n+1)!}{s^{2n+2}} = \frac{b}{s^2} \sum_{n \geq 0} \left(-\frac{b^2}{s^2}\right)^n = \frac{b}{s^2 + b^2}, \quad \left|\frac{b}{s}\right| < 1.
""")

# ============================================================
# CASE F
# ============================================================
st.subheader("Case F: f(t) = t sin(bt)")

st.markdown("**Series:** $\\sin(bt) = \\sum_{n \\geq 0} \\frac{(-1)^n b^{2n+1}}{(2n+1)!} t^{2n+1}$")
st.latex(r"\Rightarrow t \sin(bt) = \sum_{n \geq 0} \frac{(-1)^n b^{2n+1}}{(2n+1)!} t^{2n+2}.")
st.markdown("Thus $k = 2n + 2$,")
st.latex(r"a_k = a_{2n+2} = \frac{(-1)^n b^{2n+1}}{(2n+1)!}.")
st.markdown("**Rank:** $D^k = D^{2n+2}$.")
st.markdown("**Plant & Sum.**")
st.latex(r"""
T\{t \sin(bt)\}(s) = \sum_{n \geq 0} a_{2n+2}(-1)^{2n+2} D^{2n+2} \left( \frac{1}{s} \right) = \sum_{n \geq 0} \frac{(-1)^n b^{2n+1}}{(2n+1)!} \frac{(2n+2)!}{s^{2n+3}}.
""")
st.latex(r"""
= \frac{b}{s^3} \sum_{n \geq 0} (2n+2) \left( -\frac{b^2}{s^2} \right)^n.
""")
st.markdown("With $u = \\frac{b^2}{s^2}$ we use")
st.latex(r"\sum_{n \geq 0} (2n+2) u^n = \frac{2}{(1-u)^2},")
st.markdown("hence")
st.latex(r"""
T\{t \sin(bt)\}(s) = \frac{b}{s^3} \frac{2}{(1+\frac{b^2}{s^2})^2} = \frac{2b s}{(s^2 + b^2)^2}, \quad s > |b|.
""")

# ============================================================
# CASE G
# ============================================================
st.subheader("Case G: f(t) = t cos(bt)")

st.markdown("**Series:** $\\cos(bt) = \\sum_{n \\geq 0} \\frac{(-1)^n b^{2n}}{(2n)!} t^{2n}$")
st.latex(r"\Rightarrow t \cos(bt) = \sum_{n \geq 0} \frac{(-1)^n b^{2n}}{(2n)!} t^{2n+1}.")
st.markdown("Thus $k = 2n + 1$,")
st.latex(r"a_k = a_{2n+1} = \frac{(-1)^n b^{2n}}{(2n)!}.")
st.markdown("**Rank:** $D^k = D^{2n+1}$.")
st.markdown("**Plant & Sum.**")
st.latex(r"""
T\{t \cos(bt)\}(s) = \sum_{n \geq 0} a_{2n+1}(-1)^{2n+1} D^{2n+1} \left( \frac{1}{s} \right) = \sum_{n \geq 0} \frac{(-1)^n b^{2n}}{(2n)!} \frac{(2n+1)!}{s^{2n+2}}.
""")
st.latex(r"""
= \frac{1}{s^2} \sum_{n \geq 0} (2n+1) \left( -\frac{b^2}{s^2} \right)^n.
""")
st.markdown("Using")
st.latex(r"\sum_{n \geq 0} (2n+1)(-u)^n = \frac{1-u}{(1+u)^2}, \quad u = \frac{b^2}{s^2},")
st.markdown("we obtain")
st.latex(r"""
T\{t \cos(bt)\}(s) = \frac{1}{s^2} \frac{1-\frac{b^2}{s^2}}{(1+\frac{b^2}{s^2})^2} = \frac{s^2 - b^2}{(s^2 + b^2)^2}, \quad s > |b|.
""")

# ============================================================
# CASE H
# ============================================================
st.subheader("Case H: f(t) = cosh(bt) and f(t) = sinh(bt)")

st.markdown("**Series:**")
st.latex(r"""
\cosh(bt) = \sum_{n \geq 0} \frac{b^{2n}}{(2n)!} t^{2n}, \quad \sinh(bt) = \sum_{n \geq 0} \frac{b^{2n+1}}{(2n+1)!} t^{2n+1}.
""")
st.markdown("**Ranks:** even $\\rightarrow D^{2n}$, odd $\\rightarrow D^{2n+1}$.")
st.markdown("**Sums.**")
st.latex(r"""
T\{\cosh(bt)\}(s) = \frac{1}{s} \sum_{n \geq 0} \left( \frac{b^2}{s^2} \right)^n = \frac{1}{s} \cdot \frac{1}{1 - \frac{b^2}{s^2}} = \frac{s}{s^2 - b^2}, \quad \left| \frac{b}{s} \right| < 1.
""")
st.latex(r"""
T\{\sinh(bt)\}(s) = \frac{b}{s^2} \sum_{n \geq 0} \left( \frac{b^2}{s^2} \right)^n = \frac{b}{s^2} \cdot \frac{1}{1 - \frac{b^2}{s^2}} = \frac{b}{s^2 - b^2}, \quad \left| \frac{b}{s} \right| < 1.
""")

# ============================================================
# SHIFTED CASES
# ============================================================
st.subheader("Shifted Cases: Exponentially tilted trig/hyperbolic")

st.markdown("Using the shifted version $s \\mapsto s-a$:")
st.latex(r"""
T_a\{f\}(s) = \sum_{n=0}^{\infty} (-1)^n a_n D^n \left( \frac{1}{s - a} \right).
""")
st.latex(r"""
T\{e^{at} \cos(bt)\} = \frac{s - a}{(s - a)^2 + b^2}, \quad T\{e^{at} \sin(bt)\} = \frac{b}{(s - a)^2 + b^2}.
""")
st.latex(r"""
T\{e^{at} \cosh(bt)\} = \frac{s - a}{(s - a)^2 - b^2}, \quad T\{e^{at} \sinh(bt)\} = \frac{b}{(s - a)^2 - b^2}.
""")

# ============================================================
# CASE I
# ============================================================
st.subheader("Case I: f(t) = sinc(bt) = sin(bt)/(bt)")

st.markdown("**Series.**")
st.latex(r"""
\sin(bt) = \sum_{n \geq 0} \frac{(-1)^n b^{2n+1}}{(2n+1)!} t^{2n+1} \implies \mathrm{sinc}(bt) = \frac{\sin(bt)}{bt} = \sum_{n \geq 0} \frac{(-1)^n b^{2n}}{(2n+1)!} t^{2n}.
""")
st.markdown("Hence,")
st.latex(r"""
a_{2n} = \frac{(-1)^n b^{2n}}{(2n+1)!}, \quad \text{Rank: } D^{2n}.
""")
st.markdown("**Plant & Sum.**")
st.latex(r"""
T\{\mathrm{sinc}(bt)\}(s) = \sum_{n \geq 0} a_{2n} (-1)^{2n} D^{2n} \left( \frac{1}{s} \right) = \sum_{n \geq 0} \frac{(-1)^n b^{2n}}{(2n+1)!} \frac{(2n)!}{s^{2n+1}}.
""")
st.latex(r"""
= \frac{1}{s} \sum_{n \geq 0} \frac{(-1)^n}{2n+1} \left( \frac{b}{s} \right)^{2n}.
""")
st.markdown("Using the identity")
st.latex(r"\sum_{n \geq 0} \frac{(-1)^n x^{2n+1}}{2n+1} = \arctan(x), \quad |x| < 1,")
st.markdown("with $x = \\frac{b}{s}$, we get")
st.latex(r"""
T\{\mathrm{sinc}(bt)\}(s) = \frac{1}{b} \arctan\left(\frac{b}{s}\right).
""")

# ============================================================
# CASE J1
# ============================================================
st.subheader("Case J1: f(t) = (cos(bt)-1)/t")

st.markdown("**Remark (skipping the $n = 0$ term).** Since")
st.latex(r"""
\frac{\cos(bt)}{t} = \frac{1}{t} + \sum_{n \geq 1} \frac{(-1)^n b^{2n}}{(2n)!} t^{2n-1},
""")
st.markdown("the term $1/t$ is not admissible at $t = 0$; hence we exclude the $n = 0$ term and work with the series starting at $n = 1$.")

st.markdown("**Series.**")
st.latex(r"""
\cos(bt) = \sum_{n \geq 0} \frac{(-1)^n (bt)^{2n}}{(2n)!} \implies \frac{\cos(bt) - 1}{t} = \sum_{n \geq 1} \frac{(-1)^n b^{2n}}{(2n)!} t^{2n-1}.
""")
st.markdown("Hence, the planted coefficients and rank are")
st.latex(r"""
a_{2n-1} = \frac{(-1)^n b^{2n}}{(2n)!}, \quad \text{rank} = 2n - 1.
""")
st.markdown("Using the operator form")
st.latex(r"""
\mathcal{T} \{ f(t) \}(s) = \sum_{k \geq 0} a_k (-1)^k D_s^k \left( \frac{1}{s} \right), \quad D_s^k \left( \frac{1}{s} \right) = \frac{k!}{s^{k+1}},
""")
st.markdown("we obtain")
st.latex(r"""
\mathcal{T} \left\{ \frac{\cos(bt) - 1}{t} \right\}(s) = \sum_{n \geq 1} a_{2n-1} (-1)^{2n-1} D_s^{2n-1} \left( \frac{1}{s} \right)
""")
st.latex(r"""
= \sum_{n \geq 1} \frac{(-1)^n b^{2n}}{(2n)!} \frac{(2n - 1)!}{s^{2n}}
""")
st.latex(r"""
= \sum_{n \geq 1} \frac{(-1)^{n+1} b^{2n}}{2n} \frac{1}{s^{2n}} = \frac{1}{2} \sum_{n \geq 1} \frac{1}{n} \left( -\frac{b^2}{s^2} \right)^n.
""")
st.markdown("Using the identity")
st.latex(r"\sum_{n \geq 1} \frac{r^n}{n} = -\ln(1 - r), \quad |r| < 1,")
st.markdown("we conclude")
st.latex(r"""
\mathcal{T} \left\{ \frac{\cos(bt) - 1}{t} \right\}(s) = -\frac{1}{2} \ln \left( 1 + \frac{b^2}{s^2} \right), \quad s > |b|.
""")
st.latex(r"""
\mathcal{T} \left\{ \frac{\cos(bt) - 1}{t} \right\}(s) = \int_0^\infty e^{-st} \frac{\cos(bt) - 1}{t} \, dt = -\frac{1}{2} \ln \left( 1 + \frac{b^2}{s^2} \right).
""")

# ============================================================
# CASE J2
# ============================================================
st.subheader("Case J2: f(t) = (cos(bt)-1)/t^2")

st.markdown("**Series.**")
st.latex(r"""
\cos(bt) - 1 = \sum_{n \geq 1} \frac{(-1)^n b^{2n}}{(2n)!} t^{2n} \implies \frac{\cos(bt) - 1}{t^2} = \sum_{n \geq 1} \frac{(-1)^n b^{2n}}{(2n)!} t^{2n-2}.
""")
st.markdown("Hence the planted coefficients and rank are")
st.latex(r"""
a_{2n-2} = \frac{(-1)^n b^{2n}}{(2n)!}, \quad \text{rank} = D^{2n-2}.
""")
st.markdown("**Plant & Sum.** Using the unified operator form")
st.latex(r"""
T\{f(t)\}(s) = \sum_{k \geq 0} a_k(-1)^k D_s^k \left( \frac{1}{s} \right), \quad D_s^n \left( \frac{1}{s} \right) = \frac{n!}{s^{n+1}},
""")
st.markdown("we get")
st.latex(r"""
T \left\{ \frac{\cos(bt) - 1}{t^2} \right\}(s) = \sum_{n \geq 1} a_{2n-2} (-1)^{2n-2} D_s^{2n-2} \left( \frac{1}{s} \right)
""")
st.latex(r"""
= \sum_{n \geq 1} \frac{(-1)^n b^{2n}}{(2n)!} \frac{(2n-2)!}{s^{2n-1}} = \sum_{n \geq 1} \frac{(-1)^n b^{2n}}{2n(2n-1)} \frac{1}{s^{2n-1}}.
""")
st.latex(r"""
= s \sum_{n \geq 1} \frac{(-1)^n}{2n(2n-1)} \left( \frac{b^2}{s^2} \right)^n.
""")
st.markdown("**Partial fraction and summation.** With")
st.latex(r"""
\frac{1}{2n(2n-1)} = \frac{1}{2n} + \frac{1}{2n-1}, \quad r = \frac{b^2}{s^2} \quad (|r| < 1 \iff s > |b|),
""")
st.markdown("we write")
st.latex(r"""
T \left\{ \frac{\cos(bt) - 1}{t^2} \right\}(s) = s \left[ -\frac{1}{2} \sum_{n \geq 1} \frac{(-r)^n}{n} + \sum_{n \geq 1} \frac{(-r)^n}{2n-1} \right].
""")
st.markdown("Using the identities (for $|z| < 1$):")
st.latex(r"""
\sum_{n \geq 1} \frac{z^n}{n} = -\ln(1-z), \quad \sum_{n \geq 1} \frac{z^n}{2n-1} = \sqrt{z} \arctan(\sqrt{z}),
""")
st.markdown("and substituting $z = -r$ (so that $\\sqrt{-r} = i\\sqrt{r}$ and $\\arctan(iu) = i \\arctan(u)$), we obtain the real form")
st.latex(r"""
\sum_{n \geq 1} \frac{(-r)^n}{2n-1} = -\sqrt{r} \arctan(\sqrt{r}) = -\frac{b}{s} \arctan\left(\frac{b}{s}\right).
""")
st.markdown("Also,")
st.latex(r"""
\sum_{n \geq 1} \frac{(-r)^n}{n} = -\ln(1+r).
""")
st.markdown("Therefore,")
st.latex(r"""
T \left\{ \frac{\cos(bt) - 1}{t^2} \right\}(s) = \frac{s}{2} \ln \left( 1 + \frac{b^2}{s^2} \right) - b \arctan\left(\frac{b}{s}\right), \quad s > |b|.
""")
st.latex(r"""
T \left\{ \frac{\cos(bt) - 1}{t^2} \right\}(s) \equiv \int_0^\infty e^{-st} \frac{\cos(bt) - 1}{t^2} dt = \frac{s}{2} \ln \left( 1 + \frac{b^2}{s^2} \right) - b \arctan\left(\frac{b}{s}\right).
""")

# ============================================================
# CASE J3
# ============================================================
st.subheader("Case J3: Bessel Function under the Operator Laplace Transform")

st.markdown("""
We now apply the differential planting definition of the operator Laplace transform to the Bessel function of the first kind:
""")
st.latex(r"""
J_\nu(kt) = \sum_{n=0}^\infty \frac{(-1)^n}{n! \Gamma(n + \nu + 1)} \left( \frac{kt}{2} \right)^{2n + \nu}.
""")
st.markdown("""
According to the differential-operator formulation, each term $a_{2n+\nu}$ is planted as a derivative of the seed kernel $1/s$:
""")
st.latex(r"""
\mathcal{T} \{ f(t) \} (s) = \sum_{n=0}^\infty (-1)^n a_n D^n \left( \frac{1}{s} \right),
""")
st.markdown("thus giving")
st.latex(r"""
\mathcal{T} \{ J_\nu(kt) \} (s) = \sum_{n=0}^\infty \frac{(-1)^n}{n! \Gamma(n + \nu + 1)} \left( \frac{k}{2} \right)^{2n + \nu} D^{2n+\nu} \left( \frac{1}{s} \right).
""")
st.markdown("Using the fractional-derivative identity")
st.latex(r"""
D^n \left( \frac{1}{s} \right) = (-1)^n \frac{\Gamma(n+1)}{s^{n+1}},
""")
st.markdown("we obtain")
st.latex(r"""
\mathcal{T} \{ J_\nu(kt) \} (s) = \left( \frac{k}{2} \right)^\nu \frac{1}{s^{\nu+1}} \sum_{n=0}^\infty \frac{(-1)^n \Gamma(2n+\nu+1)}{n! \Gamma(n+\nu+1)} \left( \frac{k^2}{4s^2} \right)^n.
""")
st.markdown("For the case $\\nu = 0$, this simplifies to")
st.latex(r"""
\mathcal{T} \{ J_0(kt) \} (s) = \frac{1}{s} \sum_{n=0}^\infty \frac{(2n)!}{(n!)^2} \left( -\frac{k^2}{4s^2} \right)^n.
""")
st.markdown("Using the identity")
st.latex(r"""
\sum_{n=0}^\infty \frac{(2n)!}{(n!)^2} z^n = \frac{1}{\sqrt{1-4z}},
""")
st.markdown("we obtain")
st.latex(r"""
\frac{1}{s} \sum_{n=0}^\infty \frac{(2n)!}{(n!)^2} z^n = \frac{1}{s \sqrt{1-4z}}.
""")
st.latex(r"""
\text{Since } 1 - 4z = 1 + \frac{k^2}{s^2} = \frac{s^2 + k^2}{s^2},
""")
st.latex(r"""
\sqrt{1 - 4z} = \frac{\sqrt{s^2 + k^2}}{s}.
""")
st.latex(r"""
\frac{1}{s\sqrt{1 - 4z}} = \frac{1}{s} \cdot \frac{s}{\sqrt{s^2 + k^2}} = \frac{1}{\sqrt{s^2 + k^2}}.
""")
st.markdown("OR. Recognizing the standard binomial expansion,")
st.latex(r"""
(1 + z)^{-1/2} = \sum_{n=0}^\infty \binom{2n}{n} \left(\frac{-z}{4}\right)^n,
""")
st.markdown("we obtain the closed form")
st.latex(r"""
\mathcal{T}\{J_0(kt)\}(s) = \frac{1}{\sqrt{s^2 + k^2}}.
""")
st.markdown("**Verification.** This coincides perfectly with the classical Laplace transform of $J_0(kt)$:")
st.latex(r"""
\int_0^\infty e^{-st}J_0(kt) \, dt = \frac{1}{\sqrt{s^2 + k^2}}.
""")
st.markdown("Hence, the operator Laplace transform reproduces the standard integral result directly from the planted differential series, confirming its internal consistency and computational power.")

st.subheader("Example: J_1(kt) via the derivative identity")

st.markdown("""
The Bessel functions satisfy the well-known derivative relation
""")
st.latex(r"\frac{d}{dz}J_0(z) = -J_1(z).")
st.markdown("For the scaled argument $z = kt$ this gives")
st.latex(r"""
\frac{d}{dt}J_0(kt) = kJ_0'(kt) = -kJ_1(kt),
""")
st.markdown("hence")
st.latex(r"""
J_1(kt) = -\frac{1}{k} \frac{d}{dt}J_0(kt).
""")
st.markdown("Using the operator Laplace transform")
st.latex(r"""
\mathcal{T}\{f(t)\}(s) = \int_0^\infty e^{-st}f(t) \, dt,
""")
st.markdown("integration by parts yields the standard derivative rule")
st.latex(r"""
\mathcal{T}\{f'(t)\}(s) = s\mathcal{T}\{f(t)\}(s) - f(0),
""")
st.markdown("whenever the boundary terms vanish at infinity. Applying this with $f(t) = J_0(kt)$ and noting that")
st.latex(r"""
J_0(0) = 1, \quad \mathcal{T}\{J_0(kt)\}(s) = \frac{1}{\sqrt{s^2 + k^2}},
""")
st.markdown("we obtain")
st.latex(r"""
\mathcal{T}\{J_1(kt)\}(s) = -\frac{1}{k} \mathcal{T}\left\{\frac{d}{dt}J_0(kt)\right\}(s)
""")
st.latex(r"""
= -\frac{1}{k} \left[ s\mathcal{T}\{J_0(kt)\}(s) - J_0(0) \right] = -\frac{1}{k} \left[ \frac{s}{\sqrt{s^2 + k^2}} - 1 \right].
""")
st.markdown("Thus,")
st.latex(r"""
\mathcal{T}\{J_1(kt)\}(s) = \frac{1}{k} \left( 1 - \frac{s}{\sqrt{s^2 + k^2}} \right) = \frac{\sqrt{s^2 + k^2} - s}{k\sqrt{s^2 + k^2}}.
""")
st.markdown("This agrees with the general closed form")
st.latex(r"""
\mathcal{T}\{J_\nu(kt)\}(s) = \frac{(\sqrt{s^2 + k^2} - s)^\nu}{k^\nu \sqrt{s^2 + k^2}}, \quad (\nu = 1),
""")
st.markdown("confirming that the operator Laplace transform reproduces the classical Laplace transform of $J_1(kt)$ directly from the differential structure of the Bessel hierarchy. This expression seamlessly matches the classical Laplace form while emerging purely from the differential-operator construction without requiring an initial integral definition.")

st.divider()

# ============================================================
# SECTION 2.6: INTEGRALS DERIVED FROM THE SERIES
# ============================================================
st.header("2.6 Integrals derived from the series")

st.markdown(r"""
From Case I (sinc function):
""")
st.latex(r"""
\int_0^\infty e^{-st} \frac{\sin(bt)}{bt} \, dt = \mathcal{T} \{\mathrm{sinc}(bt)\}(s) = \frac{1}{b} \tan^{-1} \left( \frac{b}{s} \right).
""")
st.markdown(r"$\Rightarrow s = 1, \quad b = 1$")
st.latex(r"""
\int_0^\infty e^{-t} \frac{\sin t}{t} \, dt = \tan^{-1}(1) = \frac{\pi}{4}.
""")
st.markdown(r"$\Rightarrow s = 0, \quad b = 1$")
st.latex(r"""
\int_0^\infty \frac{\sin t}{t} \, dt = \frac{\pi}{2}.
""")

st.markdown(r"""
From Case J1:
""")
st.latex(r"""
\int_0^\infty e^{-st} \frac{\cos(bt) - 1}{t} \, dt = -\frac{1}{2} \ln \left( 1 + \frac{b^2}{s^2} \right), \quad s > |b|.
""")

st.markdown(r"""
From Case J2:
""")
st.latex(r"""
\int_0^\infty e^{-st} \frac{\cos(bt) - 1}{t^2} \, dt = \frac{s}{2} \ln \left( 1 + \frac{b^2}{s^2} \right) - b \arctan\left(\frac{b}{s}\right), \quad s > |b|.
""")

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
