import math
import numpy as np
import sympy as sp
import streamlit as st


# ============================================================
# Page setup
# ============================================================
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
# Symbols
# ============================================================
x = sp.Symbol("x", real=True)
s = sp.Symbol("s", positive=True, real=True)
t = sp.Symbol("t", positive=True, real=True)
r = sp.Symbol("r", positive=True, real=True)

sigma = sp.Symbol("sigma", positive=True, real=True)
omega = sp.Symbol("omega", real=True)

a = sp.Symbol("a", real=True)
b = sp.Symbol("b", real=True)
k = sp.Symbol("k", positive=True, real=True)
R = sp.Symbol("R", positive=True, real=True)
mu = sp.Symbol("mu", real=True)
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


def to_expr(value):
    try:
        return sp.nsimplify(value)
    except Exception:
        return sp.Float(value)


def render_formula(title: str, formula: str):
    st.markdown(f"**{title}**")
    st.latex(formula)


def render_case(case_data: dict):
    st.subheader(case_data["title"])

    st.markdown("**Function**")
    st.latex(case_data["function"])

    st.markdown("**Series**")
    st.latex(case_data["series"])

    st.markdown("**Coefficients**")
    st.latex(case_data["coefficients"])

    st.markdown("**Rank**")
    st.latex(case_data["rank"])

    st.markdown("**Plant & Sum**")
    for step in case_data["plant_sum"]:
        st.latex(step)

    st.markdown("**Closed Form**")
    st.latex(case_data["closed_form"])

    st.markdown("**Validity / Convergence Condition**")
    st.latex(case_data["condition"])


# ============================================================
# Introduction
# ============================================================
def render_intro():
    st.header("1. Introduction")

    st.markdown(
        """
        This application presents a first interactive version of a unified operator-based
        transform framework. The main idea is to start from the Maclaurin expansion
        of a function and to plant its coefficients into differential actions on a simple kernel.
        """
    )

    st.latex(r"f(x)=\sum_{n=0}^{\infty} a_n x^n")

    st.markdown("The framework is organized around four main transforms:")

    st.latex(r"\text{Laplace} \quad \text{Fourier} \quad \text{Mellin} \quad \text{Hankel}")

    st.markdown("The unifying principle is the planting rule:")

    st.latex(r"x^n \;\mapsto\; (-\partial)^n(\text{kernel})")

    st.markdown("Different transforms arise by changing the kernel and the effective rank.")

    st.latex(r"\text{Laplace: } (-\partial_s)^n\left(\frac{1}{s}\right)")
    st.latex(r"\text{Fourier: } \frac{1}{s},\;\frac{1}{\bar s},\quad s=\sigma+i\omega")
    st.latex(r"\text{Mellin: } (-\partial_t)^{n+s-1}\left(\frac{1}{t}\right)")
    st.latex(r"\text{Hankel: } (-\partial_t)^{n+\nu}\left(\frac{1}{t}\right)")

    st.markdown(
        """
        Each transform section below contains:
        1. a definition,
        2. a proof sketch or derivation,
        3. an interactive calculator for representative families.
        """
    )


# ============================================================
# Laplace: definition and proof
# ============================================================
def render_laplace_definition_and_proof():
    st.header("Part I — Operator-Based Laplace Transform")

    st.markdown(
        """
        This section presents the operator-based Laplace transform in the same structural
        way as the monograph. The classical Laplace transform is reconstructed from
        Maclaurin coefficients planted into derivatives of the seed kernel $1/s$.
        """
    )

    st.subheader("Definition")

    st.latex(
        r"""
        f(x)=\sum_{n=0}^{\infty} a_n x^n
        """
    )

    st.latex(
        r"""
        T\{f\}(s)
        :=
        \sum_{n=0}^{\infty} a_n (-D_s)^n\!\left(\frac{1}{s}\right),
        \qquad
        D_s=\frac{d}{ds}.
        """
    )

    st.subheader("Equivalence with the Classical Laplace Transform")

    st.latex(
        r"""
        \mathcal{L}\{f\}(s)=\int_{0}^{\infty} e^{-sx} f(x)\,dx
        """
    )

    st.markdown("**Proof sketch**")

    st.latex(
        r"""
        D_s^n\!\left(\frac{1}{s}\right)=(-1)^n\frac{n!}{s^{n+1}}
        \quad\Longrightarrow\quad
        (-D_s)^n\!\left(\frac{1}{s}\right)=\frac{n!}{s^{n+1}}.
        """
    )

    st.latex(
        r"""
        T\{f\}(s)
        =
        \sum_{n=0}^{\infty} a_n \frac{n!}{s^{n+1}}.
        """
    )

    st.latex(
        r"""
        \frac{1}{s^{n+1}}
        =
        \frac{1}{n!}\int_{0}^{\infty} x^n e^{-sx}\,dx,
        \qquad s>0.
        """
    )

    st.latex(
        r"""
        T\{f\}(s)
        =
        \sum_{n=0}^{\infty} a_n
        \int_{0}^{\infty} x^n e^{-sx}\,dx
        =
        \int_{0}^{\infty} e^{-sx}
        \left(\sum_{n=0}^{\infty} a_n x^n\right)dx.
        """
    )

    st.latex(
        r"""
        T\{f\}(s)=\int_{0}^{\infty} e^{-sx} f(x)\,dx
        =
        \mathcal{L}\{f\}(s).
        """
    )

    st.info(
        "Below, each case is displayed in the same detailed style as the monograph: "
        "series, coefficient law, rank, planted operator form, summation, and final closed form."
    )


# ============================================================
# Laplace detailed symbolic cases
# ============================================================
def get_laplace_cases():
    cases = {}

    cases["1"] = {
        "title": "Case A: Constant Function",
        "function": r"f(x)=1",
        "series": r"1=\sum_{n\ge 0} a_n x^n,\qquad a_0=1,\; a_{n>0}=0.",
        "coefficients": r"a_0=1,\qquad a_{n>0}=0.",
        "rank": r"D^0",
        "plant_sum": [
            r"T\{1\}(s)=D^0\!\left(\frac{1}{s}\right)",
        ],
        "closed_form": r"T\{1\}(s)=\frac{1}{s}",
        "condition": r"s>0",
    }

    cases["x^n"] = {
        "title": "Case B: Monomial",
        "function": r"f(x)=x^n",
        "series": r"x^n=\sum_{k\ge 0} a_k x^k,\qquad a_k=\delta_{k,n}",
        "coefficients": r"a_k=\delta_{k,n}\quad\text{(only the term }k=n\text{ survives)}.",
        "rank": r"D^n",
        "plant_sum": [
            r"T\{x^n\}(s)=(-D)^n\!\left(\frac{1}{s}\right)",
            r"D^n\!\left(\frac{1}{s}\right)=(-1)^n\frac{n!}{s^{n+1}}",
            r"(-D)^n\!\left(\frac{1}{s}\right)=\frac{n!}{s^{n+1}}",
        ],
        "closed_form": r"T\{x^n\}(s)=\frac{n!}{s^{n+1}}",
        "condition": r"s>0",
    }

    cases["e^(a x)"] = {
        "title": "Case C: Exponential Function",
        "function": r"f(x)=e^{ax}",
        "series": r"e^{ax}=\sum_{n\ge 0}\frac{a^n}{n!}x^n",
        "coefficients": r"a_n=\frac{a^n}{n!}",
        "rank": r"D^n",
        "plant_sum": [
            r"T\{e^{ax}\}(s)=\sum_{n\ge 0}\frac{a^n}{n!}(-D)^n\!\left(\frac{1}{s}\right)",
            r"=\sum_{n\ge 0}\frac{a^n}{n!}\frac{n!}{s^{n+1}}",
            r"=\frac{1}{s}\sum_{n\ge 0}\left(\frac{a}{s}\right)^n",
        ],
        "closed_form": r"T\{e^{ax}\}(s)=\frac{1}{s-a}",
        "condition": r"\left|\frac{a}{s}\right|<1",
    }

    cases["cos(b x)"] = {
        "title": "Case D: Cosine Function",
        "function": r"f(x)=\cos(bx)",
        "series": r"\cos(bx)=\sum_{n\ge 0}\frac{(-1)^n b^{2n}}{(2n)!}x^{2n}",
        "coefficients": r"a_{2n}=\frac{(-1)^n b^{2n}}{(2n)!}",
        "rank": r"D^{2n}",
        "plant_sum": [
            r"T\{\cos(bx)\}(s)=\sum_{n\ge 0} a_{2n}(-1)^{2n}D^{2n}\!\left(\frac{1}{s}\right)",
            r"=\sum_{n\ge 0}\frac{(-1)^n b^{2n}}{(2n)!}\frac{(2n)!}{s^{2n+1}}",
            r"=\frac{1}{s}\sum_{n\ge 0}\left(-\frac{b^2}{s^2}\right)^n",
        ],
        "closed_form": r"T\{\cos(bx)\}(s)=\frac{s}{s^2+b^2}",
        "condition": r"\left|\frac{b}{s}\right|<1",
    }

    cases["sin(b x)"] = {
        "title": "Case E: Sine Function",
        "function": r"f(x)=\sin(bx)",
        "series": r"\sin(bx)=\sum_{n\ge 0}\frac{(-1)^n b^{2n+1}}{(2n+1)!}x^{2n+1}",
        "coefficients": r"a_{2n+1}=\frac{(-1)^n b^{2n+1}}{(2n+1)!}",
        "rank": r"D^{2n+1}",
        "plant_sum": [
            r"T\{\sin(bx)\}(s)=\sum_{n\ge 0} a_{2n+1}(-1)^{2n+1}D^{2n+1}\!\left(\frac{1}{s}\right)",
            r"=\sum_{n\ge 0}\frac{(-1)^n b^{2n+1}}{(2n+1)!}\frac{(2n+1)!}{s^{2n+2}}",
            r"=\frac{b}{s^2}\sum_{n\ge 0}\left(-\frac{b^2}{s^2}\right)^n",
        ],
        "closed_form": r"T\{\sin(bx)\}(s)=\frac{b}{s^2+b^2}",
        "condition": r"\left|\frac{b}{s}\right|<1",
    }

    cases["x sin(bx)"] = {
        "title": "Case F: x sin(bx)",
        "function": r"f(x)=x\sin(bx)",
        "series": r"x\sin(bx)=\sum_{n\ge 0}\frac{(-1)^n b^{2n+1}}{(2n+1)!}x^{2n+2}",
        "coefficients": r"a_{2n+2}=\frac{(-1)^n b^{2n+1}}{(2n+1)!}",
        "rank": r"D^{2n+2}",
        "plant_sum": [
            r"T\{x\sin(bx)\}(s)=\sum_{n\ge 0} a_{2n+2}(-1)^{2n+2}D^{2n+2}\!\left(\frac{1}{s}\right)",
            r"=\sum_{n\ge 0}\frac{(-1)^n b^{2n+1}}{(2n+1)!}\frac{(2n+2)!}{s^{2n+3}}",
            r"=\frac{b}{s^3}\sum_{n\ge 0}(2n+2)\left(-\frac{b^2}{s^2}\right)^n",
            r"\sum_{n\ge 0}(2n+2)u^n=\frac{2}{(1-u)^2},\qquad u=-\frac{b^2}{s^2}",
        ],
        "closed_form": r"T\{x\sin(bx)\}(s)=\frac{2bs}{(s^2+b^2)^2}",
        "condition": r"s>|b|",
    }

    cases["x cos(bx)"] = {
        "title": "Case G: x cos(bx)",
        "function": r"f(x)=x\cos(bx)",
        "series": r"x\cos(bx)=\sum_{n\ge 0}\frac{(-1)^n b^{2n}}{(2n)!}x^{2n+1}",
        "coefficients": r"a_{2n+1}=\frac{(-1)^n b^{2n}}{(2n)!}",
        "rank": r"D^{2n+1}",
        "plant_sum": [
            r"T\{x\cos(bx)\}(s)=\sum_{n\ge 0} a_{2n+1}(-1)^{2n+1}D^{2n+1}\!\left(\frac{1}{s}\right)",
            r"=\sum_{n\ge 0}\frac{(-1)^n b^{2n}}{(2n)!}\frac{(2n+1)!}{s^{2n+2}}",
            r"=\frac{1}{s^2}\sum_{n\ge 0}(2n+1)\left(-\frac{b^2}{s^2}\right)^n",
            r"\sum_{n\ge 0}(2n+1)(-u)^n=\frac{1-u}{(1+u)^2},\qquad u=\frac{b^2}{s^2}",
        ],
        "closed_form": r"T\{x\cos(bx)\}(s)=\frac{s^2-b^2}{(s^2+b^2)^2}",
        "condition": r"s>|b|",
    }

    cases["cosh(b x)"] = {
        "title": "Case H1: Hyperbolic Cosine",
        "function": r"f(x)=\cosh(bx)",
        "series": r"\cosh(bx)=\sum_{n\ge 0}\frac{b^{2n}}{(2n)!}x^{2n}",
        "coefficients": r"a_{2n}=\frac{b^{2n}}{(2n)!}",
        "rank": r"D^{2n}",
        "plant_sum": [
            r"T\{\cosh(bx)\}(s)=\sum_{n\ge 0}\frac{b^{2n}}{(2n)!}(-D)^{2n}\!\left(\frac{1}{s}\right)",
            r"=\sum_{n\ge 0}\frac{b^{2n}}{(2n)!}\frac{(2n)!}{s^{2n+1}}",
            r"=\frac{1}{s}\sum_{n\ge 0}\left(\frac{b^2}{s^2}\right)^n",
        ],
        "closed_form": r"T\{\cosh(bx)\}(s)=\frac{s}{s^2-b^2}",
        "condition": r"\left|\frac{b}{s}\right|<1",
    }

    cases["sinh(b x)"] = {
        "title": "Case H2: Hyperbolic Sine",
        "function": r"f(x)=\sinh(bx)",
        "series": r"\sinh(bx)=\sum_{n\ge 0}\frac{b^{2n+1}}{(2n+1)!}x^{2n+1}",
        "coefficients": r"a_{2n+1}=\frac{b^{2n+1}}{(2n+1)!}",
        "rank": r"D^{2n+1}",
        "plant_sum": [
            r"T\{\sinh(bx)\}(s)=\sum_{n\ge 0}\frac{b^{2n+1}}{(2n+1)!}(-D)^{2n+1}\!\left(\frac{1}{s}\right)",
            r"=\sum_{n\ge 0}\frac{b^{2n+1}}{(2n+1)!}\frac{(2n+1)!}{s^{2n+2}}",
            r"=\frac{b}{s^2}\sum_{n\ge 0}\left(\frac{b^2}{s^2}\right)^n",
        ],
        "closed_form": r"T\{\sinh(bx)\}(s)=\frac{b}{s^2-b^2}",
        "condition": r"\left|\frac{b}{s}\right|<1",
    }

    cases["e^(a x) cos(bx)"] = {
        "title": "Shifted Case: e^{ax} cos(bx)",
        "function": r"f(x)=e^{ax}\cos(bx)",
        "series": r"\text{Use the shifted kernel } \frac{1}{s-a} \text{ instead of } \frac{1}{s}.",
        "coefficients": r"\text{The trigonometric coefficients remain unchanged; only the kernel is shifted.}",
        "rank": r"D^{2n}",
        "plant_sum": [
            r"T_a\{f\}(s)=\sum_{n\ge 0} a_{2n}(-D)^{2n}\!\left(\frac{1}{s-a}\right)",
            r"(-D)^n\!\left(\frac{1}{s-a}\right)=\frac{n!}{(s-a)^{n+1}}",
        ],
        "closed_form": r"T\{e^{ax}\cos(bx)\}(s)=\frac{s-a}{(s-a)^2+b^2}",
        "condition": r"\left|\frac{b}{s-a}\right|<1",
    }

    cases["e^(a x) sin(bx)"] = {
        "title": "Shifted Case: e^{ax} sin(bx)",
        "function": r"f(x)=e^{ax}\sin(bx)",
        "series": r"\text{Use the shifted kernel } \frac{1}{s-a} \text{ instead of } \frac{1}{s}.",
        "coefficients": r"\text{The sine coefficients remain unchanged; only the kernel is shifted.}",
        "rank": r"D^{2n+1}",
        "plant_sum": [
            r"T_a\{f\}(s)=\sum_{n\ge 0} a_{2n+1}(-D)^{2n+1}\!\left(\frac{1}{s-a}\right)",
            r"(-D)^n\!\left(\frac{1}{s-a}\right)=\frac{n!}{(s-a)^{n+1}}",
        ],
        "closed_form": r"T\{e^{ax}\sin(bx)\}(s)=\frac{b}{(s-a)^2+b^2}",
        "condition": r"\left|\frac{b}{s-a}\right|<1",
    }

    cases["sinc(b x) = sin(bx)/(b x)"] = {
        "title": "Case I: sinc(bx)",
        "function": r"f(x)=\mathrm{sinc}(bx)=\frac{\sin(bx)}{bx}",
        "series": r"\mathrm{sinc}(bx)=\sum_{n\ge 0}\frac{(-1)^n b^{2n}}{(2n+1)!}x^{2n}",
        "coefficients": r"a_{2n}=\frac{(-1)^n b^{2n}}{(2n+1)!}",
        "rank": r"D^{2n}",
        "plant_sum": [
            r"T\{\mathrm{sinc}(bx)\}(s)=\sum_{n\ge 0}a_{2n}(-D)^{2n}\!\left(\frac{1}{s}\right)",
            r"=\sum_{n\ge 0}\frac{(-1)^n b^{2n}}{(2n+1)!}\frac{(2n)!}{s^{2n+1}}",
            r"=\frac{1}{s}\sum_{n\ge 0}\frac{(-1)^n}{2n+1}\left(\frac{b}{s}\right)^{2n}",
            r"\sum_{n\ge 0}\frac{(-1)^n z^{2n+1}}{2n+1}=\arctan(z),\qquad z=\frac{b}{s}",
        ],
        "closed_form": r"T\{\mathrm{sinc}(bx)\}(s)=\frac{1}{b}\arctan\!\left(\frac{b}{s}\right)",
        "condition": r"\left|\frac{b}{s}\right|<1",
    }

    cases["(cos(bx)-1)/x"] = {
        "title": "Case J1: (cos(bx)-1)/x",
        "function": r"f(x)=\frac{\cos(bx)-1}{x}",
        "series": r"\frac{\cos(bx)-1}{x}=\sum_{n\ge 1}\frac{(-1)^n b^{2n}}{(2n)!}x^{2n-1}",
        "coefficients": r"a_{2n-1}=\frac{(-1)^n b^{2n}}{(2n)!}",
        "rank": r"D^{2n-1}",
        "plant_sum": [
            r"T\!\left\{\frac{\cos(bx)-1}{x}\right\}(s)=\sum_{n\ge 1}a_{2n-1}(-D)^{2n-1}\!\left(\frac{1}{s}\right)",
            r"=\sum_{n\ge 1}\frac{(-1)^n b^{2n}}{(2n)!}\frac{(2n-1)!}{s^{2n}}",
            r"=\frac{1}{2}\sum_{n\ge 1}\frac{1}{n}\left(-\frac{b^2}{s^2}\right)^n",
            r"\sum_{n\ge 1}\frac{r^n}{n}=-\ln(1-r)",
        ],
        "closed_form": r"T\!\left\{\frac{\cos(bx)-1}{x}\right\}(s)=-\frac{1}{2}\ln\!\left(1+\frac{b^2}{s^2}\right)",
        "condition": r"s>|b|",
    }

    cases["(cos(bx)-1)/x^2"] = {
        "title": "Case J2: (cos(bx)-1)/x^2",
        "function": r"f(x)=\frac{\cos(bx)-1}{x^2}",
        "series": r"\frac{\cos(bx)-1}{x^2}=\sum_{n\ge 1}\frac{(-1)^n b^{2n}}{(2n)!}x^{2n-2}",
        "coefficients": r"a_{2n-2}=\frac{(-1)^n b^{2n}}{(2n)!}",
        "rank": r"D^{2n-2}",
        "plant_sum": [
            r"T\!\left\{\frac{\cos(bx)-1}{x^2}\right\}(s)=\sum_{n\ge 1}a_{2n-2}(-D)^{2n-2}\!\left(\frac{1}{s}\right)",
            r"=\sum_{n\ge 1}\frac{(-1)^n b^{2n}}{(2n)!}\frac{(2n-2)!}{s^{2n-1}}",
            r"=s\sum_{n\ge 1}\frac{(-1)^n}{2n(2n-1)}\left(\frac{b^2}{s^2}\right)^n",
            r"\frac{1}{2n(2n-1)}=-\frac{1}{2n}+\frac{1}{2n-1}",
        ],
        "closed_form": r"T\!\left\{\frac{\cos(bx)-1}{x^2}\right\}(s)=\frac{s}{2}\ln\!\left(1+\frac{b^2}{s^2}\right)-b\arctan\!\left(\frac{b}{s}\right)",
        "condition": r"s>|b|",
    }

    cases["J0(k x)"] = {
        "title": "Case J3: Bessel Function J_0(kx)",
        "function": r"f(x)=J_0(kx)",
        "series": r"J_0(kx)=\sum_{n\ge 0}\frac{(-1)^n}{(n!)^2}\left(\frac{kx}{2}\right)^{2n}",
        "coefficients": r"a_{2n}=\frac{(-1)^n}{(n!)^2}\left(\frac{k}{2}\right)^{2n}",
        "rank": r"D^{2n}",
        "plant_sum": [
            r"T\{J_0(kx)\}(s)=\sum_{n\ge 0}a_{2n}(-D)^{2n}\!\left(\frac{1}{s}\right)",
            r"=\sum_{n\ge 0}\frac{(-1)^n}{(n!)^2}\left(\frac{k}{2}\right)^{2n}\frac{(2n)!}{s^{2n+1}}",
            r"=\frac{1}{s}\sum_{n\ge 0}\frac{(2n)!}{(n!)^2}\left(-\frac{k^2}{4s^2}\right)^n",
            r"\sum_{n\ge 0}\frac{(2n)!}{(n!)^2}z^n=\frac{1}{\sqrt{1-4z}}",
        ],
        "closed_form": r"T\{J_0(kx)\}(s)=\frac{1}{\sqrt{s^2+k^2}}",
        "condition": r"s>0",
    }

    cases["J_nu(kx)"] = {
        "title": "General Bessel Case: J_\\nu(kx)",
        "function": r"f(x)=J_\nu(kx)",
        "series": r"J_\nu(kx)=\sum_{n\ge 0}\frac{(-1)^n}{n!\,\Gamma(n+\nu+1)}\left(\frac{kx}{2}\right)^{2n+\nu}",
        "coefficients": r"a_{2n+\nu}=\frac{(-1)^n}{n!\,\Gamma(n+\nu+1)}\left(\frac{k}{2}\right)^{2n+\nu}",
        "rank": r"D^{2n+\nu}",
        "plant_sum": [
            r"T\{J_\nu(kx)\}(s)=\sum_{n\ge 0}a_{2n+\nu}(-D)^{2n+\nu}\!\left(\frac{1}{s}\right)",
            r"=\sum_{n\ge 0}\frac{(-1)^n}{n!\,\Gamma(n+\nu+1)}\left(\frac{k}{2}\right)^{2n+\nu}\frac{\Gamma(2n+\nu+1)}{s^{2n+\nu+1}}",
            r"\text{After summation, the planted series reproduces the classical Laplace–Bessel closed form.}",
        ],
        "closed_form": r"T\{J_\nu(kx)\}(s)=\frac{\left(\sqrt{s^2+k^2}-s\right)^\nu}{k^\nu\sqrt{s^2+k^2}}",
        "condition": r"s>0",
    }

    return cases



# ============================================================
# Fourier data
# ============================================================
def fourier_transform_data(choice, params):
    sigma_local = sp.Symbol("sigma", positive=True, real=True)
    omega_local = sp.Symbol("omega", real=True)
    t_local = sp.Symbol("t", real=True)

    s_complex = sigma_local + sp.I * omega_local
    s_bar = sigma_local - sp.I * omega_local

    if choice == "1":
        return {
            "regulated": 1 / s_complex + 1 / s_bar,
            "classical": r"2\pi\,\delta(\omega)",
            "comment": r"\mathcal{F}_\sigma\{1\}(\omega)=\frac{1}{\sigma+i\omega}+\frac{1}{\sigma-i\omega}=\frac{2\sigma}{\sigma^2+\omega^2}",
        }
        
        if choice == "e^(-a |t|)":
            a_val = to_expr(params["a"])
        return {
            "regulated": 2 * (a_val + sigma_local) / ((a_val + sigma_local)**2 + omega_local**2),
            "classical": 2 * a_val / (a_val**2 + omega_local**2),
            "comment": r"\mathcal{F}_\sigma\{e^{-a|t|}\}(\omega)=\frac{2(a+\sigma)}{(a+\sigma)^2+\omega^2}",
        }

    if choice == "cos(t)":
        return {
            "regulated": sigma_local / (sigma_local**2 + (omega_local - 1)**2)
            + sigma_local / (sigma_local**2 + (omega_local + 1)**2),
            "classical": r"\pi[\delta(\omega-1)+\delta(\omega+1)]",
            "comment": r"\mathcal{F}_\sigma\{\cos t\}(\omega)=\frac{\sigma}{\sigma^2+(\omega-1)^2}+\frac{\sigma}{\sigma^2+(\omega+1)^2}",
        }

    if choice == "sin(t)":
        return {
            "regulated": (sigma_local / sp.I)
            * (
                1 / (sigma_local**2 + (omega_local - 1)**2)
                - 1 / (sigma_local**2 + (omega_local + 1)**2)
            ),
            "classical": r"\frac{\pi}{i}[\delta(\omega-1)-\delta(\omega+1)]",
            "comment": r"\mathcal{F}_\sigma\{\sin t\}(\omega)=\frac{\sigma}{i}\left[\frac{1}{\sigma^2+(\omega-1)^2}-\frac{1}{\sigma^2+(\omega+1)^2}\right]",
        }

    if choice == "Gaussian e^(-a t^2)":
        a_val = to_expr(params["a"])
        return {
            "regulated": sp.Symbol(
                r"\frac{\sqrt{\pi}}{2\sqrt{a}}\left[e^{s^2/(4a)}\operatorname{erfc}\!\left(\frac{s}{2\sqrt{a}}\right)+e^{\bar s^2/(4a)}\operatorname{erfc}\!\left(\frac{\bar s}{2\sqrt{a}}\right)\right]"
            ),
            "classical": sp.sqrt(sp.pi / a_val) * sp.exp(-omega_local**2 / (4 * a_val)),
            "comment": r"\lim_{\sigma\to0^+}\mathcal{F}_\sigma\{e^{-a t^2}\}(\omega)=\sqrt{\frac{\pi}{a}}\,e^{-\omega^2/(4a)}",
        }

    if choice == "sinc(t) = sin(t)/t":
        return {
            "regulated": sp.Symbol(
                r"\arctan\!\left(\frac{1}{\sigma+i\omega}\right)+\arctan\!\left(\frac{1}{\sigma-i\omega}\right)"
            ),
            "classical": r"\pi\,\mathbf{1}_{(|\omega|<1)}",
            "comment": r"\mathcal{F}_\sigma\{\mathrm{sinc}\,t\}(\omega)=\arctan\!\left(\frac{1}{\sigma+i\omega}\right)+\arctan\!\left(\frac{1}{\sigma-i\omega}\right)",
        }

    if choice == "delta(t)":
        return {
            "regulated": sp.Integer(1),
            "classical": sp.Integer(1),
            "comment": r"\mathcal{F}_\sigma\{\delta(t)\}(\omega)=1",
        }

    return {
        "regulated": 1 / s_complex + 1 / s_bar,
        "classical": r"2\pi\,\delta(\omega)",
        "comment": r"\mathcal{F}_\sigma\{1\}(\omega)=\frac{2\sigma}{\sigma^2+\omega^2}",
    }


# ============================================================
# Mellin data
# ============================================================
def mellin_transform_data(choice, params):
    x_local = sp.Symbol("x", positive=True)
    s_local = sp.Symbol("s", real=True)
    t_local = sp.Symbol("t", positive=True)

    if choice == "e^(-x)":
        f = sp.exp(-x_local)
        return {
            "f": f,
            "series": f,
            "planted": r"\mathcal{M}_T\{e^{-x}\}(s,t)=\sum_{n\ge 0}\frac{(-1)^n}{n!}\frac{\Gamma(s+n)}{t^{s+n}}",
            "closed": sp.gamma(s_local) / (t_local + 1) ** s_local,
            "classical": sp.gamma(s_local),
        }

    if choice == "e^(i x)":
        f = sp.exp(sp.I * x_local)
        return {
            "f": f,
            "series": f,
            "planted": r"\mathcal{M}_T\{e^{ix}\}(s,t)=\sum_{n\ge 0}\frac{i^n}{n!}\frac{\Gamma(s+n)}{t^{s+n}}",
            "closed": sp.gamma(s_local) / (t_local - sp.I) ** s_local,
            "classical": sp.gamma(s_local) * sp.exp(sp.I * sp.pi * s_local / 2),
        }

    if choice == "cos(x)":
        f = sp.cos(x_local)
        return {
            "f": f,
            "series": f,
            "planted": r"\mathcal{M}_T\{\cos x\}(s,t)=\sum_{n\ge 0}\frac{(-1)^n\Gamma(s+2n)}{(2n)!t^{s+2n}}",
            "closed": sp.gamma(s_local) * (t_local**2 + 1) ** (-s_local / 2) * sp.cos(s_local * sp.atan(1 / t_local)),
            "classical": sp.gamma(s_local) * sp.cos(sp.pi * s_local / 2),
        }

    if choice == "sin(x)":
        f = sp.sin(x_local)
        return {
            "f": f,
            "series": f,
            "planted": r"\mathcal{M}_T\{\sin x\}(s,t)=\sum_{n\ge 0}\frac{(-1)^n\Gamma(s+2n+1)}{(2n+1)!t^{s+2n+1}}",
            "closed": sp.gamma(s_local) * (t_local**2 + 1) ** (-s_local / 2) * sp.sin(s_local * sp.atan(1 / t_local)),
            "classical": sp.gamma(s_local) * sp.sin(sp.pi * s_local / 2),
        }

    if choice == "(1+x)^(-a)":
        a_val = to_expr(params["a"])
        f = (1 + x_local) ** (-a_val)
        return {
            "f": f,
            "series": f,
            "planted": r"\mathcal{M}_T\{(1+x)^{-a}\}(s,t)=\Gamma(s)t^{-s}{}_2F_1\!\left(a,s;1;-\frac{1}{t}\right)",
            "closed": sp.gamma(s_local) * sp.hyper([a_val, s_local], [1], -1 / t_local) / t_local**s_local,
            "classical": sp.gamma(s_local) * sp.gamma(a_val - s_local) / sp.gamma(a_val),
        }

    if choice == "e^(-x^2)":
        f = sp.exp(-x_local**2)
        return {
            "f": f,
            "series": f,
            "planted": r"\mathcal{M}_T\{e^{-x^2}\}(s,t)=\sum_{n\ge 0}\frac{(-1)^n}{n!}\frac{\Gamma(s+2n)}{t^{s+2n}}",
            "closed": sp.Symbol(
                r"\Gamma(s)\,t^{-s}\,{}_2F_1\!\left(\frac{s}{2},\frac{s+1}{2};1;-\frac{4}{t^2}\right)"
            ),
            "classical": sp.gamma(s_local / 2) / 2,
        }

    return {
        "f": sp.exp(-x_local),
        "series": sp.exp(-x_local),
        "planted": r"\mathcal{M}_T\{e^{-x}\}(s,t)=\sum_{n\ge 0}\frac{(-1)^n}{n!}\frac{\Gamma(s+n)}{t^{s+n}}",
        "closed": sp.gamma(s_local) / (t_local + 1) ** s_local,
        "classical": sp.gamma(s_local),
    }


# ============================================================
# Hankel data
# ============================================================
def hankel_transform_data(choice):
    k_local = sp.Symbol("k", positive=True)
    t_local = sp.Symbol("t", positive=True)
    R_local = sp.Symbol("R", positive=True)
    r_local = sp.Symbol("r", positive=True)

    if choice == "e^(-t r)":
        return {
            "f": sp.exp(-t_local * r_local),
            "planted": r"\mathcal{H}\{e^{-tr}\}(k)=\sum_{m\ge 0}\frac{(-1)^m}{(m!)^2}\left(\frac{k}{2}\right)^{2m}(-\partial_t)^{2m+1}\left(\frac{1}{t}\right)",
            "closed": t_local / (t_local**2 + k_local**2) ** sp.Rational(3, 2),
        }

    if choice == "r e^(-t r)":
        return {
            "f": r_local * sp.exp(-t_local * r_local),
            "planted": r"\mathcal{H}\{re^{-tr}\}(k)=\sum_{m\ge 0}\frac{(-1)^m}{(m!)^2}\left(\frac{k}{2}\right)^{2m}(-\partial_t)^{2m+2}\left(\frac{1}{t}\right)",
            "closed": (2 * t_local**2 - k_local**2) / (t_local**2 + k_local**2) ** sp.Rational(5, 2),
        }

    if choice == "e^(-t r)/r":
        return {
            "f": sp.exp(-t_local * r_local) / r_local,
            "planted": r"\mathcal{H}_0\left\{\frac{e^{-tr}}{r}\right\}(k)=\sum_{m\ge 0}\frac{(-1)^m}{(m!)^2}\left(\frac{k}{2}\right)^{2m}(-\partial_t)^{2m}\left(\frac{1}{t}\right)",
            "closed": 1 / sp.sqrt(t_local**2 + k_local**2),
        }

    if choice == "indicator chi_[0,R]":
        return {
            "f": sp.Symbol(r"\chi_{[0,R]}(r)"),
            "planted": r"\mathcal{H}_T\{\chi_{[0,R]}\}(k,0)=\int_0^R rJ_0(kr)\,dr",
            "closed": R_local * sp.besselj(1, k_local * R_local) / k_local,
        }

    return {
        "f": sp.exp(-t_local * r_local),
        "planted": r"\mathcal{H}\{e^{-tr}\}(k)=\sum_{m\ge 0}\frac{(-1)^m}{(m!)^2}\left(\frac{k}{2}\right)^{2m}(-\partial_t)^{2m+1}\left(\frac{1}{t}\right)",
        "closed": t_local / (t_local**2 + k_local**2) ** sp.Rational(3, 2),
    }


# ============================================================
# Sidebar navigation
# ============================================================
section = st.sidebar.radio(
    "Go to:",
    [
        "Introduction",
        "Operator Laplace Transform",
        "Operator Fourier Transform",
        "Operator Mellin Transform",
        "Operator Hankel Transform",
        "Unified Transform Table",
        "Logarithmic Snapshot",
    ],
)


# ============================================================
# Header
# ============================================================
st.title("Operator Transform Laboratory")
st.subheader("Kernel Geometry and Planted Differential Structures")
st.caption("Interactive first version for Laplace, Fourier, Mellin, and Hankel transforms")
st.caption("by: Rida jamal badawi abu sokon")
st.caption("Amman-Jordan")
# ============================================================
# Section rendering
# ============================================================
if section == "Introduction":
    render_intro()

elif section == "Operator Laplace Transform":
    render_intro()
    st.divider()

    render_laplace_definition_and_proof()

    st.subheader("Interactive Laplace Case Explorer")

    cases = get_laplace_cases()
    selected_case = st.selectbox(
        "Choose a symbolic Laplace case",
        list(cases.keys()),
        index=0,
        key="laplace_case_explorer",
    )
    render_case(cases[selected_case])

    st.divider()
# ============================================================
# SECTION 3
# ============================================================
elif section == "Operator Fourier Transform":
    st.header("3. Operator Fourier Transform")

    
        
    st.markdown(r"""
In this section we extend the operator-based Laplace framework to the oscillatory Fourier domain. The key structural point is that the bilateral Fourier transform naturally produces a symmetric pair of complex Laplace kernels rather than a single one-sided kernel.

The regulator $e^{-\sigma |t|}$, with $\sigma>0$, is not an artificial addition.

It appears as the minimal symmetric damping required to make both half-axes integrable at once.

This yields a regulated Fourier--Laplace operator transform, and the classical Fourier transform is recovered in the distributional limit $\sigma \to 0^+$.
""")
      
    

    st.subheader("Definition")

    st.latex(r"s=\sigma+i\omega,\qquad \bar{s}=\sigma-i\omega,\qquad \sigma>0")

    st.latex(
        r"""
        \mathcal{F}_{\sigma}\{f\}(\omega)
        =
        \int_{-\infty}^{\infty} f(t)e^{-\sigma|t|}e^{-i\omega t}\,dt
        """
    )

    st.markdown("The planted Fourier operator acts on the symmetric kernel pair")
    st.latex(
        r"""
        \frac{1}{s},\qquad \frac{1}{\bar{s}}
        """
    )

    st.latex(
        r"""
        f(t)=\sum_{n=0}^{\infty} a_n t^n
        \quad\Longrightarrow\quad
        \mathcal{F}_{\sigma}\{f\}(\omega)
        =
        \sum_{n=0}^{\infty} a_n (-\partial_{\sigma})^n
        \left(
        \frac{1}{s}
        +
        (-1)^n\frac{1}{\bar{s}}
        \right)
        """
    )

    with st.expander("Detailed derivation: from one-sided Laplace structure to the regulated bilateral Fourier transform", expanded=True):
        st.markdown("**Step 1. Start from the formal bilateral Fourier integral**")
        st.latex(
            r"""
            \int_{-\infty}^{\infty} f(t)e^{-i\omega t}\,dt
            =
            \int_{0}^{\infty} f(t)e^{-i\omega t}\,dt
            +
            \int_{-\infty}^{0} f(t)e^{-i\omega t}\,dt
            """
        )

        st.markdown("**Step 2. Introduce the symmetric exponential regulator**")
        st.latex(r"w(t)=e^{-\sigma|t|},\qquad \sigma>0")
        st.latex(
            r"""
            \mathcal{F}_{\sigma}\{f\}(\omega)
            =
            \int_{-\infty}^{\infty} f(t)e^{-\sigma|t|}e^{-i\omega t}\,dt
            """
        )

        st.markdown("**Step 3. Split into positive and negative half-axes**")
        st.latex(
            r"""
            \mathcal{F}_{\sigma}\{f\}(\omega)
            =
            \int_{0}^{\infty} f(t)e^{-(\sigma+i\omega)t}\,dt
            +
            \int_{-\infty}^{0} f(t)e^{+\sigma t}e^{-i\omega t}\,dt
            """
        )

        st.markdown("**Step 4. Change variable on the negative half: \(u=-t\)**")
        st.latex(
            r"""
            \int_{-\infty}^{0} f(t)e^{+\sigma t}e^{-i\omega t}\,dt
            =
            \int_{0}^{\infty} f(-u)e^{-(\sigma-i\omega)u}\,du
            """
        )

        st.markdown("**Step 5. Identify the conjugate kernel pair**")
        st.latex(
            r"""
            s=\sigma+i\omega,\qquad \bar{s}=\sigma-i\omega
            """
        )
        st.latex(
            r"""
            \mathcal{F}_{\sigma}\{f\}(\omega)
            =
            \int_{0}^{\infty} f(t)e^{-st}\,dt
            +
            \int_{0}^{\infty} f(-t)e^{-\bar{s}t}\,dt
            """
        )

        st.markdown("**Step 6. Why the absolute value \(|t|\) appears**")
        st.latex(
            r"""
            e^{-\sigma|t|}
            =
            \begin{cases}
            e^{-\sigma t}, & t>0,\\
            e^{+\sigma t}, & t<0,
            \end{cases}
            """
        )

        st.markdown(
            """
            Hence the absolute value is a structural consequence of bilateral symmetric damping:
            it is exactly the choice that treats the positive and negative half-axes equally.
            """
        )

        st.markdown("**Step 7. Plant the monomial seed \(t^n\)**")
        st.latex(
            r"""
            \int_{0}^{\infty} t^n e^{-st}\,dt=\frac{\Gamma(n+1)}{s^{n+1}}
            \qquad\text{and}\qquad
            \int_{0}^{\infty} t^n e^{-\bar{s}t}\,dt=\frac{\Gamma(n+1)}{\bar{s}^{\,n+1}}
            """
        )

        st.latex(
            r"""
            \mathcal{F}_{\sigma}\{t^n\}(\omega)
            =
            \Gamma(n+1)\left(
            \frac{1}{s^{n+1}}
            +
            (-1)^n\frac{1}{\bar{s}^{\,n+1}}
            \right)
            """
        )

        st.markdown("**Step 8. Extend by Maclaurin linearity**")
        st.latex(
            r"""
            f(t)=\sum_{n=0}^{\infty} a_n t^n
            \quad\Longrightarrow\quad
            \mathcal{F}_{\sigma}\{f\}(\omega)
            =
            \sum_{n=0}^{\infty}
            a_n\Gamma(n+1)
            \left(
            \frac{1}{s^{n+1}}
            +
            (-1)^n\frac{1}{\bar{s}^{\,n+1}}
            \right)
            """
        )

        st.markdown("**Step 9. Distributional classical limit**")
        st.latex(
            r"""
            \lim_{\sigma\to0^+}\mathcal{F}_{\sigma}\{f\}(\omega)=\mathcal{F}\{f\}(\omega)
            """
        )

    st.subheader("Interactive Fourier Case Explorer")

    fourier_case = st.selectbox(
        "Choose a symbolic Fourier case",
        [
            "1",
            r"e^{-a|t|}",
            r"cos(t)",
            r"sin(t)",
            r"e^{-a t^2}",
            r"{sinc}(t)=sin(t)/t",
            r"\delta(t)",
            r"\chi_{[-R,R]}(t)",
        ],
        key="fourier_case_detailed"
    )

    sigma_sym = sp.Symbol("sigma", positive=True, real=True)
    omega_sym = sp.Symbol("omega", real=True)
    t_sym = sp.Symbol("t", real=True)
    a_sym = sp.Symbol("a", positive=True, real=True)
    R_sym = sp.Symbol("R", positive=True, real=True)

    s_sym = sigma_sym + sp.I * omega_sym
    sbar_sym = sigma_sym - sp.I * omega_sym

    if fourier_case == "1":
        f_expr = sp.Integer(1)
        series_text = r"1=\sum_{n\ge 0} a_n t^n,\qquad a_0=1,\; a_{n>0}=0."
        coeff_text = r"a_0=1,\qquad a_{n>0}=0."
        rank_text = r"0"
        plant_steps = [
            r"\mathcal{F}_{\sigma}\{1\}(\omega)=\frac{1}{s}+\frac{1}{\bar s}",
            r"=\frac{1}{\sigma+i\omega}+\frac{1}{\sigma-i\omega}",
            r"=\frac{2\sigma}{\sigma^2+\omega^2}"
        ]
        closed_text = r"\mathcal{F}_{\sigma}\{1\}(\omega)=\frac{2\sigma}{\sigma^2+\omega^2}"
        classical_text = r"\lim_{\sigma\to0^+}\mathcal{F}_{\sigma}\{1\}(\omega)=2\pi\delta(\omega)"
        extra_note = r"\lim_{\sigma\to0^+}\frac{1}{\pi}\frac{\sigma}{\sigma^2+\omega^2}=\delta(\omega)"

    elif fourier_case == r"e^{-a|t|}":
        f_expr = sp.exp(-a_sym * sp.Abs(t_sym))
        series_text = r"e^{-a|t|}=\sum_{n=0}^{\infty}\frac{(-a|t|)^n}{n!}"
        coeff_text = r"a_n=\frac{(-a)^n}{n!}"
        rank_text = r"D^n"
        plant_steps = [
            r"\mathcal{F}_{\sigma}\{e^{-a|t|}\}(\omega)=\sum_{n=0}^{\infty}\frac{(-a)^n}{n!}\,\mathcal{F}_{\sigma}\{|t|^n\}(\omega)",
            r"\mathcal{F}_{\sigma}\{|t|^n\}(\omega)=\Gamma(n+1)\left[\frac{1}{s^{n+1}}+\frac{1}{\bar s^{\,n+1}}\right]",
            r"\mathcal{F}_{\sigma}\{e^{-a|t|}\}(\omega)=\sum_{n=0}^{\infty}\frac{(-a)^n}{n!}\Gamma(n+1)\left[\frac{1}{s^{n+1}}+\frac{1}{\bar s^{\,n+1}}\right]",
            r"\Gamma(n+1)=n!\quad\Longrightarrow\quad \mathcal{F}_{\sigma}\{e^{-a|t|}\}(\omega)=\sum_{n=0}^{\infty}(-a)^n\left[\frac{1}{s^{n+1}}+\frac{1}{\bar s^{\,n+1}}\right]",
            r"=\frac{1}{s}\sum_{n=0}^{\infty}\left(-\frac{a}{s}\right)^n+\frac{1}{\bar s}\sum_{n=0}^{\infty}\left(-\frac{a}{\bar s}\right)^n",
            r"=\frac{1}{s+a}+\frac{1}{\bar s+a}",
            r"=\frac{1}{a+\sigma+i\omega}+\frac{1}{a+\sigma-i\omega}",
            r"=\frac{2(a+\sigma)}{(a+\sigma)^2+\omega^2}"
        ]
        closed_text = r"\mathcal{F}_{\sigma}\{e^{-a|t|}\}(\omega)=\frac{2(a+\sigma)}{(a+\sigma)^2+\omega^2}"
        classical_text = r"\lim_{\sigma\to0^+}\mathcal{F}_{\sigma}\{e^{-a|t|}\}(\omega)=\frac{2a}{a^2+\omega^2}"
        extra_note = ""

    elif fourier_case == r"cos(t)":
        f_expr = sp.cos(t_sym)
        series_text = r"\cos t=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n)!}\,t^{2n}"
        coeff_text = r"a_{2n}=\frac{(-1)^n}{(2n)!}"
        rank_text = r"2n"
        plant_steps = [
            r"\mathcal{F}_{\sigma}\{\cos t\}(\omega)=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n)!}\,\mathcal{F}_{\sigma}\{t^{2n}\}(\omega)",
            r"\mathcal{F}_{\sigma}\{t^{2n}\}(\omega)=\Gamma(2n+1)\left[\frac{1}{s^{2n+1}}+\frac{1}{\bar s^{\,2n+1}}\right]",
            r"\Gamma(2n+1)=(2n)!\quad\Longrightarrow\quad \mathcal{F}_{\sigma}\{\cos t\}(\omega)=\sum_{n=0}^{\infty}(-1)^n\left[\frac{1}{s^{2n+1}}+\frac{1}{\bar s^{\,2n+1}}\right]",
            r"\sum_{n=0}^{\infty}\frac{(-1)^n}{s^{2n+1}}=\frac{1}{s}\sum_{n=0}^{\infty}\left(-\frac{1}{s^2}\right)^n=\frac{s}{s^2+1}",
            r"\sum_{n=0}^{\infty}\frac{(-1)^n}{\bar s^{\,2n+1}}=\frac{\bar s}{\bar s^{\,2}+1}",
            r"\mathcal{F}_{\sigma}\{\cos t\}(\omega)=\frac{s}{s^2+1}+\frac{\bar s}{\bar s^{\,2}+1}",
            r"\frac{s}{s^2+1}=\frac12\left(\frac{1}{\sigma+i(\omega-1)}+\frac{1}{\sigma+i(\omega+1)}\right)",
            r"\frac{\bar s}{\bar s^{\,2}+1}=\frac12\left(\frac{1}{\sigma-i(\omega-1)}+\frac{1}{\sigma-i(\omega+1)}\right)",
            r"\mathcal{F}_{\sigma}\{\cos t\}(\omega)=\frac{\sigma}{\sigma^2+(\omega-1)^2}+\frac{\sigma}{\sigma^2+(\omega+1)^2}"
        ]
        closed_text = r"\mathcal{F}_{\sigma}\{\cos t\}(\omega)=\frac{\sigma}{\sigma^2+(\omega-1)^2}+\frac{\sigma}{\sigma^2+(\omega+1)^2}"

        classical_text = r"\lim_{\sigma\to0^+}\mathcal{F}_{\sigma}\{\cos t\}(\omega)=\pi\,[\delta(\omega-1)+\delta(\omega+1)]"

        extra_note = r"\text{Using }\lim_{\sigma\to0^+}\frac{\sigma}{(\omega-a)^2+\sigma^2}=\pi\,\delta(\omega-a)"

    elif fourier_case == r"sin(t)":
        f_expr = sp.sin(t_sym)
        series_text = r"\sin t=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n+1)!}\,t^{2n+1}"
        coeff_text = r"a_{2n+1}=\frac{(-1)^n}{(2n+1)!}"
        rank_text = r"2n+1"
        plant_steps = [
            r"\mathcal{F}_{\sigma}\{\sin t\}(\omega)=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n+1)!}\,\mathcal{F}_{\sigma}\{t^{2n+1}\}(\omega)",
            r"\mathcal{F}_{\sigma}\{t^{2n+1}\}(\omega)=\Gamma(2n+2)\left[\frac{1}{s^{2n+2}}-\frac{1}{\bar s^{\,2n+2}}\right]",
            r"\Gamma(2n+2)=(2n+1)!\quad\Longrightarrow\quad \mathcal{F}_{\sigma}\{\sin t\}(\omega)=\sum_{n=0}^{\infty}(-1)^n\left[\frac{1}{s^{2n+2}}-\frac{1}{\bar s^{\,2n+2}}\right]",
            r"\sum_{n=0}^{\infty}\frac{(-1)^n}{s^{2n+2}}=\frac{1}{s^2+1},\qquad \sum_{n=0}^{\infty}\frac{(-1)^n}{\bar s^{\,2n+2}}=\frac{1}{\bar s^{\,2}+1}",
            r"\mathcal{F}_{\sigma}\{\sin t\}(\omega)=\frac{1}{s^2+1}-\frac{1}{\bar s^{\,2}+1}",
            r"\frac{1}{s^2+1}=\frac{1}{2i}\left[\frac{1}{\sigma+i(\omega-1)}-\frac{1}{\sigma+i(\omega+1)}\right]",
            r"\frac{1}{\bar s^{\,2}+1}=-\frac{1}{2i}\left[\frac{1}{\sigma-i(\omega-1)}-\frac{1}{\sigma-i(\omega+1)}\right]",
            r"\mathcal{F}_{\sigma}\{\sin t\}(\omega)=\frac{\sigma}{i}\left[\frac{1}{\sigma^2+(\omega-1)^2}-\frac{1}{\sigma^2+(\omega+1)^2}\right]"
        ]
        closed_text = r"\mathcal{F}_{\sigma}\{\sin t\}(\omega)=\frac{\sigma}{i}\left[\frac{1}{\sigma^2+(\omega-1)^2}-\frac{1}{\sigma^2+(\omega+1)^2}\right]"
        classical_text = r"\lim_{\sigma\to0^+}\mathcal{F}_{\sigma}\{\sin t\}(\omega)=\frac{\pi}{i}\,[\delta(\omega-1)-\delta(\omega+1)]"
        extra_note = ""

    elif fourier_case == r"e^{-a t^2}":
        f_expr = sp.exp(-a_sym * t_sym**2)
        series_text = r"e^{-a t^2}=\sum_{n=0}^{\infty}\frac{(-a)^n}{n!}\,t^{2n}"
        coeff_text = r"a_{2n}=\frac{(-a)^n}{n!}"
        rank_text = r"2n"
        plant_steps = [
            r"\mathcal{F}_{\sigma}\{e^{-a t^2}\}(\omega)=\sum_{n=0}^{\infty}\frac{(-a)^n}{n!}\,\mathcal{F}_{\sigma}\{t^{2n}\}(\omega)",
            r"\mathcal{F}_{\sigma}\{t^{2n}\}(\omega)=\Gamma(2n+1)\left[\frac{1}{s^{2n+1}}+\frac{1}{\bar s^{\,2n+1}}\right]",
            r"\mathcal{F}_{\sigma}\{e^{-a t^2}\}(\omega)=\sum_{n=0}^{\infty}\frac{(-a)^n}{n!}\Gamma(2n+1)\left[\frac{1}{s^{2n+1}}+\frac{1}{\bar s^{\,2n+1}}\right]",
            r"\Gamma(2n+1)=(2n)!\quad\Longrightarrow\quad \mathcal{F}_{\sigma}\{e^{-a t^2}\}(\omega)=\sum_{n=0}^{\infty}\frac{(-a)^n(2n)!}{n!}\left[\frac{1}{s^{2n+1}}+\frac{1}{\bar s^{\,2n+1}}\right]",
            r"\sum_{n=0}^{\infty}\frac{(-a)^n(2n)!}{n!}\frac{1}{z^{2n+1}}\ \text{resums to}\ \frac{\sqrt{\pi}}{2\sqrt{a}}\,e^{z^2/(4a)}\operatorname{erfc}\!\left(\frac{z}{2\sqrt a}\right)",
            r"\mathcal{F}_{\sigma}\{e^{-a t^2}\}(\omega)=\frac{\sqrt{\pi}}{2\sqrt a}\left[e^{s^2/(4a)}\operatorname{erfc}\!\left(\frac{s}{2\sqrt a}\right)+e^{\bar s^{\,2}/(4a)}\operatorname{erfc}\!\left(\frac{\bar s}{2\sqrt a}\right)\right]",
            r"\text{Taking } \sigma\to0^+ \text{ combines the two complementary-error terms and yields the Gaussian Fourier law.}"
        ]
        closed_text = r"\mathcal{F}_{\sigma}\{e^{-a t^2}\}(\omega)=\frac{\sqrt{\pi}}{2\sqrt a}\left[e^{s^2/(4a)}\operatorname{erfc}\!\left(\frac{s}{2\sqrt a}\right)+e^{\bar s^{\,2}/(4a)}\operatorname{erfc}\!\left(\frac{\bar s}{2\sqrt a}\right)\right]"
        classical_text = r"\lim_{\sigma\to0^+}\mathcal{F}_{\sigma}\{e^{-a t^2}\}(\omega)=\sqrt{\frac{\pi}{a}}\,e^{-\omega^2/(4a)}"
        extra_note = ""

    elif fourier_case == r"{sinc}(t)=sin(t)/t":
        f_expr = sp.sin(t_sym) / t_sym
        series_text = r"{sinc}(t)=\frac{\sin t}{t}=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n+1)!}\,t^{2n}"
        coeff_text = r"a_{2n}=\frac{(-1)^n}{(2n+1)!}"
        rank_text = r"2n"
        plant_steps = [
            r"\mathcal{F}_{\sigma}\{\mathrm{sinc}(t)\}(\omega)=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n+1)!}\,\mathcal{F}_{\sigma}\{t^{2n}\}(\omega)",
            r"\mathcal{F}_{\sigma}\{t^{2n}\}(\omega)=\Gamma(2n+1)\left[\frac{1}{s^{2n+1}}+\frac{1}{\bar{s}^{\,2n+1}}\right]",
            r"\frac{\Gamma(2n+1)}{(2n+1)!}=\frac{(2n)!}{(2n+1)!}=\frac{1}{2n+1}",
            r"\mathcal{F}_{\sigma}\{\mathrm{sinc}(t)\}(\omega)=\sum_{n=0}^{\infty}\frac{(-1)^n}{2n+1}\left[\frac{1}{s^{2n+1}}+\frac{1}{\bar{s}^{\,2n+1}}\right]",
            r"\sum_{n=0}^{\infty}\frac{(-1)^n z^{2n+1}}{2n+1}=\arctan(z)",
            r"\mathcal{F}_{\sigma}\{\mathrm{sinc}(t)\}(\omega)=\arctan\!\left(\frac{1}{s}\right)+\arctan\!\left(\frac{1}{\bar{s}}\right)",
            r"\text{Equivalently}",
            r"\mathcal{F}_{\sigma}\{\mathrm{sinc}(t)\}(\omega)=\arctan\!\left(\frac{1}{\sigma+i\omega}\right)+\arctan\!\left(\frac{1}{\sigma-i\omega}\right)",
            r"\text{Conversion from complex to real form}",
            r"\frac{1}{\sigma+i\omega}=\frac{\sigma-i\omega}{\sigma^2+\omega^2}",
            r"\text{The two terms are complex conjugates of each other, hence their sum is twice the real part.}",
            r"\arctan\!\left(\frac{1}{\sigma+i\omega}\right)+\arctan\!\left(\frac{1}{\sigma-i\omega}\right)=2\,\Re\!\left[\arctan\!\left(\frac{1}{\sigma+i\omega}\right)\right]",
            r"\text{After simplification, one obtains an equivalent real-valued representation suitable for physical interpretation.}",
           
        ]
        closed_text = r"\mathcal{F}_{\sigma}\{\mathrm{sinc}(t)\}(\omega)=\arctan\!\left(\frac{1}{\sigma+i\omega}\right)+\arctan\!\left(\frac{1}{\sigma-i\omega}\right)"
        classical_text = r"\lim_{\sigma\to0^+}\mathcal{F}_{\sigma}\{\mathrm{sinc}(t)\}(\omega)=\pi\,\mathbf{1}_{(|\omega|<1)},\qquad \mathcal{F}\{\mathrm{sinc}(t)\}(\pm1)=\frac{\pi}{2}"
        extra_note = ""
        

    elif fourier_case == r"\delta(t)":
        f_expr = sp.Symbol(r"\delta(t)")
        series_text = r"\delta(t)\ \text{is handled distributionally rather than by a Maclaurin series.}"
        coeff_text = r"\text{Mass concentrated at } t=0."
        rank_text = r"\text{distributional}"
        plant_steps = [
            r"\mathcal{F}_{\sigma}\{\delta(t)\}(\omega)=\int_{0}^{\infty}\delta(t)e^{-\sigma t}e^{-i\omega t}\,dt+\int_{0}^{\infty}\delta(-t)e^{-\sigma t}e^{+i\omega t}\,dt",
            r"\int_{0}^{\infty}\delta(t)\phi(t)\,dt=\frac12\phi(0),\qquad \delta(-t)=\delta(t)",
            r"\mathcal{F}_{\sigma}\{\delta(t)\}(\omega)=\frac12\cdot 1+\frac12\cdot 1=1"
        ]
        closed_text = r"\mathcal{F}_{\sigma}\{\delta(t)\}(\omega)=1"
        classical_text = r"\mathcal{F}\{\delta(t)\}(\omega)=1"
        extra_note = ""

    elif fourier_case == r"\chi_{[-R,R]}(t)":
        f_expr = sp.Symbol(r"\chi_{[-R,R]}(t)")
        series_text = r"\chi_{[-R,R]}(t)\ \text{is interval-supported and is best handled directly from the regulated bilateral form.}"
        coeff_text = r"\text{This case is finite-support rather than Maclaurin-planted.}"
        rank_text = r"\text{finite interval case}"
        plant_steps = [
            r"\mathcal{F}_{\sigma}\{\chi_{[-R,R]}\}(\omega)=\int_{-R}^{R}e^{-\sigma|t|}e^{-i\omega t}\,dt",
            r"=\int_{0}^{R}e^{-(\sigma+i\omega)t}\,dt+\int_{0}^{R}e^{-(\sigma-i\omega)t}\,dt",
            r"=\frac{1-e^{-sR}}{s}+\frac{1-e^{-\bar s R}}{\bar s}"
        ]
        closed_text = r"\mathcal{F}_{\sigma}\{\chi_{[-R,R]}\}(\omega)=\frac{1-e^{-sR}}{s}+\frac{1-e^{-\bar s R}}{\bar s}"
        classical_text = r"\lim_{\sigma\to0^+}\mathcal{F}_{\sigma}\{\chi_{[-R,R]}\}(\omega)=\frac{1-e^{-i\omega R}}{i\omega}-\frac{1-e^{i\omega R}}{i\omega}=\frac{2\sin(\omega R)}{\omega}=2R\,\operatorname{sinc}(\omega R)"
        extra_note = ""

    st.markdown("### Function")
    if isinstance(f_expr, sp.Basic):
        st.latex(r"f(t)=" + safe_latex(f_expr))
    else:
        st.latex(str(f_expr))

    st.markdown("### Series")
    st.latex(series_text)

    st.markdown("### Coefficients")
    st.latex(coeff_text)

    st.markdown("### Rank")
    st.latex(rank_text)

    st.markdown("### Plant \\& Sum")
    for step in plant_steps:
        st.latex(step)

    st.markdown("### Closed regulated form")
    st.latex(closed_text)

    st.markdown("### Classical limit")
    st.latex(classical_text)

    if extra_note:
        st.markdown("### Remark")
        st.latex(extra_note)

# ==================================================
# SECTION 4
#=================================================
elif section == "Operator Mellin Transform":
    st.header("4. Operator Mellin Transform")

    st.markdown(r"""
    In this section we introduce an operator version of the Mellin transform built by
    planting Maclaurin coefficients inside a Gamma-weighted kernel. The resulting framework
    reproduces the classical Mellin table directly from a planted series, demonstrating that
    the operator viewpoint can be both unifying and computationally efficient.
    """)

    st.latex(r"""
    x^n \;\longmapsto\; (-\partial_t)^{\,n+s-1}\!\left(\frac{1}{t}\right)
    = \frac{\Gamma(n+s)}{t^{\,n+s}}
    """)

    st.subheader("Differential Derivation of the Mellin Integral Form (Planting Method)")

    st.markdown("### 1. Differential Planting Foundation")
    st.markdown(r"""
    Let $f(x)$ be an analytic function expanded as a Maclaurin series:
    """)
    st.latex(r"f(x)=\sum_{n=0}^{\infty} a_n x^n")

    st.markdown(r"""
    The Mellin transform is defined in its differential planting form by
    """)
    st.latex(r"""
    MT\{f\}(s,t)=\sum_{n=0}^{\infty} a_n (-\partial_t)^{\,s+n-1}\!\left(\frac{1}{t}\right),
    \qquad t>0
    """)

    st.markdown(r"""
    Each term $a_n x^n$ of the series is planted as a fractional derivative of order
    $(s+n-1)$ applied to the seed function $(1/t)$.
    """)

    st.markdown("### 2. Fractional Derivative Identity")
    st.markdown(r"""
    For $(Re(\alpha)>-1)$, the fractional derivative of $(1/t)$ satisfies
    """)
    st.latex(r"""
    (-\partial_t)^\alpha\!\left(\frac{1}{t}\right)=\frac{\Gamma(\alpha+1)}{t^{\alpha+1}}
    """)

    st.markdown(r"Substituting $(\alpha=s+n-1)$ gives")
    st.latex(r"""
    (-\partial_t)^{\,s+n-1}\!\left(\frac{1}{t}\right)=\frac{\Gamma(s+n)}{t^{s+n}}
    """)

    st.markdown(r"Hence the differential planting law becomes")
    st.latex(r"""
    MT\{f\}(s,t)=\sum_{n=0}^{\infty} a_n \frac{\Gamma(s+n)}{t^{s+n}}
    """)

    st.markdown("### 3. Transition to the Integral Representation")
    st.markdown(r"Using the Gamma integral identity")
    st.latex(r"""
    \frac{\Gamma(s+n)}{t^{s+n}}=\int_0^\infty x^{s+n-1}e^{-tx}\,dx,
    \qquad t>0
    """)

    st.markdown(r"we substitute this into the planted series:")
    st.latex(r"""
    MT\{f\}(s,t)=\sum_{n=0}^{\infty} a_n \int_0^\infty x^{s+n-1}e^{-tx}\,dx
    """)

    st.markdown("### 4. Interchanging Sum and Integral")
    st.markdown(r"""
    Under absolute convergence (for standard analytic functions), the sum and the integral may
    be interchanged (Tonelli–Fubini theorem):
    """)
    st.latex(r"""
    MT\{f\}(s,t)=\int_0^\infty \left(\sum_{n=0}^{\infty} a_n x^n\right)x^{s-1}e^{-tx}\,dx
    """)

    st.markdown(r"Recognizing the internal series as $f(x)$ yields the canonical integral form:")
    st.latex(r"""
    MT\{f\}(s,t)=\int_0^\infty f(x)\,x^{s-1}e^{-tx}\,dx
    """)

    st.markdown("### 5. Classical Mellin Limit")
    st.markdown(r"Taking the limit as $(t\to0^+)$ removes the exponential regulator:")
    st.latex(r"""
    \lim_{t\to0^+} MT\{f\}(s,t)=\int_0^\infty f(x)x^{s-1}\,dx
    """)

    st.markdown(r"""
    which is precisely the classical Mellin transform.

    Perfect agreement between the planted differential form and the classical integral form
    confirms the internal consistency.
    """)

    st.markdown("### 7. Interpretation")
    st.markdown(r"""
    This derivation shows that the Mellin integral form is not a primitive definition, but a
    natural consequence of the differential planting law. The exponential regulator $(e^{-tx})$
    emerges automatically from the internal structure of the fractional derivative of $(1/t)$.
    Thus, the differential approach reconstructs the Mellin transform entirely from planted
    derivative dynamics, bridging the discrete hierarchy of derivatives with the continuous
    integral hierarchy governed by the Gamma function.
    """)

    st.markdown(r"""
    **Additional note.** The exponential regulator $(e^{-tx})$ appearing in the integral form
    of the Mellin operator is not an externally added convergence factor, but rather an
    intrinsic byproduct of the differential planting sequence. This shows that the exponential
    damping in Mellin’s transform can be reconstructed endogenously from the internal structure
    of the fractional derivatives acting on $(1/t)$.
    """)

    st.subheader("Summary of the Operator Mellin Transform")

    st.markdown("### 10.1 Integral form")
    st.markdown(r"The classical Mellin transform is defined as")
    st.latex(r"""
    \mathcal{M}\{f(x)\}(s)=\int_0^\infty f(x)x^{s-1}\,dx
    """)
    st.markdown(r"valid whenever the integral converges.")
    st.markdown(r"In the operator framework, we introduce an exponential regulator:")
    st.latex(r"""
    MT\{f(x)\}(s,t)=\int_0^\infty f(x)x^{s-1}e^{-tx}\,dx,\qquad t>0
    """)

    st.markdown("### 10.2 Series form (planting law)")
    st.markdown(r"If $(f(x))$ has a Maclaurin expansion")
    st.latex(r"f(x)=\sum_{n=0}^{\infty} a_n x^n")
    st.markdown(r"then the transform is obtained by planting each term:")
    st.latex(r"""
    MT\{f\}(s,t)=\sum_{n=0}^{\infty} a_n \frac{\Gamma(s+n)}{t^{s+n}}
    """)
    st.markdown(r"""
    This is the series planting law: each \(x^n\) contributes a gamma factor $(\Gamma(s+n))$
    and a regulator $(t^{-(s+n)})$.
    """)

    st.subheader("Interactive Mellin Case Explorer")

    mellin_case = st.selectbox(
        "Choose a symbolic Mellin case",
        [
            r"e^{-x}",
            r"e^{ix}",
            r"cos x",
            r"sin x",
            r"(1+x)^{-a}",
            r"e^{-x^2}",
        ],
        key="mellin_case_detailed"
    )

    s_sym = sp.Symbol("s", complex=True)
    treg_sym = sp.Symbol("t", positive=True, real=True)
    x_sym = sp.Symbol("x", positive=True, real=True)
    a_sym = sp.Symbol("a", positive=True, real=True)

    if mellin_case == r"e^{-x}":
        f_expr = sp.exp(-x_sym)
        series_text = r"e^{-x}=\sum_{n=0}^{\infty}\frac{(-1)^n}{n!}x^n"
        coeff_text = r"a_n=\frac{(-1)^n}{n!}"
        rank_text = r"n"
        plant_steps = [
            r"MT\{e^{-x}\}(s,t)=\sum_{n=0}^{\infty}\frac{(-1)^n}{n!}\frac{\Gamma(s+n)}{t^{s+n}},\qquad t>0",
            r"\Gamma(s+n)=\Gamma(s)(s)_n,\qquad (s)_n=\frac{\Gamma(s+n)}{\Gamma(s)}",
            r"MT\{e^{-x}\}(s,t)=\Gamma(s)t^{-s}\sum_{n=0}^{\infty}\frac{(s)_n}{n!}\left(-\frac{1}{t}\right)^n",
            r"(1+z)^{-s}=\sum_{n=0}^{\infty}\frac{(s)_n}{n!}(-z)^n",
            r"\sum_{n=0}^{\infty}\frac{(s)_n}{n!}\left(-\frac{1}{t}\right)^n=\left(1+\frac{1}{t}\right)^{-s}",
            r"MT\{e^{-x}\}(s,t)=\Gamma(s)t^{-s}\left(1+\frac{1}{t}\right)^{-s}",
            r"=\frac{\Gamma(s)}{(t+1)^s}",
        ]
        closed_text = r"MT\{e^{-x}\}(s,t)=\frac{\Gamma(s)}{(t+1)^s}"
        classical_text = r"\lim_{t\to0^+}MT\{e^{-x}\}(s,t)=\Gamma(s)"
        extra_note = r"\int_0^\infty e^{-x}x^{s-1}\,dx=\Gamma(s)"

    elif mellin_case == r"e^{ix}":
        f_expr = sp.exp(sp.I * x_sym)
        series_text = r"e^{ix}=\sum_{n=0}^{\infty}\frac{i^n}{n!}x^n"
        coeff_text = r"a_n=\frac{i^n}{n!}"
        rank_text = r"n"
        plant_steps = [
            r"MT\{e^{ix}\}(s,t)=\sum_{n=0}^{\infty}\frac{i^n}{n!}\frac{\Gamma(s+n)}{t^{s+n}},\qquad t>0",
            r"\Gamma(s+n)=\Gamma(s)(s)_n",
            r"MT\{e^{ix}\}(s,t)=\Gamma(s)t^{-s}\sum_{n=0}^{\infty}\frac{(s)_n}{n!}\left(\frac{i}{t}\right)^n",
            r"(1-z)^{-s}=\sum_{n=0}^{\infty}\frac{(s)_n}{n!}z^n",
            r"\sum_{n=0}^{\infty}\frac{(s)_n}{n!}\left(\frac{i}{t}\right)^n=\left(1-\frac{i}{t}\right)^{-s}",
            r"MT\{e^{ix}\}(s,t)=\Gamma(s)t^{-s}\left(1-\frac{i}{t}\right)^{-s}",
            r"=\frac{\Gamma(s)}{(t-i)^s},\qquad t>0",
            r"t-i=re^{-i\theta},\qquad r=\sqrt{t^2+1},\qquad \theta=\arctan\!\left(\frac{1}{t}\right)",
            r"\frac{1}{(t-i)^s}=r^{-s}e^{is\theta}",
            r"MT\{e^{ix}\}(s,t)=\Gamma(s)r^{-s}e^{is\theta}=\Gamma(s)r^{-s}\big[\cos(s\theta)+i\sin(s\theta)\big]",
            r"MT\{\cos x\}(s,t)=\Gamma(s)r^{-s}\cos(s\theta),\qquad MT\{\sin x\}(s,t)=\Gamma(s)r^{-s}\sin(s\theta)",
            r"\text{Now take the limit }t\to0^+:\qquad r=\sqrt{t^2+1}\to1,\qquad \theta=\arctan\!\left(\frac{1}{t}\right)\to\frac{\pi}{2}",
        r"MT\{\cos x\}(s,t)\longrightarrow \Gamma(s)\cos\!\left(\frac{\pi s}{2}\right)",
        r"MT\{\sin x\}(s,t)\longrightarrow \Gamma(s)\sin\!\left(\frac{\pi s}{2}\right)"

        ]
  
        closed_text = r"MT\{e^{ix}\}(s,t)=\frac{\Gamma(s)}{(t-i)^s}"
        classical_text = r"\lim_{t\to0^+}MT\{e^{ix}\}(s,t)=\Gamma(s)e^{i\pi s/2}"
        extra_note = r"MT\{e^{ix}\}(s,t)=\int_0^\infty x^{s-1}e^{-tx}e^{ix}\,dx=\int_0^\infty x^{s-1}e^{-(t-i)x}\,dx=\frac{\Gamma(s)}{(t-i)^s}"

       

    elif mellin_case == r"cos x":
        f_expr = sp.cos(x_sym)
        series_text = r"\cos x=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n)!}x^{2n}"
        coeff_text = r"a_{2n}=\frac{(-1)^n}{(2n)!}"
        rank_text = r"2n"
        plant_steps = [
            r"MT\{\cos x\}(s,t)=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n)!}\frac{\Gamma(s+2n)}{t^{s+2n}}",
            r"\Gamma(s+2n)=\Gamma(s)\,2^{2n}\left(\frac{s}{2}\right)_n\left(\frac{s+1}{2}\right)_n",
            r"(2n)!=2^{2n}\left(\frac12\right)_n n!",
            r"MT\{\cos x\}(s,t)=\Gamma(s)t^{-s}\,{}_2F_1\!\left(\frac{s}{2},\frac{s+1}{2};\frac12;-\frac{1}{t^2}\right)",
            r"\text{Equivalently, }MT\{\cos x\}(s,t)=\Gamma(s)(t^2+1)^{-s/2}\cos\!\left(s\arctan\!\frac{1}{t}\right)",
        ]
        closed_text = r"MT\{\cos x\}(s,t)=\Gamma(s)(t^2+1)^{-s/2}\cos\!\left(s\arctan\!\frac{1}{t}\right)"
        classical_text = r"\lim_{t\to0^+}MT\{\cos x\}(s,t)=\Gamma(s)\cos\!\left(\frac{\pi s}{2}\right),\qquad 0<\Re(s)<1"
        extra_note = r"\int_0^\infty x^{s-1}\cos x\,dx=\Gamma(s)\cos\!\left(\frac{\pi s}{2}\right),\qquad 0<\Re(s)<1"

    elif mellin_case == r"sin x":
        f_expr = sp.sin(x_sym)
        series_text = r"\sin x=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n+1)!}x^{2n+1}"
        coeff_text = r"a_{2n+1}=\frac{(-1)^n}{(2n+1)!}"
        rank_text = r"2n+1"
        plant_steps = [
            r"MT\{\sin x\}(s,t)=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n+1)!}\frac{\Gamma(s+2n+1)}{t^{s+2n+1}}",
            r"\Gamma(s+2n+1)=\Gamma(s+1)\,2^{2n}\left(\frac{s+1}{2}\right)_n\left(\frac{s+2}{2}\right)_n",
            r"(2n+1)!=2^{2n}\left(\frac32\right)_n n!",
            r"MT\{\sin x\}(s,t)=\Gamma(s+1)t^{-(s+1)}\,{}_2F_1\!\left(\frac{s+1}{2},\frac{s+2}{2};\frac32;-\frac{1}{t^2}\right)",
            r"\text{Equivalently, }MT\{\sin x\}(s,t)=\Gamma(s)(t^2+1)^{-s/2}\sin\!\left(s\arctan\!\frac{1}{t}\right)",
        ]
        closed_text = r"MT\{\sin x\}(s,t)=\Gamma(s)(t^2+1)^{-s/2}\sin\!\left(s\arctan\!\frac{1}{t}\right)"
        classical_text = r"\lim_{t\to0^+}MT\{\sin x\}(s,t)=\Gamma(s)\sin\!\left(\frac{\pi s}{2}\right),\qquad 0<\Re(s)<1"
        extra_note = r"\int_0^\infty x^{s-1}\sin x\,dx=\Gamma(s)\sin\!\left(\frac{\pi s}{2}\right),\qquad 0<\Re(s)<1"

    elif mellin_case == r"(1+x)^{-a}":
        f_expr = (1 + x_sym) ** (-a_sym)
        series_text = r"(1+x)^{-a}=\sum_{n=0}^{\infty}(-1)^n\binom{a+n-1}{n}x^n=\sum_{n=0}^{\infty}(-1)^n\frac{(a)_n}{n!}x^n"
        coeff_text = r"a_n=(-1)^n\binom{a+n-1}{n}=(-1)^n\frac{(a)_n}{n!}"
        rank_text = r"n"
        plant_steps = [
            r"MT\{(1+x)^{-a}\}(s,t)=\sum_{n=0}^{\infty}(-1)^n\binom{a+n-1}{n}\frac{\Gamma(s+n)}{t^{s+n}}",
            r"=\Gamma(s)t^{-s}\sum_{n=0}^{\infty}\frac{(a)_n(s)_n}{(1)_n\,n!}\left(-\frac1t\right)^n",
            r"MT\{(1+x)^{-a}\}(s,t)=\Gamma(s)t^{-s}\,{}_2F_1\!\left(a,s;1;-\frac1t\right)",
            r"\text{At }t=1\text{, Euler's evaluation yields }{}_2F_1(a,s;1;-1)=\frac{\Gamma(a-s)}{\Gamma(a)\Gamma(1-s)}",
            r"MT\{(1+x)^{-a}\}(s,1)=\frac{\Gamma(s)\Gamma(a-s)}{\Gamma(a)},\qquad 0<\Re(s)<a",
            r"\int_0^\infty x^{s-1}(1+x)^{-a}\,dx=B(s,a-s)=\frac{\Gamma(s)\Gamma(a-s)}{\Gamma(a)}",
        ]
        closed_text = r"MT\{(1+x)^{-a}\}(s,t)=\Gamma(s)t^{-s}\,{}_2F_1\!\left(a,s;1;-\frac1t\right)"
        classical_text = r"\lim_{t\to0^+}MT\{(1+x)^{-a}\}(s,t)=\frac{\Gamma(s)\Gamma(a-s)}{\Gamma(a)},\qquad 0<\Re(s)<a"
        extra_note = r"\text{At }t=1:\quad MT\{(1+x)^{-a}\}(s,1)=\frac{\Gamma(s)\Gamma(a-s)}{\Gamma(a)}"

    elif mellin_case == r"e^{-x^2}":
        f_expr = sp.exp(-x_sym**2)
        series_text = r"e^{-x^2}=\sum_{n=0}^{\infty}\frac{(-1)^n}{n!}x^{2n}"
        coeff_text = r"a_{2n}=\frac{(-1)^n}{n!}"
        rank_text = r"2n"
        plant_steps = [
            r"MT\{e^{-x^2}\}(s,t)=\sum_{n=0}^{\infty}\frac{(-1)^n}{n!}\frac{\Gamma(s+2n)}{t^{s+2n}}",
            r"\Gamma(s+2n)=\Gamma(s)\,2^{2n}\left(\frac{s}{2}\right)_n\left(\frac{s+1}{2}\right)_n",
            r"MT\{e^{-x^2}\}(s,t)=\Gamma(s)t^{-s}\,{}_2F_1\!\left(\frac{s}{2},\frac{s+1}{2};1;-\frac{4}{t^2}\right)",
            r"\text{Taking the limit as }t\to0^+\text{ yields}",
            r"\lim_{t\to0^+}MT\{e^{-x^2}\}(s,t)=\frac12\,\Gamma\!\left(\frac{s}{2}\right)",
            r"\int_0^\infty x^{s-1}e^{-x^2}\,dx=\frac12\,\Gamma\!\left(\frac{s}{2}\right),\qquad \Re(s)>0",
        ]
        closed_text = r"MT\{e^{-x^2}\}(s,t)=\Gamma(s)t^{-s}\,{}_2F_1\!\left(\frac{s}{2},\frac{s+1}{2};1;-\frac{4}{t^2}\right)"
        classical_text = r"\lim_{t\to0^+}MT\{e^{-x^2}\}(s,t)=\frac12\,\Gamma\!\left(\frac{s}{2}\right),\qquad \Re(s)>0"
        extra_note = r"\int_0^\infty x^{s-1}e^{-x^2}\,dx=\frac12\,\Gamma\!\left(\frac{s}{2}\right)"

    st.markdown("### Function")
    st.latex(r"f(x)=" + sp.latex(f_expr))

    st.markdown("### Series")
    st.latex(series_text)

    st.markdown("### Coefficients")
    st.latex(coeff_text)

    st.markdown("### Rank")
    st.latex(rank_text)

    st.markdown("### Plant \\& Sum")
    for step in plant_steps:
        st.latex(step)

    st.markdown("### Closed regulated form")
    st.latex(closed_text)

    st.markdown("### Classical Mellin limit")
    st.latex(classical_text)

    if extra_note:
        st.markdown("### Cross-check / Remark")
        st.latex(extra_note)

    st.markdown("### Comparison Table (Expanded)")
    st.latex(r"""
    \begin{array}{|c|c|c|c|}
    \hline
    \text{Function }f(x) & \text{Series expansion} & \text{Planted operator form} & \text{Classical Mellin result}\\
    \hline
    e^{-x} &
    \sum_{n=0}^{\infty}\frac{(-1)^n}{n!}x^n &
    \sum_{n=0}^{\infty}\frac{(-1)^n}{n!}\frac{\Gamma(s+n)}{t^{s+n}} &
    \Gamma(s)\\
    \hline
    (1+x)^{-a} &
    \sum_{n=0}^{\infty}(-1)^n\binom{a+n-1}{n}x^n &
    \sum_{n=0}^{\infty}(-1)^n\binom{a+n-1}{n}\frac{\Gamma(s+n)}{t^{s+n}} &
    \frac{\Gamma(s)\Gamma(a-s)}{\Gamma(a)}\\
    \hline
    \cos x &
    \sum_{n=0}^{\infty}\frac{(-1)^n x^{2n}}{(2n)!} &
    \sum_{n=0}^{\infty}\frac{(-1)^n\Gamma(s+2n)}{(2n)!t^{s+2n}} &
    \Gamma(s)\cos\!\left(\frac{\pi s}{2}\right)\\
    \hline
    \sin x &
    \sum_{n=0}^{\infty}\frac{(-1)^n x^{2n+1}}{(2n+1)!} &
    \sum_{n=0}^{\infty}\frac{(-1)^n\Gamma(s+2n+1)}{(2n+1)!t^{s+2n+1}} &
    \Gamma(s)\sin\!\left(\frac{\pi s}{2}\right)\\
    \hline
    e^{-x^2} &
    \sum_{n=0}^{\infty}\frac{(-1)^n}{n!}x^{2n} &
    \sum_{n=0}^{\infty}\frac{(-1)^n}{n!}\frac{\Gamma(s+2n)}{t^{s+2n}} &
    \frac12\Gamma\!\left(\frac{s}{2}\right)\\
    \hline
    \end{array}
    """)

elif section == "Operator Hankel Transform":
    st.header("5. Operator Hankel Transform")

    st.markdown(r"""
    In this section we introduce the Hankel Operator Transform as a natural counterpart of
    the Laplace, Fourier, and Mellin operator transforms. The key idea is a planting rule
    that maps radial monomials $(r^n)$ into differential operators acting on the damping
    seed $(1/t)$. This produces explicit series expansions and recovers closed forms for
    exponential radial inputs.
    """)

    st.latex(r"""
    r^n \;\longmapsto\; (-\partial_t)^{\,n+\nu}\!\left(\frac{1}{t}\right)
    \;=\;
    \frac{\Gamma(n+\nu+1)}{t^{\,n+\nu+1}}
    """)

    st.subheader("Definition and Planting Rule")

    st.markdown(r"""
    The Hankel Operator Transform of order $(\nu)$ is defined by
    """)
    st.latex(r"""
    H_T\{f(r)\}(k)
    =
    \int_0^\infty f(r)\,J_\nu(kr)\,r\,dr,
    \qquad \nu>-\frac12
    """)

    st.markdown(r"""
    We now derive the regulated integral form from the planted-operator point of view.
    """)

    st.markdown("### Step 1. General planted operator")
    st.markdown(r"""
    Guided by the radial power appearing in the Bessel series together with the measure
    $(r\,dr)$, we take the planted operator basis to be $(r^{2m+\nu+1})$:
    """)
    st.latex(r"""
    H_T\{e^{-tr}\}(k)
    =
    \sum_{m=0}^{\infty}
    a_m\,(-\partial_t)^{\,2m+\nu+1}\!\left(\frac{1}{t}\right)
    """)

    st.markdown("### Step 2. Coefficients from the Bessel kernel")
    st.markdown(r"""
    Using the standard Bessel-series expansion
    """)
    st.latex(r"""
    J_\nu(kr)
    =
    \sum_{m=0}^{\infty}
    \frac{(-1)^m}{m!\,\Gamma(m+\nu+1)}
    \left(\frac{kr}{2}\right)^{2m+\nu},
    """)
    st.markdown(r"""
    the planted coefficients are
    """)
    st.latex(r"""
    a_m
    =
    \frac{(-1)^m}{m!\,\Gamma(m+\nu+1)}
    \left(\frac{k}{2}\right)^{2m+\nu}.
    """)

    st.markdown("### Step 3. Operator-to-integral substitution")
    st.markdown(r"""
    For any integer $(n\ge 0)$, the Laplace identity gives
    """)
    st.latex(r"""
    (-\partial_t)^n\!\left(\frac{1}{t}\right)
    =
    \int_0^\infty e^{-tr}r^n\,dr.
    """)

    st.markdown(r"""
    Applying this with $(n=2m+\nu+1)$ yields
    """)
    st.latex(r"""
    H_T\{e^{-tr}\}(k)
    =
    \sum_{m=0}^{\infty}
    a_m
    \int_0^\infty e^{-tr}r^{2m+\nu+1}\,dr.
    """)

    st.markdown("### Step 4. Recognition of the Bessel kernel")
    st.markdown(r"""
    Reassembling the Bessel series inside the integral gives
    """)
    st.latex(r"""
    \sum_{m=0}^{\infty} a_m r^{2m+\nu}
    =
    J_\nu(kr).
    """)

    st.markdown(r"""
    Therefore the planted expression collapses to the regulated Hankel integral form
    """)
    st.latex(r"""
    H_T\{e^{-tr}\}(k)
    =
    \int_0^\infty e^{-tr}J_\nu(kr)\,r\,dr.
    """)

    st.markdown("### Interpretation of the exponential regulator")
    st.markdown(r"""
    The exponential regulator \(e^{-tr}\) arises endogenously from the differential
    planting mechanism: each \(t\)-derivative acting on \(1/t\) generates one more power
    of \(r\) inside a Laplace-type kernel. Thus the planted differential hierarchy is
    analytically equivalent to the classical Bessel-integral representation.
    """)

    st.subheader("Interactive Hankel Case Explorer")

    hankel_case = st.selectbox(
        "Choose a symbolic Hankel case",
        [
            r"e^{-tr}",
            r"r e^{-tr}",
            r"\frac{e^{-tr}}{r}",
            r"\chi_{[0,R]}(r)",
        ],
        key="hankel_case_detailed",
    )

    nu_sym = sp.Symbol("nu", real=True)
    k_sym = sp.Symbol("k", positive=True, real=True)
    t_sym = sp.Symbol("t", positive=True, real=True)
    r_sym = sp.Symbol("r", positive=True, real=True)
    R_sym = sp.Symbol("R", positive=True, real=True)

    if hankel_case == r"e^{-tr}":
        f_expr = sp.exp(-t_sym * r_sym)

        series_text = r"J_0(kr)=\sum_{m=0}^{\infty}\frac{(-1)^m}{(m!)^2}\left(\frac{kr}{2}\right)^{2m}"
        coeff_text = r"a_m=\frac{(-1)^m}{(m!)^2}\left(\frac{k}{2}\right)^{2m}"
        rank_text = r"2m+1"

        plant_steps = [
            r"H\{e^{-tr}\}(k)=\sum_{m=0}^{\infty}a_m\,(-\partial_t)^{2m+1}\!\left(\frac{1}{t}\right)",
            r"J_0(kr)=\sum_{m=0}^{\infty}\frac{(-1)^m}{(m!)^2}\left(\frac{kr}{2}\right)^{2m}",
            r"H\{e^{-tr}\}(k)=\sum_{m=0}^{\infty}\frac{(-1)^m}{(m!)^2}\left(\frac{k}{2}\right)^{2m}(-\partial_t)^{2m+1}\!\left(\frac{1}{t}\right)",
            r"(-\partial_t)^{2m+1}\!\left(\frac{1}{t}\right)=\frac{(2m+1)!}{t^{2m+2}}",
            r"H\{e^{-tr}\}(k)=\sum_{m=0}^{\infty}\frac{(-1)^m(2m+1)!}{(m!)^2}\left(\frac{k}{2}\right)^{2m}\frac{1}{t^{2m+2}}",
            r"H\{e^{-tr}\}(k)=\frac{1}{t^2}\sum_{m=0}^{\infty}(2m+1)\binom{2m}{m}\left(-\frac{k^2}{4t^2}\right)^m",
            r"\text{Let } z=-\frac{k^2}{4t^2}",
            r"(2m+1)\binom{2m}{m}=2m\binom{2m}{m}+\binom{2m}{m}",
            r"\sum_{m=0}^{\infty}\binom{2m}{m}z^m=(1-4z)^{-1/2}",
            r"\sum_{m=0}^{\infty}2m\binom{2m}{m}z^m=\frac{4z}{(1-4z)^{3/2}}",
            r"H\{e^{-tr}\}(k)=\frac{1}{t^2}\left[\frac{4z}{(1-4z)^{3/2}}+\frac{1}{\sqrt{1-4z}}\right]",
            r"\frac{4z}{(1-4z)^{3/2}}+\frac{1}{(1-4z)^{1/2}}=\frac{1}{(1-4z)^{3/2}}",
            r"H\{e^{-tr}\}(k)=\frac{1}{t^2(1-4z)^{3/2}}",
            r"1-4z=1+\frac{k^2}{t^2}=\frac{t^2+k^2}{t^2}",
            r"(1-4z)^{3/2}=\frac{(t^2+k^2)^{3/2}}{t^3}",
            r"H\{e^{-tr}\}(k)=\frac{1}{t^2}\cdot\frac{t^3}{(t^2+k^2)^{3/2}}=\frac{t}{(t^2+k^2)^{3/2}}",
        ]

        closed_text = r"H\{e^{-tr}\}(k)=\frac{t}{(t^2+k^2)^{3/2}}"
        classical_text = r"\text{This is already the regulated closed form for } t>0"
        extra_note = r"\int_0^\infty e^{-tr}J_0(kr)\,r\,dr=\frac{t}{(t^2+k^2)^{3/2}}"

    elif hankel_case == r"r e^{-tr}":
        f_expr = r_sym * sp.exp(-t_sym * r_sym)

        series_text = r"J_0(kr)=\sum_{m=0}^{\infty}\frac{(-1)^m}{(m!)^2}\left(\frac{kr}{2}\right)^{2m}"
        coeff_text = r"a_m=\frac{(-1)^m}{(m!)^2}\left(\frac{k}{2}\right)^{2m}"
        rank_text = r"2m+2"

        plant_steps = [
            r"H\{re^{-tr}\}(k)=\sum_{m=0}^{\infty}a_m\,(-\partial_t)^{2m+2}\!\left(\frac{1}{t}\right)",
            r"H\{re^{-tr}\}(k)=\sum_{m=0}^{\infty}\frac{(-1)^m}{(m!)^2}\left(\frac{k}{2}\right)^{2m}(-\partial_t)^{2m+2}\!\left(\frac{1}{t}\right)",
            r"(-\partial_t)^{2m+2}\!\left(\frac{1}{t}\right)=\frac{(2m+2)!}{t^{2m+3}}",
            r"H\{re^{-tr}\}(k)=\sum_{m=0}^{\infty}\frac{(-1)^m(2m+2)!}{(m!)^2}\left(\frac{k}{2}\right)^{2m}\frac{1}{t^{2m+3}}",
            r"H\{re^{-tr}\}(k)=\frac{1}{t^3}\sum_{m=0}^{\infty}\frac{(2m+2)!}{(m!)^2}\left(-\frac{k^2}{4t^2}\right)^m",
            r"\text{Let } z=-\frac{k^2}{4t^2}",
            r"\frac{(2m+2)!}{(m!)^2}=n^2\binom{2n}{n}\ \text{ with } n=m+1",
            r"F(z)=\sum_{n=0}^{\infty}\binom{2n}{n}z^n=(1-4z)^{-1/2}",
            r"F'(z)=2(1-4z)^{-3/2},\qquad F''(z)=12(1-4z)^{-5/2}",
            r"F'(z)+zF''(z)=\sum_{n=1}^{\infty}n^2\binom{2n}{n}z^{n-1}",
            r"F'(z)+zF''(z)=\frac{2(1+2z)}{(1-4z)^{5/2}}",
            r"H\{re^{-tr}\}(k)=\frac{1}{t^3}\cdot\frac{2(1+2z)}{(1-4z)^{5/2}}",
            r"1+2z=1-\frac{k^2}{2t^2}=\frac{2t^2-k^2}{2t^2}",
            r"1-4z=1+\frac{k^2}{t^2}=\frac{t^2+k^2}{t^2}",
            r"(1-4z)^{5/2}=\frac{(t^2+k^2)^{5/2}}{t^5}",
            r"H\{re^{-tr}\}(k)=\frac{1}{t^3}\cdot\frac{2\left(\frac{2t^2-k^2}{2t^2}\right)}{(t^2+k^2)^{5/2}/t^5}=\frac{2t^2-k^2}{(t^2+k^2)^{5/2}}",
        ]

        closed_text = r"H\{re^{-tr}\}(k)=\frac{2t^2-k^2}{(t^2+k^2)^{5/2}}"
        classical_text = r"\text{This is the regulated closed form for } t>0"
        extra_note = r"H\{re^{-tr}\}(k)=-\frac{d}{dt}\left(\frac{t}{(t^2+k^2)^{3/2}}\right)=\frac{2t^2-k^2}{(t^2+k^2)^{5/2}}"

    elif hankel_case == r"\frac{e^{-tr}}{r}":
        f_expr = sp.exp(-t_sym * r_sym) / r_sym

        series_text = r"J_0(kr)=\sum_{m=0}^{\infty}\frac{(-1)^m}{(m!)^2}\left(\frac{kr}{2}\right)^{2m}"
        coeff_text = r"a_m=\frac{(-1)^m}{(m!)^2}\left(\frac{k}{2}\right)^{2m}"
        rank_text = r"2m"

        plant_steps = [
            r"H_0\!\left\{\frac{e^{-tr}}{r}\right\}(k)=\sum_{m=0}^{\infty}a_m\,(-\partial_t)^{2m}\!\left(\frac{1}{t}\right)",
            r"H_0\!\left\{\frac{e^{-tr}}{r}\right\}(k)=\sum_{m=0}^{\infty}\frac{(-1)^m}{(m!)^2}\left(\frac{k}{2}\right)^{2m}(-\partial_t)^{2m}\!\left(\frac{1}{t}\right)",
            r"(-\partial_t)^{2m}\!\left(\frac{1}{t}\right)=\frac{(2m)!}{t^{2m+1}}",
            r"H_0\!\left\{\frac{e^{-tr}}{r}\right\}(k)=\sum_{m=0}^{\infty}\frac{(-1)^m(2m)!}{(m!)^2}\left(\frac{k}{2}\right)^{2m}\frac{1}{t^{2m+1}}",
            r"\frac{(2m)!}{(m!)^2}=\binom{2m}{m}",
            r"H_0\!\left\{\frac{e^{-tr}}{r}\right\}(k)=\frac{1}{t}\sum_{m=0}^{\infty}\binom{2m}{m}\left(-\frac{k^2}{4t^2}\right)^m",
            r"\text{Let } z=-\frac{k^2}{4t^2}",
            r"\sum_{m=0}^{\infty}\binom{2m}{m}z^m=(1-4z)^{-1/2}",
            r"H_0\!\left\{\frac{e^{-tr}}{r}\right\}(k)=\frac{1}{t}(1-4z)^{-1/2}",
            r"1-4z=1+\frac{k^2}{t^2}=\frac{t^2+k^2}{t^2}",
            r"(1-4z)^{-1/2}=\frac{t}{\sqrt{t^2+k^2}}",
            r"H_0\!\left\{\frac{e^{-tr}}{r}\right\}(k)=\frac{1}{t}\cdot\frac{t}{\sqrt{t^2+k^2}}=\frac{1}{\sqrt{t^2+k^2}}",
        ]

        closed_text = r"H_0\!\left\{\frac{e^{-tr}}{r}\right\}(k)=\frac{1}{\sqrt{t^2+k^2}},\qquad t>0,\ k\ge 0"
        classical_text = r"\text{This is the regulated closed form for order } \nu=0"
        extra_note = r"\text{This is exactly the seed generated by the central-binomial series.}"

    elif hankel_case == r"\chi_{[0,R]}(r)":
        f_expr = sp.Piecewise((1, (r_sym >= 0) & (r_sym <= R_sym)), (0, True))

        series_text = r"J_\nu(kr)=\sum_{m=0}^{\infty}\frac{(-1)^m}{m!\,\Gamma(m+\nu+1)}\left(\frac{kr}{2}\right)^{2m+\nu}"
        coeff_text = r"a_m=\frac{(-1)^m}{m!\,\Gamma(m+\nu+1)}\left(\frac{k}{2}\right)^{2m+\nu}"
        rank_text = r"2m+\nu+1"

        plant_steps = [
            r"H_T\{\chi_{[0,R]}\}(k,0)=\int_0^R r\,J_\nu(kr)\,dr",
            r"J_\nu(kr)=\sum_{m=0}^{\infty}\frac{(-1)^m}{m!\,\Gamma(m+\nu+1)}\left(\frac{kr}{2}\right)^{2m+\nu}",
            r"H_T\{\chi_{[0,R]}\}(k,0)=\sum_{m=0}^{\infty}\frac{(-1)^m}{m!\,\Gamma(m+\nu+1)}\left(\frac{k}{2}\right)^{2m+\nu}\int_0^R r^{2m+\nu+1}\,dr",
            r"\int_0^R r^{2m+\nu+1}\,dr=\frac{R^{2m+\nu+2}}{2m+\nu+2}",
            r"H_T\{\chi_{[0,R]}\}(k,0)=\sum_{m=0}^{\infty}\frac{(-1)^m}{m!\,\Gamma(m+\nu+1)}\left(\frac{k}{2}\right)^{2m+\nu}\frac{R^{2m+\nu+2}}{2m+\nu+2}",
            r"\frac{1}{(2m+\nu+2)\Gamma(m+\nu+1)}=\frac{1}{2\,\Gamma(m+\nu+2)}",
            r"H_T\{\chi_{[0,R]}\}(k,0)=\frac{R^{\nu+2}}{2}\left(\frac{k}{2}\right)^\nu \sum_{m=0}^{\infty}\frac{(-1)^m}{m!\,\Gamma(m+\nu+2)}\left(\frac{kR}{2}\right)^{2m}",
            r"J_{\nu+1}(kR)=\sum_{m=0}^{\infty}\frac{(-1)^m}{m!\,\Gamma(m+\nu+2)}\left(\frac{kR}{2}\right)^{2m+\nu+1}",
            r"J_{\nu+1}(kR)=\left(\frac{kR}{2}\right)^{\nu+1}\sum_{m=0}^{\infty}\frac{(-1)^m}{m!\,\Gamma(m+\nu+2)}\left(\frac{kR}{2}\right)^{2m}",
            r"\sum_{m=0}^{\infty}\frac{(-1)^m}{m!\,\Gamma(m+\nu+2)}\left(\frac{kR}{2}\right)^{2m}=\frac{J_{\nu+1}(kR)}{\left(\frac{kR}{2}\right)^{\nu+1}}",
            r"H_T\{\chi_{[0,R]}\}(k,0)=\frac{R^{\nu+2}}{2}\left(\frac{k}{2}\right)^\nu \frac{J_{\nu+1}(kR)}{\left(\frac{kR}{2}\right)^{\nu+1}}",
            r"H_T\{\chi_{[0,R]}\}(k,0)=\frac{R}{k}J_{\nu+1}(kR)",
        ]

        closed_text = r"H_T\{\chi_{[0,R]}\}(k,0)=\frac{R}{k}J_{\nu+1}(kR)"
        classical_text = r"\text{For }\nu=0:\quad H_T\{\chi_{[0,R]}\}(k,0)=\frac{R}{k}J_1(kR)"
        extra_note = r"\lim_{R\to\infty}\frac{R}{k}J_1(kR)=\frac{\delta(k)}{k}\quad\text{(distributionally, for }\nu=0\text{)}"

    st.markdown("### Function")
    if hankel_case == r"\chi_{[0,R]}(r)":
        st.latex(r"f(r)=\chi_{[0,R]}(r)=\begin{cases}1,&0\le r\le R\\0,&r>R\end{cases}")
    else:
        st.latex(r"f(r)=" + sp.latex(f_expr))

    st.markdown("### Series")
    st.latex(series_text)

    st.markdown("### Coefficients")
    st.latex(coeff_text)

    st.markdown("### Rank")
    st.latex(rank_text)

    st.markdown("### Plant \\& Sum")
    for step in plant_steps:
        st.latex(step)

    st.markdown("### Closed regulated form")
    st.latex(closed_text)

    st.markdown("### Classical / special limit")
    st.latex(classical_text)

    if extra_note:
        st.markdown("### Cross-check / Remark")
        st.latex(extra_note)

    st.subheader("General Case")
    st.markdown(r"""
    For the general radial input $(f(r)=r^m e^{-tr})$, the planted Hankel form becomes
    """)
    st.latex(r"""
    H_T\{r^m e^{-tr}\}(k,t)
    =
    \sum_{p=0}^{\infty}
    \frac{(-1)^p}{p!\,\Gamma(p+\nu+1)}
    \left(\frac{k}{2}\right)^{2p+\nu}
    \frac{\Gamma(m+2p+\nu+2)}{t^{m+2p+\nu+2}}.
    """)

    st.markdown(r"""
    In particular:
    """)
    st.latex(r"""
    m=0 \Longrightarrow \frac{t}{(t^2+k^2)^{3/2}},
    \qquad
    m=1 \Longrightarrow \frac{2t^2-k^2}{(t^2+k^2)^{5/2}},
    \qquad
    m=-1 \Longrightarrow \frac{1}{\sqrt{t^2+k^2}}
    """)

    st.markdown("### Summary Table")
    st.latex(r"""
    \begin{array}{|c|c|c|c|}
    \hline
    \text{Function }f(r) & \text{Series expansion} & \text{Planted operator form} & \text{Closed form} \\
    \hline
    e^{-tr} &
    \sum_{m=0}^{\infty}\frac{(-1)^m}{(m!)^2}\left(\frac{kr}{2}\right)^{2m} &
    \sum_{m=0}^{\infty}\frac{(-1)^m}{(m!)^2}\left(\frac{k}{2}\right)^{2m}
    (-\partial_t)^{2m+1}\left(\frac{1}{t}\right) &
    \frac{t}{(t^2+k^2)^{3/2}}
    \\
    \hline
    r e^{-tr} &
    \sum_{m=0}^{\infty}\frac{(-1)^m}{(m!)^2}\left(\frac{kr}{2}\right)^{2m} &
    \sum_{m=0}^{\infty}\frac{(-1)^m}{(m!)^2}\left(\frac{k}{2}\right)^{2m}
    (-\partial_t)^{2m+2}\left(\frac{1}{t}\right) &
    \frac{2t^2-k^2}{(t^2+k^2)^{5/2}}
    \\
    \hline
    \frac{e^{-tr}}{r} &
    \sum_{m=0}^{\infty}\frac{(-1)^m}{(m!)^2}\left(\frac{kr}{2}\right)^{2m} &
    \sum_{m=0}^{\infty}\frac{(-1)^m}{(m!)^2}\left(\frac{k}{2}\right)^{2m}
    (-\partial_t)^{2m}\left(\frac{1}{t}\right) &
    \frac{1}{\sqrt{t^2+k^2}}
    \\
    \hline
    \chi_{[0,R]}(r) &
    J_\nu\text{-series on }[0,R] &
    \sum_{m=0}^{\infty}\frac{(-1)^m}{m!\Gamma(m+\nu+1)}
    \left(\frac{k}{2}\right)^{2m+\nu}\int_0^R r^{2m+\nu+1}dr &
    \frac{R}{k}J_{\nu+1}(kR)
    \\
    \hline
    \end{array}
    """)

    st.markdown("### Interpretation")
    st.markdown(r"""
    The Hankel operator transform completes the operator-planting family alongside the
    Laplace, Fourier, and Mellin forms. While Laplace planting governs exponential damping,
    Mellin planting governs power-law scaling, and Fourier planting governs oscillatory
    complex kernels, Hankel planting introduces radial oscillatory structure through the
    Bessel kernel. In this way, all four transforms arise from one unified differential
    planting principle.
    """)
elif section == "Unified Transform Table":
    st.header("5. Unified Transform Table")

    st.markdown(r"""
    This section consolidates the operator-planting viewpoint developed for the Laplace,
    Fourier, Mellin, and Hankel transforms, then extends the framework to the
    inverse-kernel duality and the Laplace--Gamma reconstruction. The goal is to show
    that all these transforms arise from a single planted differential mechanism acting
    on suitable kernels.
    """)

    st.subheader("Unified operator planting framework")

    st.latex(r"""
    \begin{array}{|c|c|c|}
    \hline
    \text{Transform} & \text{Rank }\alpha & \text{Operator planting form} \\
    \hline
    \text{Laplace} & \alpha=n &
    (-\partial_s)^\alpha\!\left(\frac1s\right)
    = \frac{\Gamma(\alpha+1)}{s^{\alpha+1}}
    \\
    \hline
    \text{Fourier} & \alpha=n &
    (-\partial_s)^\alpha\!\left(\frac1s\right)
    +(\partial_{\bar s})^\alpha\!\left(\frac1{\bar s}\right)
    = \Gamma(\alpha+1)\!\left(\frac1{s^{\alpha+1}}+\frac{(-1)^\alpha}{\bar s^{\alpha+1}}\right)
    \\
    \hline
    \text{Mellin} & \alpha=n+s-1 &
    (-\partial_t)^\alpha\!\left(\frac1t\right)
    = \frac{\Gamma(n+s)}{t^{n+s}}
    \\
    \hline
    \text{Hankel} & \alpha=n+\nu &
    (-\partial_t)^\alpha\!\left(\frac1t\right)
    = \frac{\Gamma(n+\nu+1)}{t^{n+\nu+1}}
    \\
    \hline
    \end{array}
    """)

    st.markdown(r"""
    **Summary: Unified Operator Planting**
    """)

    st.markdown(r"""
    - The parameter $(\alpha)$ denotes the planting rank, which may be integer,
      fractional, or bilateral, depending on the underlying transform.
    - Laplace planting corresponds to integer rank $(\alpha=n)$, acting on $(1/s)$.
    - Fourier planting is the bilateral extension acting simultaneously on $(1/s)$
      and $(1/\bar s)$.
    - Mellin planting extends the rank to $(\alpha=n+s-1)$, acting on $(1/t)$.
    - Hankel planting introduces the radial shift $(\nu)$, with $(\alpha=n+\nu)$.
    """)

    st.latex(r"""
    \text{Laplace: }\quad
    \alpha=n,\qquad
    (-\partial_s)^\alpha\!\left(\frac1s\right)=\frac{\Gamma(\alpha+1)}{s^{\alpha+1}}
    """)

    st.latex(r"""
    \text{Fourier: }\quad
    s=\sigma+i\omega,\qquad
    \alpha=n
    """)

    st.latex(r"""
    \text{Mellin: }\quad
    \alpha=n+s-1,\qquad
    (-\partial_t)^\alpha\!\left(\frac1t\right)=\frac{\Gamma(n+s)}{t^{n+s}}
    """)

    st.latex(r"""
    \text{Hankel: }\quad
    \alpha=n+\nu,\qquad
    (-\partial_t)^\alpha\!\left(\frac1t\right)=\frac{\Gamma(n+\nu+1)}{t^{n+\nu+1}}
    """)

    st.subheader("General kernel")

    st.markdown(r"""
    In this part we introduce the **Inverse-Kernel Duality** within the framework of the
    differential operator transform. Starting from the planted-series definition
    """)

    st.latex(r"""
    \mathcal{T}\{f\}(t)=\sum_{n=0}^{\infty}(-1)^n a_n D_t^n G(t),
    \qquad
    f(x)=\sum_{n=0}^{\infty}a_n x^n
    """)

    st.markdown(r"""
    where \(G(t)\) is a generated kernel, we show that this purely differential
    formulation admits the equivalent integral representation
    """)

    st.latex(r"""
    \mathcal{T}\{f\}(t)=\int_0^\infty e^{-tx} f(x)\,g(x)\,dx,
    \qquad
    g(x)=\mathcal{T}^{-1}\{G(t)\}(x).
    """)

    st.markdown(r"""
    This identity reveals a structural duality: every planted derivative in the
    \(t\)-domain corresponds to a weighted Laplace-type integration in the \(x\)-domain,
    with the weighting function being the inverse kernel of the transform itself.
    """)

    st.markdown("### General kernel setup")

    st.markdown(r"""
    In the original operator-Laplace setting, the planted transform was
    """)

    st.latex(r"""
    \mathcal{T}\{f\}(t)
    =\sum_{n=0}^{\infty}(-1)^n a_n D_t^n\!\left(\frac1t\right),
    \qquad
    D_t^n\!\left(\frac1t\right)=(-1)^n\frac{n!}{t^{n+1}}
    """)

    st.markdown(r"""
    and it reproduces the classical Laplace integral
    """)

    st.latex(r"""
    \mathcal{L}\{f(x)\}(t)=\int_0^\infty e^{-tx}f(x)\,dx.
    """)

    st.markdown(r"""
    The present work generalizes the fixed kernel \(1/t\) to an arbitrary generated
    kernel \(G(t)\):
    """)

    st.latex(r"""
    \mathcal{T}\{f\}(t)=\sum_{n=0}^{\infty}(-1)^n a_n D_t^n G(t)
    """)

    st.latex(r"""
    \mathcal{T}\{f\}(t)=\int_0^\infty e^{-tx}f(x)g(x)\,dx,
    \qquad
    g(x)=\mathcal{T}^{-1}\{G(t)\}(x).
    """)

    st.subheader("Theorem 14.1 (Inverse-Kernel Form of the Operator Transform)")

    st.markdown(r"""
    Let
    """)

    st.latex(r"""
    G(t)=\mathcal{T}\{g(x)\}(t)=\int_0^\infty e^{-tx}g(x)\,dx
    """)

    st.markdown(r"""
    be any operator-type transform of a kernel $(g(x))$, and let
    """)

    st.latex(r"""
    f(x)=\sum_{n=0}^{\infty}a_n x^n
    """)

    st.markdown(r"""
    be analytic near $(x=0)$. Then the operator-planted series
    """)

    st.latex(r"""
    \mathcal{T}\{f\}(t)=\sum_{n=0}^{\infty}(-1)^n a_n D^n G(t)
    """)

    st.markdown(r"""
    admits the integral representation
    """)

    st.latex(r"""
    \mathcal{T}\{f\}(t)=\int_0^\infty e^{-tx}f(x)g(x)\,dx.
    """)

    st.markdown(r"""
    **Proof.** For each integer $(n\ge 0)$,
    """)

    st.latex(r"""
    D^n G(t)=(-1)^n\int_0^\infty x^n e^{-tx}g(x)\,dx
    """)

    st.markdown(r"""
    hence
    """)

    st.latex(r"""
    \sum_{n=0}^{\infty} a_n(-1)^n D^n G(t)
    =
    \sum_{n=0}^{\infty} a_n \int_0^\infty x^n e^{-tx}g(x)\,dx
    """)

    st.latex(r"""
    =
    \int_0^\infty e^{-tx}g(x)\left(\sum_{n=0}^{\infty}a_nx^n\right)dx
    =
    \int_0^\infty e^{-tx}f(x)g(x)\,dx.
    """)

    st.markdown(r"""
    This completes the proof.
    """)

    st.subheader("Convergence Requirements")

    st.markdown(r"""
    Throughout this framework we assume:
    """)

    st.markdown(r"""
    - $(f(x)=\sum_{n=0}^\infty a_n x^n)$ is analytic near $(x=0)$.
    - The kernel generator $(g(x))$ satisfies sufficient exponential decay/integrability.
    - The double series-integral is absolutely convergent so that Tonelli/Fubini apply.
    """)

    st.latex(r"""
    \sum_{n=0}^{\infty}|a_n|\int_0^\infty x^n e^{-(\Re t)x}|g(x)|\,dx<\infty
    """)

    st.markdown(r"""
    Under these mild assumptions, the inverse-kernel duality identity is valid.
    """)

    st.subheader("Examples of inverse-kernel duality")

    st.markdown("### Example 14.2")
    st.latex(r"""
    G(t)=\frac{t}{t^2+b^2},
    \qquad
    g(x)=\mathcal{T}^{-1}\{G\}(x)=\cos(bx)
    """)

    st.latex(r"""
    \sum_{n=0}^{\infty}(-1)^n a_n D^n\!\left(\frac{t}{t^2+b^2}\right)
    =
    \int_0^\infty f(x)e^{-tx}\cos(bx)\,dx
    """)

    st.markdown("### Example 14.3")
    st.latex(r"""
    G(t)=\frac{b}{t^2+b^2},
    \qquad
    g(x)=\mathcal{T}^{-1}\{G\}(x)=\sin(bx)
    """)

    st.latex(r"""
    \sum_{n=0}^{\infty}(-1)^n a_n D^n\!\left(\frac{b}{t^2+b^2}\right)
    =
    \int_0^\infty f(x)e^{-tx}\sin(bx)\,dx
    """)

    st.subheader("Kernel multiplication and boundary contribution")

    st.markdown(r"""
    Let
    """)

    st.latex(r"""
    G(t)=\int_0^\infty e^{-tx}g(x)\,dx
    """)

    st.markdown(r"""
    Then multiplication by \(t\) satisfies
    """)

    st.latex(r"""
    tG(t)=g(0)+\int_0^\infty e^{-tx}g'(x)\,dx
    """)

    st.markdown(r"""
    and hence, in the sense of inverse kernels,
    """)

    st.latex(r"""
    \mathcal{T}^{-1}\{tG(t)\}(x)=g'(x)+g(0)\delta(x).
    """)

    st.markdown(r"""
    The boundary contribution depends only on the original inverse kernel value at the
    origin.
    """)

    st.markdown("### Example 14.5")
    st.latex(r"""
    G(t)=\frac{1}{t^2+b^2}
    =\int_0^\infty e^{-tx}\frac{\sin(bx)}{b}\,dx
    """)

    st.latex(r"""
    g(x)=\frac1b\sin(bx),
    \qquad
    g(0)=0
    """)

    st.latex(r"""
    tG(t)=\int_0^\infty e^{-tx}g'(x)\,dx
    =\int_0^\infty e^{-tx}\cos(bx)\,dx
    =\frac{t}{t^2+b^2}.
    """)

    st.subheader("Kernel division and integral injection")

    st.markdown(r"""
    Let
    """)

    st.latex(r"""
    G(t)=\int_0^\infty e^{-tx}g(x)\,dx,
    \qquad
    H(t):=\frac{G(t)}{t}, \qquad t>0
    """)

    st.markdown(r"""
    Then \(H\) admits the inverse-kernel representation
    """)

    st.latex(r"""
    H(t)=\int_0^\infty e^{-tx}h(x)\,dx,
    \qquad
    h(x)=\int_0^x g(u)\,du.
    """)

    st.latex(r"""
    \mathcal{T}^{-1}\!\left\{\frac{G(t)}{t}\right\}(x)=\int_0^x g(u)\,du
    """)

    st.markdown(r"""
    and more generally
    """)

    st.latex(r"""
    \mathcal{T}^{-1}\!\left\{\frac{G(t)}{t^k}\right\}(x)
    =
    \int_0^x\int_0^{u_1}\cdots\int_0^{u_{k-1}} g(u_k)\,du_k\cdots du_1.
    """)

    st.markdown(r"""
    Thus, division by \(t\) injects repeated integration into the inverse kernel.
    """)

    st.markdown("### Example 14.7")
    st.latex(r"""
    G(t)=\frac{1}{t^2+b^2}
    =\int_0^\infty e^{-tx}\frac{\sin(bx)}{b}\,dx
    """)

    st.latex(r"""
    \mathcal{T}^{-1}\!\left\{\frac{G(t)}{t}\right\}(x)
    =
    \int_0^x\frac{\sin(bu)}{b}\,du
    =
    \frac{1-\cos(bx)}{b^2}
    """)

    st.latex(r"""
    \frac{1}{t(t^2+b^2)}
    =
    \int_0^\infty e^{-tx}\frac{1-\cos(bx)}{b^2}\,dx
    """)

    st.markdown("### Example 14.8")
    st.latex(r"""
    G(t)=\frac{t}{t^2+b^2}
    =\int_0^\infty e^{-tx}\cos(bx)\,dx
    """)

    st.latex(r"""
    \mathcal{T}^{-1}\!\left\{\frac{G(t)}{t}\right\}(x)
    =
    \int_0^x\cos(bu)\,du
    =
    \frac{\sin(bx)}{b}
    """)

    st.latex(r"""
    \frac{1}{t^2+b^2}
    =
    \int_0^\infty e^{-tx}\frac{\sin(bx)}{b}\,dx
    """)

    st.markdown("### Example 14.10 (Laplace transform of Si)")
    st.latex(r"""
    \sum_{n=0}^{\infty}(-1)^n a_n D_t^n\!\left(\arctan\!\frac{b}{t}\right)
    =
    \int_0^\infty f(x)e^{-tx}\frac{\sin(bx)}{x}\,dx
    """)

    st.markdown(r"""
    Dividing by \(t\) injects one integration and yields the sine integral
    $(\mathrm{Si}(x)=\int_0^x \frac{\sin u}{u}\,du)$.
    """)

    st.latex(r"""
    \frac1t\arctan\!\left(\frac{b}{t}\right)
    =
    \int_0^\infty e^{-tx}\,\mathrm{Si}(bx)\,dx
    """)

    st.latex(r"""
    \int_0^\infty e^{-x}\mathrm{Si}(x)\,dx=\arctan(1)=\frac{\pi}{4}
    """)

    st.subheader("Reconstruction Identity")

    st.markdown(r"""
    We now continue from the fundamental Laplace kernel
    """)

    st.latex(r"""
    G(t)=\frac1t,
    \qquad
    g(x)=1
    """)

    st.latex(r"""
    \mathcal{T}\{f\}(t)
    =
    \sum_{n=0}^{\infty}(-1)^n a_n D^n\!\left(\frac1t\right)
    =
    \int_0^\infty f(x)e^{-tx}\,dx
    """)

    st.markdown(r"""
    The inverse-kernel reconstruction gives
    """)

    st.latex(r"""
    \mathcal{T}^{-1}\!\left\{\frac1t\right\}(x)=1,
    \qquad
    \mathcal{T}^{-1}\!\left\{\frac1{t^2}\right\}(x)=x,
    \qquad
    \mathcal{T}^{-1}\!\left\{\frac1{t^3}\right\}(x)=\frac{x^2}{2},
    \qquad
    \mathcal{T}^{-1}\!\left\{\frac1{t^\mu}\right\}(x)=\frac{x^{\mu-1}}{\Gamma(\mu)}.
    """)

    st.subheader("Laplace--Gamma Transform: Foundational Reconstruction")

    st.markdown(r"""
    We introduce the Laplace--Gamma kernel
    """)

    st.latex(r"""
    \Gamma_\mu(t)=t^{-\mu},\qquad \mu>0
    """)

    st.markdown(r"""
    and apply the planted differential hierarchy to it.
    """)

    st.markdown("### 1. Fundamental kernel")
    st.latex(r"""
    \Gamma_\mu(t)=t^{-\mu},\qquad \mu>0
    """)

    st.markdown("### 2. Differential planting law")
    st.latex(r"""
    (-\partial_t)^n \Gamma_\mu(t)=(\mu)_n\,t^{-\mu-n},
    \qquad
    (\mu)_n=\mu(\mu+1)\cdots(\mu+n-1)
    """)

    st.markdown("### 3. Definition on a Taylor series")
    st.latex(r"""
    f(x)=\sum_{n=0}^{\infty}a_nx^n
    """)

    st.latex(r"""
    \mathcal{L}_\Gamma\{f\}(t)
    =
    \sum_{n=0}^{\infty}a_n(-\partial_t)^n\Gamma_\mu(t)
    =
    \sum_{n=0}^{\infty}a_n(\mu)_n t^{-\mu-n}
    """)

    st.markdown("### 4. Conversion to an integral form")
    st.latex(r"""
    t^{-\mu-n}
    =
    \frac{1}{\Gamma(\mu+n)}
    \int_0^\infty e^{-tu}u^{\mu+n-1}\,du
    """)

    st.latex(r"""
    \mathcal{L}_\Gamma\{f\}(t)
    =
    \sum_{n=0}^{\infty}a_n(\mu)_n
    \frac{1}{\Gamma(\mu+n)}
    \int_0^\infty e^{-tu}u^{\mu+n-1}\,du
    """)

    st.latex(r"""
    =
    \frac1{\Gamma(\mu)}
    \int_0^\infty e^{-tu}u^{\mu-1}
    \left(\sum_{n=0}^{\infty}a_nu^n\right)du
    """)

    st.markdown("### 5. Internal structural definition")
    st.latex(r"""
    \sum_{n=0}^{\infty}a_nu^n=f(u)
    """)

    st.latex(r"""
    \boxed{
    \mathcal{L}_\Gamma\{f\}(t)
    =
    \frac1{\Gamma(\mu)}
    \int_0^\infty e^{-tu}u^{\mu-1}f(u)\,du
    }
    """)

    st.markdown(r"""
    This expresses the transform as a Gamma-weighted Laplace-type integral.
    """)

    st.markdown("### 6. Illustrative examples")
    st.latex(r"""
    f(x)=1 \;\Rightarrow\; \mathcal{L}_\Gamma\{1\}(t)=t^{-\mu}
    """)

    st.latex(r"""
    f(x)=x^n \;\Rightarrow\; \mathcal{L}_\Gamma\{x^n\}(t)=(\mu)_n t^{-\mu-n}
    """)

    st.latex(r"""
    f(x)=e^{-x}
    \;\Rightarrow\;
    \mathcal{L}_\Gamma\{e^{-x}\}(t)
    =
    \sum_{n=0}^{\infty}\frac{(-1)^n}{n!}(\mu)_n t^{-\mu-n}
    """)

    st.latex(r"""
    f(x)=\sin x
    \;\Rightarrow\;
    \mathcal{L}_\Gamma\{\sin x\}(t)
    =
    \sum_{n=0}^{\infty}\frac{(-1)^n}{(2n+1)!}(\mu)_{2n+1}t^{-\mu-(2n+1)}
    """)

    st.markdown("### 7. Structural summary")
    st.latex(r"""
    \boxed{
    \mathcal{L}_\Gamma\{f\}(t)
    =
    \sum_{n=0}^{\infty}a_n(\mu)_n t^{-\mu-n}
    =
    \frac1{\Gamma(\mu)}\int_0^\infty e^{-tu}u^{\mu-1}f(u)\,du
    }
    """)

    st.markdown(r"""
    This form preserves the pure differential planting structure while establishing the
    direct integral interpretation in terms of the Gamma--Laplace hierarchy.
    """)

    st.subheader("Closed-form Laplace--Gamma Transform for exponential inputs")

    st.markdown("### Case A: $(f(x)=e^x)$")
    st.latex(r"""
    \mathcal{L}_\Gamma\{e^x\}(t)
    =
    t^{-\mu}\sum_{n=0}^{\infty}\frac{(\mu)_n}{n!}\left(\frac1t\right)^n
    =
    t^{-\mu}\left(1-\frac1t\right)^{-\mu}
    """)

    st.latex(r"""
    \boxed{
    \mathcal{L}_\Gamma\{e^x\}(t)=(t-1)^{-\mu}
    },\qquad t>1
    """)

    st.markdown("### Case B: $(f(x)=e^{-x})$")
    st.latex(r"""
    \mathcal{L}_\Gamma\{e^{-x}\}(t)
    =
    t^{-\mu}\sum_{n=0}^{\infty}\frac{(\mu)_n}{n!}\left(-\frac1t\right)^n
    =
    t^{-\mu}\left(1+\frac1t\right)^{-\mu}
    """)

    st.latex(r"""
    \boxed{
    \mathcal{L}_\Gamma\{e^{-x}\}(t)=(t+1)^{-\mu}
    },\qquad t>0
    """)

    st.markdown("### Case C: $(f(x)=e^{ix})$")
    st.latex(r"""
    \mathcal{L}_\Gamma\{e^{ix}\}(t)
    =
    t^{-\mu}\sum_{n=0}^{\infty}\frac{(\mu)_n}{n!}\left(\frac{i}{t}\right)^n
    =
    t^{-\mu}\left(1-\frac{i}{t}\right)^{-\mu}
    """)

    st.latex(r"""
    \boxed{
    \mathcal{L}_\Gamma\{e^{ix}\}(t)=(t-i)^{-\mu}
    }
    """)

    st.markdown("### Polar form and Euler decomposition")
    st.latex(r"""
    t-i=re^{-i\theta},
    \qquad
    r=\sqrt{t^2+1},
    \qquad
    \theta=\tan^{-1}\!\left(\frac1t\right)
    """)

    st.latex(r"""
    (t-i)^{-\mu}=r^{-\mu}e^{i\mu\theta}
    =r^{-\mu}\big(\cos(\mu\theta)+i\sin(\mu\theta)\big)
    """)

    st.latex(r"""
    \boxed{
    \mathcal{L}_\Gamma\{\cos x\}(t)=r^{-\mu}\cos(\mu\theta)
    },\qquad
    \boxed{
    \mathcal{L}_\Gamma\{\sin x\}(t)=r^{-\mu}\sin(\mu\theta)
    }
    """)

    st.markdown("### General exponential input $(f(x)=e^{rx})$")
    st.latex(r"""
    \mathcal{L}_\Gamma\{e^{rx}\}(t)
    =
    \frac1{\Gamma(\mu)}
    \int_0^\infty e^{-(t-r)u}u^{\mu-1}\,du
    \;=\;
    (t-r)^{-\mu},
    \qquad \Re(t-r)>0
    """)

    st.markdown("### Comprehensive Laplace--Gamma Transform Table")
    st.latex(r"""
    \begin{array}{|c|c|}
    \hline
    \text{Input }f(x) & \mathcal{L}_\Gamma\{f\}(t) \\
    \hline
    1 & t^{-\mu} \\
    x^n & (\mu)_n t^{-\mu-n} \\
    e^x & (t-1)^{-\mu} \\
    e^{-x} & (t+1)^{-\mu} \\
    e^{ix} & (t-i)^{-\mu}=r^{-\mu}e^{i\mu\theta} \\
    \sin x & r^{-\mu}\sin(\mu\theta) \\
    \cos x & r^{-\mu}\cos(\mu\theta) \\
    e^{rx} & (t-r)^{-\mu} \\
    \hline
    \end{array}
    """)

    st.latex(r"""
    r=\sqrt{t^2+1},\qquad \theta=\tan^{-1}\!\left(\frac1t\right)
    """)

    st.subheader("Frullani-like representation")

    st.markdown(r"""
    Split the value at the origin:
    """)

    st.latex(r"""
    f(u)=f(0)+\big(f(u)-f(0)\big)
    """)

    st.markdown(r"""
    Substitute into the integral:
    """)

    st.latex(r"""
    \mathcal{L}_\Gamma\{f\}(t)
    =
    \frac{f(0)}{\Gamma(\mu)}
    \int_0^\infty e^{-tu}u^{\mu-1}\,du
    +
    \frac1{\Gamma(\mu)}
    \int_0^\infty e^{-tu}u^{\mu-1}[f(u)-f(0)]\,du
    """)

    st.markdown(r"""
    Evaluating the first integral gives
    """)

    st.latex(r"""
    \boxed{
    \mathcal{L}_\Gamma\{f\}(t)
    =
    f(0)t^{-\mu}
    +
    \frac1{\Gamma(\mu)}
    \int_0^\infty e^{-tu}u^{\mu-1}[f(u)-f(0)]\,du
    }
    """)

    st.subheader("Geometric and complex-phase interpretation")

    st.markdown(r"""
    For a complex input $(f(x)=e^{(a+ib)x})$, the transform becomes
    """)

    st.latex(r"""
    \mathcal{L}_\Gamma\{e^{(a+ib)x}\}(t)
    =
    (t-a-ib)^{-\mu}
    =\rho^{-\mu}e^{-i\mu\varphi}
    """)

    st.latex(r"""
    \rho=\sqrt{(t-a)^2+b^2},
    \qquad
    \varphi=\tan^{-1}\!\left(\frac{b}{t-a}\right)
    """)

    st.latex(r"""
    \boxed{
    \mathcal{L}_\Gamma\{e^{ax}\cos(bx)\}(t)=\rho^{-\mu}\cos(\mu\varphi)
    }
    """)

    st.latex(r"""
    \boxed{
    \mathcal{L}_\Gamma\{e^{ax}\sin(bx)\}(t)=\rho^{-\mu}\sin(\mu\varphi)
    }
    """)

    st.markdown(r"""
    The magnitude $(\rho^{-\mu})$ describes power-law attenuation, while the phase
    $(-\mu\varphi)$ encodes a fractional rotation in the complex Laplace plane.
    """)

    st.subheader("Fractional damping spectrum")

    st.latex(r"""
    D_\mu(t)=t^{-\mu}
    """)

    st.latex(r"""
    \frac{d}{dt}D_\mu(t)=-\mu t^{-\mu-1},
    \qquad
    \frac{d^2}{dt^2}D_\mu(t)=\mu(\mu+1)t^{-\mu-2}
    """)

    st.latex(r"""
    (-\partial_t)^n D_\mu(t)=(\mu)_n t^{-\mu-n}
    """)

    st.markdown(r"""
    This recursive law provides the internal algebra of decay orders and forms the basis
    for all differential planting operations in the Laplace--Gamma family.
    """)

    st.subheader("Concluding structural summary")

    st.latex(r"""
    \boxed{
    \mathcal{L}_\Gamma\{f\}(t)
    =
    \frac1{\Gamma(\mu)}\int_0^\infty e^{-tu}u^{\mu-1}f(u)\,du
    =
    \sum_{n=0}^{\infty}a_n(\mu)_n t^{-\mu-n}
    }
    """)

    st.markdown(r"""
    It encompasses the classical Laplace transform $(\mu=1)$ and its fractional
    Gamma-weighted extension in a single differential--integral operator framework.
    """)

    st.markdown("### Example 15.1")
    st.markdown(r"""
    Evaluate
    """)

    st.latex(r"""
    I=\int_0^\infty \frac{e^{-u}\sin u}{\sqrt{u}}\,du
    """)

    st.markdown(r"""
    Using
    """)

    st.latex(r"""
    \int_0^\infty e^{-tu}u^{\mu-1}f(u)\,du=\Gamma(\mu)\mathcal{L}_\Gamma\{f\}(t)
    """)

    st.markdown(r"""
    choose $(\mu=\tfrac12)$, $(f(u)=sin u)$, $(t=1)$. Then
    """)

    st.latex(r"""
    r=\sqrt{t^2+1}=\sqrt2,
    \qquad
    \theta=\tan^{-1}(1)=\frac{\pi}{4}
    """)

    st.latex(r"""
    I
    =
    \Gamma\!\left(\frac12\right)r^{-1/2}\sin\!\left(\frac{\theta}{2}\right)
    =
    \sqrt{\pi}\,(\sqrt2)^{-1/2}\sin\!\left(\frac{\pi}{8}\right).
    """)
elif section == "Logarithmic Snapshot":
    st.header("6. Logarithmic Snapshot")

    st.markdown(r"""
    This section develops the logarithmic transform in its differential planting form,
    derives its Frullani-regularized integral representation, and records the main
    closed forms for exponential, trigonometric, hyperbolic, and Bessel inputs.
    We then show how the logarithmic family is linked by differentiation to the
    Laplace and Mellin transforms.
    """)

    st.subheader("Logarithmic Transform (Differential Planting)")

    st.markdown("### Kernel and basic fractional law")

    st.latex(r"""
    \Gamma(t)=\ln(t),\qquad t>0
    """)

    st.markdown(r"""
    For integer $(n\ge1)$,
    """)

    st.latex(r"""
    \frac{d^n}{dt^n}\ln(t)=(-1)^{n-1}(n-1)!t^{-n}
    """)

    st.markdown(r"""
    Hence the planted \(n\)-th derivative of the logarithmic seed is
    """)

    st.latex(r"""
    (-\partial_t)^n\ln(t)=-(n-1)!t^{-n},\qquad n\ge1
    """)

    st.latex(r"""
    (-\partial_t)^0\ln(t)=\ln(t)
    """)

    st.markdown("### Definition (Differential planting form)")

    st.markdown(r"""
    For an analytic input $(f(x)=\sum_{n=0}^{\infty}a_nx^n)$ around $(x=0)$, define
    the logarithmic planting transform by
    """)

    st.latex(r"""
    T_{\ln}\{f\}(t)
    :=
    \sum_{n=0}^{\infty}a_n(-\partial_t)^n\ln(t),\qquad t>0
    """)

    st.latex(r"""
    T_{\ln}\{f\}(t)
    =
    a_0\ln(t)+\sum_{n=1}^{\infty}a_n(-\partial_t)^n\ln(t)
    """)

    st.subheader("Worked cases (step by step)")

    st.markdown("### Case A: $(f(x)=1)$")
    st.latex(r"""
    a_0=1,\quad a_{n\ge1}=0
    """)

    st.latex(r"""
    \boxed{T_{\ln}\{1\}(t)=\ln(t)}
    """)

    st.markdown("### Case B: $(f(x)=x)$")
    st.latex(r"""
    a_1=1,\quad a_n=0\;(n\neq1)
    """)

    st.latex(r"""
    T_{\ln}\{x\}(t)=(-\partial_t)\ln(t)=-\frac1t
    """)

    st.latex(r"""
    \boxed{T_{\ln}\{x\}(t)=-\frac1t}
    """)

    st.markdown("### Case C: $(f(x)=x^n), (n\ge1)$")
    st.latex(r"""
    T_{\ln}\{x^n\}(t)=(-\partial_t)^n\ln(t)=-(n-1)!t^{-n}
    """)

    st.latex(r"""
    \boxed{T_{\ln}\{x^n\}(t)=-(n-1)!t^{-n}},\qquad n\ge1
    """)

    st.markdown("### Case D: $(f(x)=e^x)$")
    st.latex(r"""
    e^x=\sum_{n=0}^{\infty}\frac{x^n}{n!}
    """)

    st.latex(r"""
    T_{\ln}\{e^x\}(t)
    =
    \ln(t)+\sum_{n=1}^{\infty}\frac1{n!}\big[-(n-1)!t^{-n}\big]
    """)

    st.latex(r"""
    =
    \ln(t)-\sum_{n=1}^{\infty}\frac1n t^{-n}
    """)

    st.latex(r"""
    \sum_{n=1}^{\infty}\frac{z^n}{n}=-\ln(1-z),\qquad |z|<1
    """)

    st.latex(r"""
    \boxed{T_{\ln}\{e^x\}(t)=\ln(t-1)},\qquad t>1
    """)

    st.markdown("### Case E: $(f(x)=e^{-x})$")
    st.latex(r"""
    e^{-x}=\sum_{n=0}^{\infty}\frac{(-1)^n}{n!}x^n
    """)

    st.latex(r"""
    T_{\ln}\{e^{-x}\}(t)
    =
    \ln(t)-\sum_{n=1}^{\infty}\frac{(-1)^n}{n}t^{-n}
    """)

    st.latex(r"""
    =
    \ln(t)+\ln\!\left(1+\frac1t\right)
    =\ln(1+t)
    """)

    st.latex(r"""
    \boxed{T_{\ln}\{e^{-x}\}(t)=\ln(1+t)}
    """)

    st.markdown("### Case F: $(f(x)=sin x)$")
    st.latex(r"""
    \sin x=\sum_{k=0}^{\infty}\frac{(-1)^k}{(2k+1)!}x^{2k+1}
    """)

    st.latex(r"""
    T_{\ln}\{\sin x\}(t)
    =
    \sum_{k=0}^{\infty}\frac{(-1)^k}{(2k+1)!}\big[-(2k)!t^{-(2k+1)}\big]
    """)

    st.latex(r"""
    =
    -\sum_{k=0}^{\infty}\frac{(-1)^k}{2k+1}t^{-(2k+1)}
    """)

    st.latex(r"""
    \boxed{
    T_{\ln}\{\sin x\}(t)=-\arctan\!\left(\frac1t\right)
    }
    """)

    st.markdown("### Case G: $(f(x)=cos x)$")
    st.latex(r"""
    \cos x=\sum_{k=0}^{\infty}\frac{(-1)^k}{(2k)!}x^{2k}
    """)

    st.latex(r"""
    T_{\ln}\{\cos x\}(t)
    =
    \ln(t)-\sum_{k=1}^{\infty}\frac{(-1)^k}{2k}t^{-2k}
    """)

    st.latex(r"""
    \sum_{k=1}^{\infty}\frac{(-1)^k}{2k}z^{2k}
    =
    -\frac12\ln(1+z^2)
    """)

    st.latex(r"""
    \boxed{
    T_{\ln}\{\cos x\}(t)=\frac12\ln(t^2+1)
    }
    """)

    st.subheader("Frullani-regularized integral form")

    st.markdown(r"""
    Let $f$ be analytic at $0$, with Maclaurin series $(f(x)=\sum_{n=0}^{\infty}a_nx^n)$.
    Define
    """)

    st.latex(r"""
    T_{\ln}\{f\}(t)
    :=
    \sum_{n=0}^{\infty}a_n(-\partial_t)^n\ln(t),\qquad t>0.
    """)

    st.markdown(r"""
    Assuming mild growth so that Tonelli/Fubini apply, the transform admits the
    Frullani-type integral representation
    """)

    st.latex(r"""
    \boxed{
    T_{\ln}\{f\}(t)
    =
    f(0)\ln(t)-\int_0^\infty e^{-tu}\frac{f(u)-f(0)}{u}\,du
    }
    """)

    st.markdown("### Proof")

    st.markdown("#### Step 1: split the \(n=0\) term")
    st.latex(r"""
    T_{\ln}\{f\}(t)=a_0\ln(t)+\sum_{n=1}^{\infty}a_n(-\partial_t)^n\ln(t),
    \qquad a_0=f(0)
    """)

    st.markdown("#### Step 2: closed form for planted derivatives")
    st.latex(r"""
    (-\partial_t)^n\ln(t)=-(n-1)!t^{-n},\qquad n\ge1
    """)

    st.markdown("#### Step 3: Laplace--Gamma identity")
    st.latex(r"""
    (n-1)!t^{-n}
    =
    \int_0^\infty e^{-tu}u^{n-1}\,du
    """)

    st.markdown("#### Step 4: interchange sum and integral")
    st.latex(r"""
    T_{\ln}\{f\}(t)
    =
    f(0)\ln(t)-\int_0^\infty e^{-tu}
    \left(\sum_{n=1}^{\infty}a_nu^{n-1}\right)du
    """)

    st.markdown("#### Step 5: identify the Maclaurin difference quotient")
    st.latex(r"""
    \sum_{n=1}^{\infty}a_nu^{n-1}
    =
    \frac{f(u)-f(0)}{u}
    """)

    st.latex(r"""
    T_{\ln}\{f\}(t)
    =
    f(0)\ln(t)-\int_0^\infty e^{-tu}\frac{f(u)-f(0)}{u}\,du
    """)

    st.subheader("Consistency check with \(f(x)=e^x\)")

    st.latex(r"""
    T_{\ln}\{e^x\}(t)=\ln(t-1),\qquad t>1
    """)

    st.latex(r"""
    T_{\ln}\{f\}(t)=f(0)\ln(t)-\int_0^\infty e^{-tu}\frac{f(u)-f(0)}{u}\,du
    """)

    st.markdown(r"""
    Since $(f(0)=1)$ and $(f(u)-f(0)=e^u-1)$,
    """)

    st.latex(r"""
    \int_0^\infty e^{-tu}\frac{e^u-1}{u}\,du
    =
    \ln(t)-\ln(t-1)
    =
    \ln\!\left(\frac{t}{t-1}\right)
    """)

    st.markdown(r"""
    Thus the Frullani-regularized integral reproduces the same differential result.
    """)

    st.subheader("Cosine logarithmic transform")

    st.markdown(r"""
    Starting from the general logarithmic integral form,
    """)

    st.latex(r"""
    \int_0^\infty e^{-tu}\frac{f(u)-f(0)}{u}\,du
    =
    f(0)\ln(t)-T_{\ln}\{f\}(t)
    """)

    st.markdown(r"""
    choose $(f(x)=cos x)$, $(f(0)=1)$. Then
    """)

    st.latex(r"""
    \int_0^\infty e^{-tu}\frac{\cos u-1}{u}\,du
    =
    \ln(t)-T_{\ln}\{\cos x\}(t)
    """)

    st.latex(r"""
    T_{\ln}\{\cos x\}(t)=\frac12\ln(t^2+1)
    """)

    st.latex(r"""
    \int_0^\infty e^{-tu}\frac{\cos u-1}{u}\,du
    =
    \ln(t)-\frac12\ln(t^2+1)
    """)

    st.latex(r"""
    \int_0^\infty e^{-2x}\frac{\cos x-1}{x}\,dx
    =
    -\frac12\ln\!\left(\frac54\right)
    """)

    st.subheader("Sine logarithmic transform")

    st.markdown(r"""
    For $(f(x)=sin x)$, $(f(0)=0)$, so
    """)

    st.latex(r"""
    \int_0^\infty e^{-tu}\frac{\sin u}{u}\,du
    =
    -T_{\ln}\{\sin x\}(t)
    """)

    st.latex(r"""
    T_{\ln}\{\sin x\}(t)
    =
    -\arctan\!\left(\frac1t\right)
    """)

    st.latex(r"""
    \boxed{
    \int_0^\infty e^{-tu}\frac{\sin u}{u}\,du
    =
    \arctan\!\left(\frac1t\right)
    }
    """)

    st.latex(r"""
    \text{At }t=1:\qquad
    \int_0^\infty e^{-u}\frac{\sin u}{u}\,du=\frac{\pi}{4}
    """)

    st.subheader("General cases of the logarithmic transform")

    st.markdown("### (1) Exponential input $(f(x)=e^{rx})$")
    st.latex(r"""
    e^{rx}=\sum_{n=0}^{\infty}\frac{r^n}{n!}x^n
    """)

    st.latex(r"""
    T_{\ln}\{e^{rx}\}(t)
    =
    \ln(t)-\sum_{n=1}^{\infty}\frac{r^n}{n}t^{-n}
    =
    \ln(t)+\ln\!\left(1-\frac{r}{t}\right)
    """)

    st.latex(r"""
    \boxed{
    T_{\ln}\{e^{rx}\}(t)=\ln(t-r)
    },\qquad t>r
    """)

    st.markdown("### (2) Cosine input $(f(x)=\cos(\omega x))$")
    st.latex(r"""
    T_{\ln}\{\cos(\omega x)\}(t)
    =
    \ln(t)-\sum_{n=1}^{\infty}\frac{(-1)^n\omega^{2n}}{2n}\,t^{-2n}
    """)

    st.latex(r"""
    =
    \ln(t)+\frac12\ln\!\left(1+\frac{\omega^2}{t^2}\right)
    """)

    st.latex(r"""
    \boxed{
    T_{\ln}\{\cos(\omega x)\}(t)
    =
    \frac12\ln(t^2+\omega^2)
    }
    """)

    st.markdown("### (3) Sine input $(f(x)=\sin(\omega x))$")
    st.latex(r"""
    T_{\ln}\{\sin(\omega x)\}(t)
    =
    -\sum_{n=0}^{\infty}\frac{(-1)^n}{2n+1}\left(\frac{\omega}{t}\right)^{2n+1}
    """)

    st.latex(r"""
    \boxed{
    T_{\ln}\{\sin(\omega x)\}(t)
    =
    -\arctan\!\left(\frac{\omega}{t}\right)
    }
    """)

    st.markdown("### (4) Hyperbolic cosine $(f(x)=\cosh(\omega x))$")
    st.latex(r"""
    T_{\ln}\{\cosh(\omega x)\}(t)
    =
    \ln(t)-\sum_{n=1}^{\infty}\frac{\omega^{2n}}{2n}t^{-2n}
    """)

    st.latex(r"""
    =
    \ln(t)+\frac12\ln\!\left(1-\frac{\omega^2}{t^2}\right)
    """)

    st.latex(r"""
    \boxed{
    T_{\ln}\{\cosh(\omega x)\}(t)
    =
    \frac12\ln(t^2-\omega^2)
    },\qquad |\omega|<t
    """)

    st.markdown("### (5) Hyperbolic sine $(f(x)=sinh(\omega x))$")
    st.latex(r"""
    T_{\ln}\{\sinh(\omega x)\}(t)
    =
    -\sum_{n=0}^{\infty}\frac{1}{2n+1}\left(\frac{\omega}{t}\right)^{2n+1}
    """)

    st.latex(r"""
    \boxed{
    T_{\ln}\{\sinh(\omega x)\}(t)
    =
    -\operatorname{artanh}\!\left(\frac{\omega}{t}\right)
    },\qquad |\omega|<t
    """)

    st.markdown("### (6) Mixed exponential--oscillatory inputs")

    st.latex(r"""
    e^{-\alpha x}\cos(\omega x)
    =
    \frac12\left(e^{(-\alpha+i\omega)x}+e^{(-\alpha-i\omega)x}\right)
    """)

    st.latex(r"""
    e^{-\alpha x}\sin(\omega x)
    =
    \frac1{2i}\left(e^{(-\alpha+i\omega)x}-e^{(-\alpha-i\omega)x}\right)
    """)

    st.latex(r"""
    T_{\ln}\{e^{-\alpha x}\cos(\omega x)\}(t)
    =
    \frac12\Big[\ln(t+\alpha-i\omega)+\ln(t+\alpha+i\omega)\Big]
    """)

    st.latex(r"""
    T_{\ln}\{e^{-\alpha x}\sin(\omega x)\}(t)
    =
    \frac1{2i}\Big[\ln(t+\alpha-i\omega)-\ln(t+\alpha+i\omega)\Big]
    """)

    st.latex(r"""
    \boxed{
    T_{\ln}\{e^{-\alpha x}\cos(\omega x)\}(t)
    =
    \frac12\ln\!\big((t+\alpha)^2+\omega^2\big)
    }
    """)

    st.latex(r"""
    \boxed{
    T_{\ln}\{e^{-\alpha x}\sin(\omega x)\}(t)
    =
    -\arctan\!\left(\frac{\omega}{t+\alpha}\right)
    }
    """)

    st.markdown(r"""
    The real part captures the logarithmic amplitude, while the imaginary part captures
    the phase angle of the complex logarithmic pair.
    """)

    st.subheader("Geometric interpretation")

    st.latex(r"""
    z_1=(t+\alpha)-i\omega,
    \qquad
    |z_1|=\sqrt{(t+\alpha)^2+\omega^2},
    \qquad
    \theta=-\arctan\!\left(\frac{\omega}{t+\alpha}\right)
    """)

    st.latex(r"""
    \ln(z_1)=\ln|z_1|+i\arg(z_1)
    """)

    st.latex(r"""
    T_{\ln}\{e^{-\alpha x}\cos(\omega x)\}(t)=\ln|z_1|
    """)

    st.latex(r"""
    T_{\ln}\{e^{-\alpha x}\sin(\omega x)\}(t)=\arg(z_1)
    """)

    st.subheader("Logarithmic transform of the Bessel function $(J_0(\omega x))$")

    st.markdown(r"""
    Starting from the series form
    """)

    st.latex(r"""
    J_0(\omega x)
    =
    \sum_{n=0}^{\infty}\frac{(-1)^n}{(n!)^2}\left(\frac{\omega x}{2}\right)^{2n}
    """)

    st.latex(r"""
    T_{\ln}\{J_0(\omega x)\}(t)
    =
    \ln t
    -
    \sum_{n=1}^{\infty}
    \frac{(-1)^n}{(n!)^2}
    \left(\frac{\omega}{2}\right)^{2n}
    (2n-1)!t^{-2n}
    """)

    st.markdown(r"""
    Using
    """)

    st.latex(r"""
    (2n-1)!=\frac{(2n)!}{2n},
    \qquad
    \frac{(2n)!}{(n!)^2}=\binom{2n}{n}
    """)

    st.latex(r"""
    T_{\ln}\{J_0(\omega x)\}(t)
    =
    \ln t
    -
    \sum_{n=1}^{\infty}
    \frac{(-1)^n}{2n}\binom{2n}{n}
    \left(\frac{\omega^2}{4t^2}\right)^n
    """)

    st.markdown(r"""
    Using the known identity
    """)

    st.latex(r"""
    \sum_{n=1}^{\infty}\frac1n\binom{2n}{n}z^n
    =
    -2\ln\!\left(\frac{1+\sqrt{1-4z}}{2}\right)
    """)

    st.markdown(r"""
    with $(z=-\omega^2/(4t^2))$, we obtain
    """)

    st.latex(r"""
    \boxed{
    T_{\ln}\{J_0(\omega x)\}(t)
    =
    \ln\!\left(\frac{t+\sqrt{t^2+\omega^2}}{2}\right)
    }
    """)

    st.markdown("### Integral form for $(J_0)$")

    st.latex(r"""
    T_{\ln}\{f\}(t)
    =
    f(0)\ln(t)-\int_0^\infty e^{-tu}\frac{f(u)-f(0)}{u}\,du
    """)

    st.markdown(r"""
    Since $(J_0(0)=1)$,
    """)

    st.latex(r"""
    \ln(t)-T_{\ln}\{J_0(\omega x)\}(t)
    =
    \int_0^\infty e^{-tu}\frac{J_0(\omega u)-1}{u}\,du
    """)

    st.latex(r"""
    \boxed{
    \int_0^\infty e^{-tu}\frac{J_0(\omega u)-1}{u}\,du
    =
    \ln\!\left(\frac{2t}{t+\sqrt{t^2+\omega^2}}\right)
    }
    """)

    st.subheader("Bessel input $(J_m(\omega x))$, integer $(m\ge1)$")

    st.latex(r"""
    J_m(\omega x)
    =
    \sum_{k=0}^{\infty}
    \frac{(-1)^k}{k!\,\Gamma(k+m+1)}
    \left(\frac{\omega}{2}\right)^{2k+m}x^{2k+m}
    """)

    st.latex(r"""
    T_{\ln}\{J_m(\omega x)\}(t)
    =
    \sum_{k=0}^{\infty}
    \frac{(-1)^k}{k!\,\Gamma(k+m+1)}
    \left(\frac{\omega}{2}\right)^{2k+m}
    (-\partial_t)^{2k+m}\ln(t)
    """)

    st.latex(r"""
    (-\partial_t)^{2k+m}\ln(t)=-(2k+m-1)!t^{-(2k+m)}
    """)

    st.markdown(r"""
    After summation, one gets the closed form
    """)

    st.latex(r"""
    \boxed{
    T_{\ln}\{J_m(\omega x)\}(t)
    =
    -\frac1m\left(\frac{\sqrt{t^2+\omega^2}-t}{\omega}\right)^m
    },\qquad m\in\mathbb{N},\;m\ge1
    """)

    st.markdown(r"""
    For $(m=0)$ the special form is
    """)

    st.latex(r"""
    \boxed{
    T_{\ln}\{J_0(\omega x)\}(t)
    =
    \ln\!\left(\frac{t+\sqrt{t^2+\omega^2}}{2}\right)
    }
    """)

    st.markdown(r"""
    Since $(J_v(0)=0)$ for $(v>0)$, the integral form becomes
    """)

    st.latex(r"""
    \boxed{
    T_{\ln}\{J_\nu(\omega x)\}(t)
    =
    \int_0^\infty e^{-tu}\frac{J_\nu(\omega u)}{u}\,du
    },\qquad \nu>0
    """)

    st.subheader("Case H: logarithmic transform of the sinc function")

    st.latex(r"""
    T_{\ln}\!\left\{\frac{\sin x}{x}\right\}(t)
    =
    -\sum_{n=1}^{\infty}(-1)^n\frac{1}{(2n)(2n+1)}\left(\frac1t\right)^{2n}
    """)

    st.markdown(r"""
    Let $(z=1/t)$. Split
    """)

    st.latex(r"""
    \frac{1}{(2n)(2n+1)}=\frac{1}{2n}-\frac{1}{2n+1}
    """)

    st.latex(r"""
    T_{\ln}\!\left\{\frac{\sin x}{x}\right\}(t)
    =
    -\frac12\sum_{n=1}^{\infty}\frac{(-1)^n z^{2n}}{n}
    +
    \sum_{n=1}^{\infty}\frac{(-1)^n z^{2n}}{2n+1}
    """)

    st.markdown(r"""
    Recognizing the sums,
    """)

    st.latex(r"""
    \sum_{n=1}^{\infty}\frac{(-1)^n z^{2n}}{n}=\ln(1+z^2)
    """)

    st.latex(r"""
    \sum_{n=1}^{\infty}\frac{(-1)^n z^{2n}}{2n+1}
    =
    -\frac{\arctan z}{z}+1
    """)

    st.markdown(r"""
    Hence
    """)

    st.latex(r"""
    T_{\ln}\!\left\{\frac{\sin x}{x}\right\}(t)
    =
    1-\frac{\arctan z}{z}-\frac12\ln(1+z^2),
    \qquad z=\frac1t
    """)

    st.latex(r"""
    \boxed{
    T_{\ln}\!\left\{\frac{\sin x}{x}\right\}(t)
    =
    1-t\arctan\!\left(\frac1t\right)-\frac12\ln\!\left(1+\frac1{t^2}\right)
    }
    """)

    st.markdown("### Integral representation for sinc")
    st.latex(r"""
    T_{\ln}\{f\}(t)
    =
    \ln(t)-\int_0^\infty e^{-tu}\frac{f(u)-1}{u}\,du
    """)

    st.markdown(r"""
    For $(f(x)=sin x/x)$,
    """)

    st.latex(r"""
    \boxed{
    T_{\ln}\!\left\{\frac{\sin x}{x}\right\}(t)
    =
    -\int_0^\infty e^{-tu}\frac{\frac{\sin u}{u}-1}{u}\,du
    =
    -\int_0^\infty e^{-tu}\frac{\sin u-u}{u^2}\,du
    }
    """)

    st.subheader("Closed-form results table")

    st.latex(r"""
    \begin{array}{|c|c|c|}
    \hline
    \text{Case} & \text{Input }f(x) & T_{\ln}\{f\}(t) \\
    \hline
    A & 1 & \ln(t) \\
    B & x & -\frac1t \\
    C & x^n & -(n-1)!t^{-n} \\
    D & e^{rx} & \ln(t-r) \\
    E & e^{-rx} & \ln(t+r) \\
    F & \sin(\omega x) & -\arctan\!\left(\frac{\omega}{t}\right) \\
    G & \cos(\omega x) & \frac12\ln(t^2+\omega^2) \\
    H & \sinh(\omega x) & -\operatorname{artanh}\!\left(\frac{\omega}{t}\right) \\
    I & \cosh(\omega x) & \frac12\ln(t^2-\omega^2) \\
    J1 & e^{-\alpha x}\cos(\omega x) & \frac12\ln((t+\alpha)^2+\omega^2) \\
    J2 & e^{-\alpha x}\sin(\omega x) & -\arctan\!\left(\frac{\omega}{t+\alpha}\right) \\
    K0 & J_0(\omega x) & \ln\!\left(\frac{t+\sqrt{t^2+\omega^2}}{2}\right) \\
    Km & J_m(\omega x) & -\frac1m\left(\frac{\sqrt{t^2+\omega^2}-t}{\omega}\right)^m \\
    \hline
    \end{array}
    """)

    st.subheader("Logarithmic shift law")

    st.markdown(r"""
    Define
    """)

    st.latex(r"""
    T_{\ln}\{f(x)\}(t)
    =
    f(0)\ln(t)-\int_0^\infty e^{-tu}\frac{f(u)-f(0)}{u}\,du
    """)

    st.markdown(r"""
    Then for any real constant \(r\),
    """)

    st.latex(r"""
    \boxed{
    T_{\ln}\{e^{rx}f(x)\}(t)=T_{\ln}\{f(x)\}(t-r)
    }
    """)

    st.markdown(r"""
    This mirrors the Laplace shift law, but here the shift occurs in the logarithmic
    seed variable \(t\).
    """)

    st.subheader("Logarithmic differentiation law")

    st.markdown(r"""
    For any differentiable function \(f(x)\), the transform of its derivative satisfies
    """)

    st.latex(r"""
    \boxed{
    T_{\ln}\{f'(x)\}(t)
    =
    t\frac{d}{dt}T_{\ln}\{f(x)\}(t)-f(0)
    }
    """)

    st.markdown(r"""
    Thus differentiation in \(x\) corresponds to a weighted scaling in the logarithmic
    domain, together with subtraction of the seed value.
    """)

    st.latex(r"""
    \begin{array}{|c|c|}
    \hline
    \text{Property} & \text{Logarithmic form} \\
    \hline
    \text{Shift} & T_{\ln}\{e^{rx}f(x)\}(t)=T_{\ln}\{f(x)\}(t-r) \\
    \text{Differentiation} & T_{\ln}\{f'(x)\}(t)=t\frac{d}{dt}T_{\ln}\{f(x)\}(t)-f(0) \\
    \hline
    \end{array}
    """)

    st.subheader("Derivative relation between logarithmic and Laplace transforms")

    st.markdown("### 18.1 Differential identity")

    st.latex(r"""
    T_{\ln}\{f\}(t)
    =
    f(0)\ln(t)-\int_0^\infty e^{-tu}\frac{f(u)-f(0)}{u}\,du
    """)

    st.markdown(r"""
    Differentiating with respect to \(t\) gives
    """)

    st.latex(r"""
    \frac{d}{dt}T_{\ln}\{f\}(t)
    =
    \frac{f(0)}{t}
    +
    \int_0^\infty e^{-tu}[f(u)-f(0)]\,du
    """)

    st.latex(r"""
    =
    \int_0^\infty e^{-tu}f(u)\,du
    """)

    st.latex(r"""
    \boxed{
    \frac{d}{dt}T_{\ln}\{f\}(t)=\mathcal{L}\{f\}(t)
    }
    """)

    st.markdown(r"""
    Hence the derivative of the logarithmic transform reproduces the classical Laplace
    transform evaluated at $(s=t)$.
    """)

    st.markdown("### 18.2 Mellin--logarithmic transform (rank $(n+s-1)$)")

    st.latex(r"""
    \mathcal{M}_{\ln}\{f\}(s,t)
    :=
    \sum_{n=0}^{\infty}a_n(-\partial_t)^{n+s-1}\ln(t)
    =
    -\sum_{n=0}^{\infty}a_n\frac{\Gamma(s+n-1)}{t^{s+n-1}}
    """)

    st.latex(r"""
    \Re(s)>1,\qquad t>0
    """)

    st.markdown(r"""
    Differentiating with respect to \(t\) yields
    """)

    st.latex(r"""
    \frac{\partial}{\partial t}\mathcal{M}_{\ln}\{f\}(s,t)
    =
    \sum_{n=0}^{\infty}a_n\frac{\Gamma(s+n)}{t^{s+n}}
    =
    \mathcal{M}\{f\}(s,t)
    """)

    st.latex(r"""
    \boxed{
    \frac{\partial}{\partial t}\mathcal{M}_{\ln}\{f\}(s,t)=\mathcal{M}\{f\}(s,t)
    }
    """)

    st.markdown("### Worked case: \($f(x)=e^{-x}$\)")
    st.latex(r"""
    a_n=\frac{(-1)^n}{n!}
    """)

    st.latex(r"""
    \mathcal{M}_{\ln}\{e^{-x}\}(s,t)
    =
    -\sum_{n=0}^{\infty}\frac{(-1)^n}{n!}\frac{\Gamma(s+n-1)}{t^{s+n-1}}
    """)

    st.latex(r"""
    =
    -\frac{\Gamma(s-1)}{t^{s-1}}
    \sum_{n=0}^{\infty}\frac{(s-1)_n}{n!}\left(-\frac1t\right)^n
    """)

    st.latex(r"""
    \sum_{n=0}^{\infty}\frac{(s-1)_n}{n!}z^n=(1-z)^{-(s-1)}
    """)

    st.latex(r"""
    \boxed{
    \mathcal{M}_{\ln}\{e^{-x}\}(s,t)
    =
    -\frac{\Gamma(s-1)}{(t+1)^{s-1}}
    },\qquad \Re(s)>1,\;t>0
    """)

    st.subheader("Hierarchical differentiation between transforms")

    st.markdown("### 19.1 Derivative bridge relation")
    st.latex(r"""
    \mathcal{M}_{\ln}\{f(x)\}(s,t)
    =
    -\sum_{n=0}^{\infty}a_n\frac{\Gamma(s+n-1)}{t^{s+n-1}}
    """)

    st.latex(r"""
    \frac{\partial}{\partial t}\mathcal{M}_{\ln}\{f(x)\}(s,t)
    =
    \sum_{n=0}^{\infty}a_n\Gamma(s+n)t^{-s-n}
    """)

    st.latex(r"""
    \boxed{
    \frac{\partial}{\partial t}\mathcal{M}_{\ln}\{f(x)\}(s,t)
    =
    \mathcal{M}\{f(x)\}(s,t)
    }
    """)

    st.markdown("### 19.2 Interpretation")
    st.markdown(r"""
    - The logarithmic--Mellin transform behaves as an integrated version of the Mellin operator.
    - The classical Mellin transform is its first \(t\)-derivative.
    """)

    st.markdown("### 19.3 Structural hierarchy")
    st.latex(r"""
    \frac{d}{dt}T_{\ln}\{f\}(t)=\mathcal{L}\{f\}(t),
    \qquad
    \frac{\partial}{\partial t}\mathcal{M}_{\ln}\{f\}(s,t)=\mathcal{M}\{f\}(s,t)
    """)

    st.markdown(r"""
    Thus, both logarithmic families (Laplace--logarithmic and Mellin--logarithmic)
    form hierarchical systems of transforms connected by differential operations in the
    auxiliary parameter \(t\).
    """)