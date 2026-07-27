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
t = sp.Symbol("t", real=True)
s = sp.Symbol("s", positive=True, real=True)
rho = sp.Symbol("rho", real=True)
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

    st.latex(r"f(t)=\sum_{n=0}^{\infty} a_n t^n")

    st.markdown("The framework is organized around four main transforms:")

    st.latex(r"\text{Laplace} \quad \text{Fourier} \quad \text{Mellin} \quad \text{Hankel}")

    st.markdown("The unifying principle is the planting rule:")

    st.latex(r"t^n \;\mapsto\; (-\partial)^n(\text{kernel})")

    st.markdown("Different transforms arise by changing the kernel and the effective rank.")

    st.latex(r"\text{Laplace: } (-\partial_s)^n\left(\frac{1}{s}\right)")
    st.latex(r"\text{Fourier: } \frac{1}{s},\;\frac{1}{\bar s},\quad s=\sigma+i\omega")
    st.latex(r"\text{Mellin: } (-\partial_s)^{n+\rho-1}\left(\frac{1}{s}\right)")
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
        f(t)=\sum_{n=0}^{\infty} a_n t^n
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
        \mathcal{L}\{f\}(s)=\int_{0}^{\infty} e^{-st} f(t)\,dt
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
        \frac{1}{n!}\int_{0}^{\infty} t^n e^{-st}\,dt,
        \qquad s>0.
        """
    )

    st.latex(
        r"""
        T\{f\}(s)
        =
        \sum_{n=0}^{\infty} a_n
        \int_{0}^{\infty} t^n e^{-st}\,dt
        =
        \int_{0}^{\infty} e^{-st}
        \left(\sum_{n=0}^{\infty} a_n t^n\right)dt.
        """
    )

    st.latex(
        r"""
        T\{f\}(s)=\int_{0}^{\infty} e^{-st} f(t)\,dt
        =
        \mathcal{L}\{f\}(s).
        """
    )

    st.info(
        "Below, each case is displayed in the same detailed style as the monograph: "
        "series, coefficient law, rank, planted operator form, summation, and final closed form."
    )


# ============================================================
# Laplace detailed symbolic cases (FULL original, with x→t)
# ============================================================
def get_laplace_cases():
    cases = {}

    cases["1"] = {
        "title": "Case A: Constant Function",
        "function": r"f(t)=1",
        "series": r"1=\sum_{n\ge 0} a_n t^n,\qquad a_0=1,\; a_{n>0}=0.",
        "coefficients": r"a_0=1,\qquad a_{n>0}=0.",
        "rank": r"D^0",
        "plant_sum": [
            r"T\{1\}(s)=D^0\!\left(\frac{1}{s}\right)",
        ],
        "closed_form": r"T\{1\}(s)=\frac{1}{s}",
        "condition": r"s>0",
    }

    cases["t^n"] = {
        "title": "Case B: Monomial",
        "function": r"f(t)=t^n",
        "series": r"t^n=\sum_{k\ge 0} a_k t^k,\qquad a_k=\delta_{k,n}",
        "coefficients": r"a_k=\delta_{k,n}\quad\text{(only the term }k=n\text{ survives)}.",
        "rank": r"D^n",
        "plant_sum": [
            r"T\{t^n\}(s)=(-D)^n\!\left(\frac{1}{s}\right)",
            r"D^n\!\left(\frac{1}{s}\right)=(-1)^n\frac{n!}{s^{n+1}}",
            r"(-D)^n\!\left(\frac{1}{s}\right)=\frac{n!}{s^{n+1}}",
        ],
        "closed_form": r"T\{t^n\}(s)=\frac{n!}{s^{n+1}}",
        "condition": r"s>0",
    }

    cases["e^(a t)"] = {
        "title": "Case C: Exponential Function",
        "function": r"f(t)=e^{at}",
        "series": r"e^{at}=\sum_{n\ge 0}\frac{a^n}{n!}t^n",
        "coefficients": r"a_n=\frac{a^n}{n!}",
        "rank": r"D^n",
        "plant_sum": [
            r"T\{e^{at}\}(s)=\sum_{n\ge 0}\frac{a^n}{n!}(-D)^n\!\left(\frac{1}{s}\right)",
            r"=\sum_{n\ge 0}\frac{a^n}{n!}\frac{n!}{s^{n+1}}",
            r"=\frac{1}{s}\sum_{n\ge 0}\left(\frac{a}{s}\right)^n",
        ],
        "closed_form": r"T\{e^{at}\}(s)=\frac{1}{s-a}",
        "condition": r"\left|\frac{a}{s}\right|<1",
    }

    cases["cos(b t)"] = {
        "title": "Case D: Cosine Function",
        "function": r"f(t)=\cos(bt)",
        "series": r"\cos(bt)=\sum_{n\ge 0}\frac{(-1)^n b^{2n}}{(2n)!}t^{2n}",
        "coefficients": r"a_{2n}=\frac{(-1)^n b^{2n}}{(2n)!}",
        "rank": r"D^{2n}",
        "plant_sum": [
            r"T\{\cos(bt)\}(s)=\sum_{n\ge 0} a_{2n}(-1)^{2n}D^{2n}\!\left(\frac{1}{s}\right)",
            r"=\sum_{n\ge 0}\frac{(-1)^n b^{2n}}{(2n)!}\frac{(2n)!}{s^{2n+1}}",
            r"=\frac{1}{s}\sum_{n\ge 0}\left(-\frac{b^2}{s^2}\right)^n",
        ],
        "closed_form": r"T\{\cos(bt)\}(s)=\frac{s}{s^2+b^2}",
        "condition": r"\left|\frac{b}{s}\right|<1",
    }

    cases["sin(b t)"] = {
        "title": "Case E: Sine Function",
        "function": r"f(t)=\sin(bt)",
        "series": r"\sin(bt)=\sum_{n\ge 0}\frac{(-1)^n b^{2n+1}}{(2n+1)!}t^{2n+1}",
        "coefficients": r"a_{2n+1}=\frac{(-1)^n b^{2n+1}}{(2n+1)!}",
        "rank": r"D^{2n+1}",
        "plant_sum": [
            r"T\{\sin(bt)\}(s)=\sum_{n\ge 0} a_{2n+1}(-1)^{2n+1}D^{2n+1}\!\left(\frac{1}{s}\right)",
            r"=\sum_{n\ge 0}\frac{(-1)^n b^{2n+1}}{(2n+1)!}\frac{(2n+1)!}{s^{2n+2}}",
            r"=\frac{b}{s^2}\sum_{n\ge 0}\left(-\frac{b^2}{s^2}\right)^n",
        ],
        "closed_form": r"T\{\sin(bt)\}(s)=\frac{b}{s^2+b^2}",
        "condition": r"\left|\frac{b}{s}\right|<1",
    }

    cases["t sin(b t)"] = {
        "title": "Case F: t sin(bt)",
        "function": r"f(t)=t\sin(bt)",
        "series": r"t\sin(bt)=\sum_{n\ge 0}\frac{(-1)^n b^{2n+1}}{(2n+1)!}t^{2n+2}",
        "coefficients": r"a_{2n+2}=\frac{(-1)^n b^{2n+1}}{(2n+1)!}",
        "rank": r"D^{2n+2}",
        "plant_sum": [
            r"T\{t\sin(bt)\}(s)=\sum_{n\ge 0} a_{2n+2}(-1)^{2n+2}D^{2n+2}\!\left(\frac{1}{s}\right)",
            r"=\sum_{n\ge 0}\frac{(-1)^n b^{2n+1}}{(2n+1)!}\frac{(2n+2)!}{s^{2n+3}}",
            r"=\frac{b}{s^3}\sum_{n\ge 0}(2n+2)\left(-\frac{b^2}{s^2}\right)^n",
            r"\sum_{n\ge 0}(2n+2)u^n=\frac{2}{(1-u)^2},\qquad u=-\frac{b^2}{s^2}",
        ],
        "closed_form": r"T\{t\sin(bt)\}(s)=\frac{2bs}{(s^2+b^2)^2}",
        "condition": r"s>|b|",
    }

    cases["t cos(b t)"] = {
        "title": "Case G: t cos(bt)",
        "function": r"f(t)=t\cos(bt)",
        "series": r"t\cos(bt)=\sum_{n\ge 0}\frac{(-1)^n b^{2n}}{(2n)!}t^{2n+1}",
        "coefficients": r"a_{2n+1}=\frac{(-1)^n b^{2n}}{(2n)!}",
        "rank": r"D^{2n+1}",
        "plant_sum": [
            r"T\{t\cos(bt)\}(s)=\sum_{n\ge 0} a_{2n+1}(-1)^{2n+1}D^{2n+1}\!\left(\frac{1}{s}\right)",
            r"=\sum_{n\ge 0}\frac{(-1)^n b^{2n}}{(2n)!}\frac{(2n+1)!}{s^{2n+2}}",
            r"=\frac{1}{s^2}\sum_{n\ge 0}(2n+1)\left(-\frac{b^2}{s^2}\right)^n",
            r"\sum_{n\ge 0}(2n+1)(-u)^n=\frac{1-u}{(1+u)^2},\qquad u=\frac{b^2}{s^2}",
        ],
        "closed_form": r"T\{t\cos(bt)\}(s)=\frac{s^2-b^2}{(s^2+b^2)^2}",
        "condition": r"s>|b|",
    }

    cases["cosh(b t)"] = {
        "title": "Case H1: Hyperbolic Cosine",
        "function": r"f(t)=\cosh(bt)",
        "series": r"\cosh(bt)=\sum_{n\ge 0}\frac{b^{2n}}{(2n)!}t^{2n}",
        "coefficients": r"a_{2n}=\frac{b^{2n}}{(2n)!}",
        "rank": r"D^{2n}",
        "plant_sum": [
            r"T\{\cosh(bt)\}(s)=\sum_{n\ge 0}\frac{b^{2n}}{(2n)!}(-D)^{2n}\!\left(\frac{1}{s}\right)",
            r"=\sum_{n\ge 0}\frac{b^{2n}}{(2n)!}\frac{(2n)!}{s^{2n+1}}",
            r"=\frac{1}{s}\sum_{n\ge 0}\left(\frac{b^2}{s^2}\right)^n",
        ],
        "closed_form": r"T\{\cosh(bt)\}(s)=\frac{s}{s^2-b^2}",
        "condition": r"\left|\frac{b}{s}\right|<1",
    }

    cases["sinh(b t)"] = {
        "title": "Case H2: Hyperbolic Sine",
        "function": r"f(t)=\sinh(bt)",
        "series": r"\sinh(bt)=\sum_{n\ge 0}\frac{b^{2n+1}}{(2n+1)!}t^{2n+1}",
        "coefficients": r"a_{2n+1}=\frac{b^{2n+1}}{(2n+1)!}",
        "rank": r"D^{2n+1}",
        "plant_sum": [
            r"T\{\sinh(bt)\}(s)=\sum_{n\ge 0}\frac{b^{2n+1}}{(2n+1)!}(-D)^{2n+1}\!\left(\frac{1}{s}\right)",
            r"=\sum_{n\ge 0}\frac{b^{2n+1}}{(2n+1)!}\frac{(2n+1)!}{s^{2n+2}}",
            r"=\frac{b}{s^2}\sum_{n\ge 0}\left(\frac{b^2}{s^2}\right)^n",
        ],
        "closed_form": r"T\{\sinh(bt)\}(s)=\frac{b}{s^2-b^2}",
        "condition": r"\left|\frac{b}{s}\right|<1",
    }

    cases["e^(a t) cos(bt)"] = {
        "title": "Shifted Case: e^{at} cos(bt)",
        "function": r"f(t)=e^{at}\cos(bt)",
        "series": r"\text{Use the shifted kernel } \frac{1}{s-a} \text{ instead of } \frac{1}{s}.",
        "coefficients": r"\text{The trigonometric coefficients remain unchanged; only the kernel is shifted.}",
        "rank": r"D^{2n}",
        "plant_sum": [
            r"T_a\{f\}(s)=\sum_{n\ge 0} a_{2n}(-D)^{2n}\!\left(\frac{1}{s-a}\right)",
            r"(-D)^n\!\left(\frac{1}{s-a}\right)=\frac{n!}{(s-a)^{n+1}}",
        ],
        "closed_form": r"T\{e^{at}\cos(bt)\}(s)=\frac{s-a}{(s-a)^2+b^2}",
        "condition": r"\left|\frac{b}{s-a}\right|<1",
    }

    cases["e^(a t) sin(bt)"] = {
        "title": "Shifted Case: e^{at} sin(bt)",
        "function": r"f(t)=e^{at}\sin(bt)",
        "series": r"\text{Use the shifted kernel } \frac{1}{s-a} \text{ instead of } \frac{1}{s}.",
        "coefficients": r"\text{The sine coefficients remain unchanged; only the kernel is shifted.}",
        "rank": r"D^{2n+1}",
        "plant_sum": [
            r"T_a\{f\}(s)=\sum_{n\ge 0} a_{2n+1}(-D)^{2n+1}\!\left(\frac{1}{s-a}\right)",
            r"(-D)^n\!\left(\frac{1}{s-a}\right)=\frac{n!}{(s-a)^{n+1}}",
        ],
        "closed_form": r"T\{e^{at}\sin(bt)\}(s)=\frac{b}{(s-a)^2+b^2}",
        "condition": r"\left|\frac{b}{s-a}\right|<1",
    }

    cases["sinc(b t) = sin(bt)/(b t)"] = {
        "title": "Case I: sinc(bt)",
        "function": r"f(t)=\mathrm{sinc}(bt)=\frac{\sin(bt)}{bt}",
        "series": r"\mathrm{sinc}(bt)=\sum_{n\ge 0}\frac{(-1)^n b^{2n}}{(2n+1)!}t^{2n}",
        "coefficients": r"a_{2n}=\frac{(-1)^n b^{2n}}{(2n+1)!}",
        "rank": r"D^{2n}",
        "plant_sum": [
            r"T\{\mathrm{sinc}(bt)\}(s)=\sum_{n\ge 0}a_{2n}(-D)^{2n}\!\left(\frac{1}{s}\right)",
            r"=\sum_{n\ge 0}\frac{(-1)^n b^{2n}}{(2n+1)!}\frac{(2n)!}{s^{2n+1}}",
            r"=\frac{1}{s}\sum_{n\ge 0}\frac{(-1)^n}{2n+1}\left(\frac{b}{s}\right)^{2n}",
            r"\sum_{n\ge 0}\frac{(-1)^n z^{2n+1}}{2n+1}=\arctan(z),\qquad z=\frac{b}{s}",
        ],
        "closed_form": r"T\{\mathrm{sinc}(bt)\}(s)=\frac{1}{b}\arctan\!\left(\frac{b}{s}\right)",
        "condition": r"\left|\frac{b}{s}\right|<1",
    }

    cases["(cos(bt)-1)/t"] = {
        "title": "Case J1: (cos(bt)-1)/t",
        "function": r"f(t)=\frac{\cos(bt)-1}{t}",
        "series": r"\frac{\cos(bt)-1}{t}=\sum_{n\ge 1}\frac{(-1)^n b^{2n}}{(2n)!}t^{2n-1}",
        "coefficients": r"a_{2n-1}=\frac{(-1)^n b^{2n}}{(2n)!}",
        "rank": r"D^{2n-1}",
        "plant_sum": [
            r"T\!\left\{\frac{\cos(bt)-1}{t}\right\}(s)=\sum_{n\ge 1}a_{2n-1}(-D)^{2n-1}\!\left(\frac{1}{s}\right)",
            r"=\sum_{n\ge 1}\frac{(-1)^n b^{2n}}{(2n)!}\frac{(2n-1)!}{s^{2n}}",
            r"=\frac{1}{2}\sum_{n\ge 1}\frac{1}{n}\left(-\frac{b^2}{s^2}\right)^n",
            r"\sum_{n\ge 1}\frac{r^n}{n}=-\ln(1-r)",
        ],
        "closed_form": r"T\!\left\{\frac{\cos(bt)-1}{t}\right\}(s)=-\frac{1}{2}\ln\!\left(1+\frac{b^2}{s^2}\right)",
        "condition": r"s>|b|",
    }

    cases["(cos(bt)-1)/t^2"] = {
        "title": "Case J2: (cos(bt)-1)/t^2",
        "function": r"f(t)=\frac{\cos(bt)-1}{t^2}",
        "series": r"\frac{\cos(bt)-1}{t^2}=\sum_{n\ge 1}\frac{(-1)^n b^{2n}}{(2n)!}t^{2n-2}",
        "coefficients": r"a_{2n-2}=\frac{(-1)^n b^{2n}}{(2n)!}",
        "rank": r"D^{2n-2}",
        "plant_sum": [
            r"T\!\left\{\frac{\cos(bt)-1}{t^2}\right\}(s)=\sum_{n\ge 1}a_{2n-2}(-D)^{2n-2}\!\left(\frac{1}{s}\right)",
            r"=\sum_{n\ge 1}\frac{(-1)^n b^{2n}}{(2n)!}\frac{(2n-2)!}{s^{2n-1}}",
            r"=s\sum_{n\ge 1}\frac{(-1)^n}{2n(2n-1)}\left(\frac{b^2}{s^2}\right)^n",
            r"\frac{1}{2n(2n-1)}=-\frac{1}{2n}+\frac{1}{2n-1}",
        ],
        "closed_form": r"T\!\left\{\frac{\cos(bt)-1}{t^2}\right\}(s)=\frac{s}{2}\ln\!\left(1+\frac{b^2}{s^2}\right)-b\arctan\!\left(\frac{b}{s}\right)",
        "condition": r"s>|b|",
    }

    cases["J0(k t)"] = {
        "title": "Case J3: Bessel Function J_0(kt)",
        "function": r"f(t)=J_0(kt)",
        "series": r"J_0(kt)=\sum_{n\ge 0}\frac{(-1)^n}{(n!)^2}\left(\frac{kt}{2}\right)^{2n}",
        "coefficients": r"a_{2n}=\frac{(-1)^n}{(n!)^2}\left(\frac{k}{2}\right)^{2n}",
        "rank": r"D^{2n}",
        "plant_sum": [
            r"T\{J_0(kt)\}(s)=\sum_{n\ge 0}a_{2n}(-D)^{2n}\!\left(\frac{1}{s}\right)",
            r"=\sum_{n\ge 0}\frac{(-1)^n}{(n!)^2}\left(\frac{k}{2}\right)^{2n}\frac{(2n)!}{s^{2n+1}}",
            r"=\frac{1}{s}\sum_{n\ge 0}\frac{(2n)!}{(n!)^2}\left(-\frac{k^2}{4s^2}\right)^n",
            r"\sum_{n\ge 0}\frac{(2n)!}{(n!)^2}z^n=\frac{1}{\sqrt{1-4z}}",
        ],
        "closed_form": r"T\{J_0(kt)\}(s)=\frac{1}{\sqrt{s^2+k^2}}",
        "condition": r"s>0",
    }

    cases["J_nu(k t)"] = {
        "title": "General Bessel Case: J_\\nu(kt)",
        "function": r"f(t)=J_\nu(kt)",
        "series": r"J_\nu(kt)=\sum_{n\ge 0}\frac{(-1)^n}{n!\,\Gamma(n+\nu+1)}\left(\frac{kt}{2}\right)^{2n+\nu}",
        "coefficients": r"a_{2n+\nu}=\frac{(-1)^n}{n!\,\Gamma(n+\nu+1)}\left(\frac{k}{2}\right)^{2n+\nu}",
        "rank": r"D^{2n+\nu}",
        "plant_sum": [
            r"T\{J_\nu(kt)\}(s)=\sum_{n\ge 0}a_{2n+\nu}(-D)^{2n+\nu}\!\left(\frac{1}{s}\right)",
            r"=\sum_{n\ge 0}\frac{(-1)^n}{n!\,\Gamma(n+\nu+1)}\left(\frac{k}{2}\right)^{2n+\nu}\frac{\Gamma(2n+\nu+1)}{s^{2n+\nu+1}}",
            r"\text{After summation, the planted series reproduces the classical Laplace–Bessel closed form.}",
        ],
        "closed_form": r"T\{J_\nu(kt)\}(s)=\frac{\left(\sqrt{s^2+k^2}-s\right)^\nu}{k^\nu\sqrt{s^2+k^2}}",
        "condition": r"s>0",
    }

    return cases


# ============================================================
# Fourier data (FULL original, no changes)
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
# Mellin data (NEW from screenshots, with rho and t)
# ============================================================
def mellin_transform_data(choice, params):
    rho_local = sp.Symbol("rho", real=True)
    s_local = sp.Symbol("s", positive=True, real=True)
    t_local = sp.Symbol("t", real=True)
    a_local = sp.Symbol("a", positive=True, real=True)

    if choice == "e^{-t}":
        f = sp.exp(-t_local)
        return {
            "f": f,
            "series": r"e^{-t} = \sum_{n=0}^{\infty} \frac{(-1)^n}{n!} t^n",
            "coeff": r"a_n = \frac{(-1)^n}{n!}",
            "rank": r"n",
            "planted": r"MT\{e^{-t}\}(\rho,s) = \sum_{n=0}^{\infty} \frac{(-1)^n}{n!} \frac{\Gamma(\rho+n)}{s^{\rho+n}}",
            "closed": sp.gamma(rho_local) / (s_local + 1) ** rho_local,
            "classical": sp.gamma(rho_local),
        }

    if choice == "e^{-at}":
        a_val = to_expr(params["a"])
        f = sp.exp(-a_val * t_local)
        return {
            "f": f,
            "series": r"e^{-at} = \sum_{n=0}^{\infty} \frac{(-a)^n}{n!} t^n",
            "coeff": r"a_n = \frac{(-a)^n}{n!}",
            "rank": r"n",
            "planted": r"MT\{e^{-at}\}(\rho,s) = \sum_{n=0}^{\infty} \frac{(-a)^n}{n!} \frac{\Gamma(\rho+n)}{s^{\rho+n}}",
            "closed": sp.gamma(rho_local) / (s_local + a_val) ** rho_local,
            "classical": a_val ** (-rho_local) * sp.gamma(rho_local),
        }

    if choice == "e^{it}":
        f = sp.exp(sp.I * t_local)
        return {
            "f": f,
            "series": r"e^{it} = \sum_{n=0}^{\infty} \frac{i^n}{n!} t^n",
            "coeff": r"a_n = \frac{i^n}{n!}",
            "rank": r"n",
            "planted": r"MT\{e^{it}\}(\rho,s) = \sum_{n=0}^{\infty} \frac{i^n}{n!} \frac{\Gamma(\rho+n)}{s^{\rho+n}}",
            "closed": sp.gamma(rho_local) / (s_local - sp.I) ** rho_local,
            "classical": sp.gamma(rho_local) * sp.exp(sp.I * sp.pi * rho_local / 2),
        }

    if choice == "cos(t)":
        f = sp.cos(t_local)
        return {
            "f": f,
            "series": r"\cos t = \sum_{n=0}^{\infty} \frac{(-1)^n}{(2n)!} t^{2n}",
            "coeff": r"a_{2n} = \frac{(-1)^n}{(2n)!}",
            "rank": r"2n",
            "planted": r"MT\{\cos t\}(\rho,s) = \sum_{n=0}^{\infty} \frac{(-1)^n}{(2n)!} \frac{\Gamma(\rho+2n)}{s^{\rho+2n}}",
            "closed": sp.gamma(rho_local) * (s_local**2 + 1) ** (-rho_local / 2) * sp.cos(rho_local * sp.atan(1 / s_local)),
            "classical": sp.gamma(rho_local) * sp.cos(sp.pi * rho_local / 2),
        }

    if choice == "sin(t)":
        f = sp.sin(t_local)
        return {
            "f": f,
            "series": r"\sin t = \sum_{n=0}^{\infty} \frac{(-1)^n}{(2n+1)!} t^{2n+1}",
            "coeff": r"a_{2n+1} = \frac{(-1)^n}{(2n+1)!}",
            "rank": r"2n+1",
            "planted": r"MT\{\sin t\}(\rho,s) = \sum_{n=0}^{\infty} \frac{(-1)^n}{(2n+1)!} \frac{\Gamma(\rho+2n+1)}{s^{\rho+2n+1}}",
            "closed": sp.gamma(rho_local) * (s_local**2 + 1) ** (-rho_local / 2) * sp.sin(rho_local * sp.atan(1 / s_local)),
            "classical": sp.gamma(rho_local) * sp.sin(sp.pi * rho_local / 2),
        }

    return {
        "f": sp.exp(-t_local),
        "series": r"e^{-t} = \sum_{n=0}^{\infty} \frac{(-1)^n}{n!} t^n",
        "coeff": r"a_n = \frac{(-1)^n}{n!}",
        "rank": r"n",
        "planted": r"MT\{e^{-t}\}(\rho,s) = \sum_{n=0}^{\infty} \frac{(-1)^n}{n!} \frac{\Gamma(\rho+n)}{s^{\rho+n}}",
        "closed": sp.gamma(rho_local) / (s_local + 1) ** rho_local,
        "classical": sp.gamma(rho_local),
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
        "Fractional Derivative Link",
    ],
)


# ============================================================
# Header
# ============================================================
st.title("Operator Transform Laboratory")
st.subheader("Kernel Geometry and Planted Differential Structures")
st.caption("Interactive first version for Laplace, Fourier, and Mellin transforms")
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
# SECTION 3: Fourier Transform
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

    with st.expander("Detailed derivation: from one-sided Laplace to regulated bilateral Fourier", expanded=True):
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

# ============================================================
# SECTION 4: Mellin Transform (NEW from screenshots)
# ============================================================
elif section == "Operator Mellin Transform":
    st.header("4. Operator-Based Mellin Transform")

    st.markdown(r"""
    In this section we introduce an operator version of the Mellin transform built by
    planting Maclaurin coefficients inside a Gamma-weighted kernel. The resulting framework
    reproduces the classical Mellin table directly from a planted series, demonstrating that
    the operator viewpoint can be both unifying and computationally efficient.
    """)

    st.latex(r"""
    t^n \;\longmapsto\; (-\partial_s)^{\,n+\rho-1}\!\left(\frac{1}{s}\right)
    = \frac{\Gamma(n+\rho)}{s^{\,n+\rho}}
    """)

    st.subheader("Differential Derivation of the Mellin Integral Form (Planting Method)")

    st.markdown("### 1. Differential Planting Foundation")
    st.markdown(r"""
    Let $f(t)$ be an analytic function expanded as a Maclaurin series:
    """)
    st.latex(r"f(t)=\sum_{n=0}^{\infty} a_n t^n")

    st.markdown(r"""
    The Mellin transform is defined in its differential planting form by
    """)
    st.latex(r"""
    MT\{f\}(\rho,s)=\sum_{n=0}^{\infty} a_n (-\partial_s)^{\,s+n-1}\!\left(\frac{1}{s}\right),
    \qquad s>0
    """)

    st.markdown(r"""
    Each term $a_n t^n$ of the series is planted as a fractional derivative of order
    $(\rho+n-1)$ applied to the seed function $(1/s)$.
    """)

    st.markdown("### 2. Fractional Derivative Identity")
    st.markdown(r"""
    For $(Re(\alpha)>-1)$, the fractional derivative of $(1/s)$ satisfies
    """)
    st.latex(r"""
    (-\partial_s)^\alpha\!\left(\frac{1}{s}\right)=\frac{\Gamma(\alpha+1)}{s^{\alpha+1}}
    """)

    st.markdown(r"Substituting $(\alpha=\rho+n-1)$ gives")
    st.latex(r"""
    (-\partial_s)^{\,\rho+n-1}\!\left(\frac{1}{s}\right)=\frac{\Gamma(\rho+n)}{s^{\rho+n}}
    """)

    st.markdown(r"Hence the differential planting law becomes")
    st.latex(r"""
    MT\{f\}(\rho,s)=\sum_{n=0}^{\infty} a_n \frac{\Gamma(\rho+n)}{s^{\rho+n}}
    """)

    st.markdown("### 3. Transition to the Integral Representation")
    st.markdown(r"Using the Gamma integral identity")
    st.latex(r"""
    \frac{\Gamma(\rho+n)}{s^{\rho+n}}=\int_0^\infty t^{\rho+n-1}e^{-st}\,dt,
    \qquad s>0
    """)

    st.markdown(r"we substitute this into the planted series:")
    st.latex(r"""
    MT\{f\}(\rho,s)=\sum_{n=0}^{\infty} a_n \int_0^\infty t^{\rho+n-1}e^{-st}\,dt
    """)

    st.markdown("### 4. Interchanging Sum and Integral")
    st.markdown(r"""
    Under absolute convergence (for standard analytic functions), the sum and the integral may
    be interchanged (Tonelli–Fubini theorem):
    """)
    st.latex(r"""
    MT\{f\}(\rho,s)=\int_0^\infty \left(\sum_{n=0}^{\infty} a_n t^n\right)t^{\rho-1}e^{-st}\,dt
    """)

    st.markdown(r"Recognizing the internal series as $f(t)$ yields the canonical integral form:")
    st.latex(r"""
    MT\{f\}(\rho,s)=\int_0^\infty f(t)\,t^{\rho-1}e^{-st}\,dt
    """)

    st.markdown("### 5. Classical Mellin Limit")
    st.markdown(r"Taking the limit as $(s\to0^+)$ removes the exponential regulator:")
    st.latex(r"""
    \lim_{s\to0^+} MT\{f\}(\rho,s)=\int_0^\infty f(t)t^{\rho-1}\,dt
    """)

    st.markdown(r"""
    which is precisely the classical Mellin transform.

    Perfect agreement between the planted differential form and the classical integral form
    confirms the internal consistency.
    """)

    st.markdown("### 7. Interpretation")
    st.markdown(r"""
    This derivation shows that the Mellin integral form is not a primitive definition, but a
    natural consequence of the differential planting law. The exponential regulator $(e^{-st})$
    emerges automatically from the internal structure of the fractional derivative of $(1/s)$.
    Thus, the differential approach reconstructs the Mellin transform entirely from planted
    derivative dynamics, bridging the discrete hierarchy of derivatives with the continuous
    integral hierarchy governed by the Gamma function.
    """)

    st.markdown(r"""
    **Additional note.** The exponential regulator $(e^{-st})$ appearing in the integral form
    of the Mellin operator is not an externally added convergence factor, but rather an
    intrinsic byproduct of the differential planting sequence. This shows that the exponential
    damping in Mellin's transform can be reconstructed endogenously from the internal structure
    of the fractional derivatives acting on $(1/s)$.
    """)

    st.subheader("Summary of the Operator Mellin Transform")

    st.markdown("### 6.3 Integral form")
    st.markdown(r"The classical Mellin transform is defined as")
    st.latex(r"""
    \mathcal{M}\{f(t)\}(\rho)=\int_0^\infty f(t)t^{\rho-1}\,dt
    """)
    st.markdown(r"valid whenever the integral converges.")
    st.markdown(r"In the operator framework, we introduce an exponential regulator:")
    st.latex(r"""
    MT\{f(t)\}(\rho,s)=\int_0^\infty f(t)t^{\rho-1}e^{-st}\,dt,\qquad s>0
    """)

    st.markdown("### 6.4 Series form (planting law)")
    st.markdown(r"If $(f(t))$ has a Maclaurin expansion")
    st.latex(r"f(t)=\sum_{n=0}^{\infty} a_n t^n")
    st.markdown(r"then the transform is obtained by planting each term:")
    st.latex(r"""
    MT\{f\}(\rho,s)=\sum_{n=0}^{\infty} a_n \frac{\Gamma(\rho+n)}{s^{\rho+n}}
    """)
    st.markdown(r"""
    This is the series planting law: each \(t^n\) contributes a gamma factor $(\Gamma(\rho+n))$
    and a regulator $(s^{-(\rho+n)})$.
    """)

    st.subheader("Interactive Mellin Case Explorer")

    mellin_case = st.selectbox(
        "Choose a symbolic Mellin case",
        [
            r"e^{-t}",
            r"e^{-at}",
            r"e^{it}",
            r"\cos t",
            r"\sin t",
        ],
        key="mellin_case_detailed"
    )

    rho_sym = sp.Symbol("rho", real=True)
    s_sym = sp.Symbol("s", positive=True, real=True)
    t_sym = sp.Symbol("t", positive=True, real=True)
    a_sym = sp.Symbol("a", positive=True, real=True)

    if mellin_case == r"e^{-t}":
        f_expr = sp.exp(-t_sym)
        series_text = r"e^{-t}=\sum_{n=0}^{\infty}\frac{(-1)^n}{n!}t^n"
        coeff_text = r"a_n=\frac{(-1)^n}{n!}"
        rank_text = r"n"
        plant_steps = [
            r"MT\{e^{-t}\}(\rho,s)=\sum_{n=0}^{\infty}\frac{(-1)^n}{n!}\frac{\Gamma(\rho+n)}{s^{\rho+n}},\qquad s>0",
            r"\Gamma(\rho+n)=\Gamma(\rho)(\rho)_n,\qquad (\rho)_n=\frac{\Gamma(\rho+n)}{\Gamma(\rho)}",
            r"MT\{e^{-t}\}(\rho,s)=\Gamma(\rho)s^{-\rho}\sum_{n=0}^{\infty}\frac{(\rho)_n}{n!}\left(-\frac{1}{s}\right)^n",
            r"(1+z)^{-\rho}=\sum_{n=0}^{\infty}\frac{(\rho)_n}{n!}(-z)^n",
            r"\sum_{n=0}^{\infty}\frac{(\rho)_n}{n!}\left(-\frac{1}{s}\right)^n=\left(1+\frac{1}{s}\right)^{-\rho}",
            r"MT\{e^{-t}\}(\rho,s)=\Gamma(\rho)s^{-\rho}\left(1+\frac{1}{s}\right)^{-\rho}",
            r"=\frac{\Gamma(\rho)}{(s+1)^{\rho}}",
        ]
        closed_text = r"MT\{e^{-t}\}(\rho,s)=\frac{\Gamma(\rho)}{(s+1)^{\rho}}"
        classical_text = r"\lim_{s\to0^+}MT\{e^{-t}\}(\rho,s)=\Gamma(\rho)"
        extra_note = r"\int_0^\infty e^{-t}t^{\rho-1}\,dt=\Gamma(\rho)"

    elif mellin_case == r"e^{-at}":
        f_expr = sp.exp(-a_sym * t_sym)
        series_text = r"e^{-at}=\sum_{n=0}^{\infty}\frac{(-a)^n}{n!}t^n"
        coeff_text = r"a_n=\frac{(-a)^n}{n!}"
        rank_text = r"n"
        plant_steps = [
            r"MT\{e^{-at}\}(\rho,s)=\sum_{n=0}^{\infty}\frac{(-a)^n}{n!}\frac{\Gamma(\rho+n)}{s^{\rho+n}},\qquad s>0",
            r"\Gamma(\rho+n)=\Gamma(\rho)(\rho)_n",
            r"MT\{e^{-at}\}(\rho,s)=\Gamma(\rho)s^{-\rho}\sum_{n=0}^{\infty}\frac{(\rho)_n}{n!}\left(-\frac{a}{s}\right)^n",
            r"(1+z)^{-\rho}=\sum_{n=0}^{\infty}\frac{(\rho)_n}{n!}(-z)^n",
            r"\sum_{n=0}^{\infty}\frac{(\rho)_n}{n!}\left(-\frac{a}{s}\right)^n=\left(1+\frac{a}{s}\right)^{-\rho}",
            r"MT\{e^{-at}\}(\rho,s)=\Gamma(\rho)s^{-\rho}\left(1+\frac{a}{s}\right)^{-\rho}",
            r"=\frac{\Gamma(\rho)}{(s+a)^{\rho}}",
        ]
        closed_text = r"MT\{e^{-at}\}(\rho,s)=\frac{\Gamma(\rho)}{(s+a)^{\rho}}"
        classical_text = r"\lim_{s\to0^+}MT\{e^{-at}\}(\rho,s)=a^{-\rho}\Gamma(\rho)"
        extra_note = r"\int_0^\infty e^{-at}t^{\rho-1}\,dt=a^{-\rho}\Gamma(\rho)"

    elif mellin_case == r"e^{it}":
        f_expr = sp.exp(sp.I * t_sym)
        series_text = r"e^{it}=\sum_{n=0}^{\infty}\frac{i^n}{n!}t^n"
        coeff_text = r"a_n=\frac{i^n}{n!}"
        rank_text = r"n"
        plant_steps = [
            r"MT\{e^{it}\}(\rho,s)=\sum_{n=0}^{\infty}\frac{i^n}{n!}\frac{\Gamma(\rho+n)}{s^{\rho+n}},\qquad s>0",
            r"\Gamma(\rho+n)=\Gamma(\rho)(\rho)_n",
            r"MT\{e^{it}\}(\rho,s)=\Gamma(\rho)s^{-\rho}\sum_{n=0}^{\infty}\frac{(\rho)_n}{n!}\left(\frac{i}{s}\right)^n",
            r"(1-z)^{-\rho}=\sum_{n=0}^{\infty}\frac{(\rho)_n}{n!}z^n",
            r"\sum_{n=0}^{\infty}\frac{(\rho)_n}{n!}\left(\frac{i}{s}\right)^n=\left(1-\frac{i}{s}\right)^{-\rho}",
            r"MT\{e^{it}\}(\rho,s)=\Gamma(\rho)s^{-\rho}\left(1-\frac{i}{s}\right)^{-\rho}",
            r"=\frac{\Gamma(\rho)}{(s-i)^{\rho}},\qquad s>0",
            r"r=\sqrt{s^2+1},\qquad \theta=\arctan\left(\frac{1}{s}\right)",
            r"\frac{1}{(s-i)^{\rho}}=r^{-\rho}e^{i\rho\theta}",
            r"MT\{e^{it}\}(\rho,s)=\Gamma(\rho)r^{-\rho}\left[\cos(\rho\theta)+i\sin(\rho\theta)\right]",
            r"MT\{\cos t\}(\rho,s)=\Gamma(\rho)r^{-\rho}\cos(\rho\theta)",
            r"MT\{\sin t\}(\rho,s)=\Gamma(\rho)r^{-\rho}\sin(\rho\theta)",
        ]
        closed_text = r"MT\{e^{it}\}(\rho,s)=\frac{\Gamma(\rho)}{(s-i)^{\rho}}"
        classical_text = r"\lim_{s\to0^+}MT\{e^{it}\}(\rho,s)=\Gamma(\rho)e^{i\pi\rho/2}"
        extra_note = r"\int_0^\infty t^{\rho-1}e^{-(s-i)t}\,dt=\frac{\Gamma(\rho)}{(s-i)^{\rho}}"

    elif mellin_case == r"\cos t":
        f_expr = sp.cos(t_sym)
        series_text = r"\cos t=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n)!}t^{2n}"
        coeff_text = r"a_{2n}=\frac{(-1)^n}{(2n)!}"
        rank_text = r"2n"
        plant_steps = [
            r"MT\{\cos t\}(\rho,s)=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n)!}\frac{\Gamma(\rho+2n)}{s^{\rho+2n}}",
            r"\Gamma(\rho+2n)=\Gamma(\rho)2^{2n}\left(\frac{\rho}{2}\right)_n\left(\frac{\rho+1}{2}\right)_n",
            r"(2n)!=2^{2n}\left(\frac12\right)_n n!",
            r"MT\{\cos t\}(\rho,s)=\Gamma(\rho)s^{-\rho}\,{}_2F_1\!\left(\frac{\rho}{2},\frac{\rho+1}{2};\frac12;-\frac{1}{s^2}\right)",
            r"\text{Equivalently, }MT\{\cos t\}(\rho,s)=\Gamma(\rho)(s^2+1)^{-\rho/2}\cos\!\left(\rho\arctan\!\frac{1}{s}\right)",
        ]
        closed_text = r"MT\{\cos t\}(\rho,s)=\Gamma(\rho)(s^2+1)^{-\rho/2}\cos\!\left(\rho\arctan\!\frac{1}{s}\right)"
        classical_text = r"\lim_{s\to0^+}MT\{\cos t\}(\rho,s)=\Gamma(\rho)\cos\!\left(\frac{\pi\rho}{2}\right),\qquad 0<\Re(\rho)<1"
        extra_note = r"\int_0^\infty t^{\rho-1}\cos t\,dt=\Gamma(\rho)\cos\!\left(\frac{\pi\rho}{2}\right),\qquad 0<\rho<1"

    elif mellin_case == r"\sin t":
        f_expr = sp.sin(t_sym)
        series_text = r"\sin t=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n+1)!}t^{2n+1}"
        coeff_text = r"a_{2n+1}=\frac{(-1)^n}{(2n+1)!}"
        rank_text = r"2n+1"
        plant_steps = [
            r"MT\{\sin t\}(\rho,s)=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n+1)!}\frac{\Gamma(\rho+2n+1)}{s^{\rho+2n+1}}",
            r"\Gamma(\rho+2n+1)=\Gamma(\rho+1)2^{2n}\left(\frac{\rho+1}{2}\right)_n\left(\frac{\rho+2}{2}\right)_n",
            r"(2n+1)!=2^{2n}\left(\frac32\right)_n n!",
            r"MT\{\sin t\}(\rho,s)=\Gamma(\rho+1)s^{-(\rho+1)}\,{}_2F_1\!\left(\frac{\rho+1}{2},\frac{\rho+2}{2};\frac32;-\frac{1}{s^2}\right)",
            r"\text{Equivalently, }MT\{\sin t\}(\rho,s)=\Gamma(\rho)(s^2+1)^{-\rho/2}\sin\!\left(\rho\arctan\!\frac{1}{s}\right)",
        ]
        closed_text = r"MT\{\sin t\}(\rho,s)=\Gamma(\rho)(s^2+1)^{-\rho/2}\sin\!\left(\rho\arctan\!\frac{1}{s}\right)"
        classical_text = r"\lim_{s\to0^+}MT\{\sin t\}(\rho,s)=\Gamma(\rho)\sin\!\left(\frac{\pi\rho}{2}\right),\qquad 0<\Re(\rho)<1"
        extra_note = r"\int_0^\infty t^{\rho-1}\sin t\,dt=\Gamma(\rho)\sin\!\left(\frac{\pi\rho}{2}\right),\qquad 0<\rho<1"

    st.markdown("### Function")
    st.latex(r"f(t)=" + sp.latex(f_expr))

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

    # 6.10 Power Multiplication Property
    st.divider()
    st.subheader("6.10 Power Multiplication Property")
    st.markdown(r"""
    Let the function be represented by its power series expansion
    """)
    st.latex(r"f(t) = \sum_{n=0}^{\infty} a_n t^n")
    st.markdown(r"""
    Multiplication by a power \(t^m\) gives
    """)
    st.latex(r"t^m f(t) = \sum_{n=0}^{\infty} a_n t^{n+m}")
    st.markdown(r"""
    According to the generated kernel rule,
    """)
    st.latex(r"t^k \to \frac{\Gamma(\rho + k)}{s^{\rho + k}}")
    st.markdown(r"""
    Hence the multiplication property is
    """)
    st.latex(r"\mathcal{MT}_{(\rho,s)}\{t^m f(t)\} = \sum_{n=0}^{\infty} a_n \frac{\Gamma(\rho + n + m)}{s^{\rho + n + m}}")
    st.markdown(r"""
    For \(f(t) = e^{-t}\):
    """)
    st.latex(r"\mathcal{MT}_{(\rho,s)}\{t^m e^{-t}\} = \frac{\Gamma(\rho + m)}{(s + 1)^{\rho + m}}")
    st.latex(r"\lim_{s\to0^+} \mathcal{MT}_{(\rho,s)}\{t^m e^{-t}\} = \Gamma(\rho + m)")

    # Comparison Table
    st.divider()
    st.subheader("Comparison Table")
    st.markdown("""
    | Function \(f(t)\) | Series Expansion | Planted Operator Form | Classical Mellin Result |
    |-------------------|-----------------|----------------------|------------------------|
    | \(e^{-t}\) | \(\sum_{n=0}^{\infty} \frac{(-1)^n t^n}{n!}\) | \(\sum_{n=0}^{\infty} \frac{(-1)^n \Gamma(\rho+n)}{n! s^{\rho+n}}\) | \(\Gamma(\rho),\ \rho>0\) |
    | \(e^{-at}\) | \(\sum_{n=0}^{\infty} \frac{(-a)^n t^n}{n!}\) | \(\sum_{n=0}^{\infty} \frac{(-a)^n \Gamma(\rho+n)}{n! s^{\rho+n}}\) | \(a^{-\rho}\Gamma(\rho),\ a>0,\ \rho>0\) |
    | \(\cos t\) | \(\sum_{n=0}^{\infty} \frac{(-1)^n t^{2n}}{(2n)!}\) | \(\sum_{n=0}^{\infty} \frac{(-1)^n \Gamma(\rho+2n)}{(2n)! s^{\rho+2n}}\) | \(\Gamma(\rho)\cos(\pi\rho/2),\ 0<\rho<1\) |
    | \(\sin t\) | \(\sum_{n=0}^{\infty} \frac{(-1)^n t^{2n+1}}{(2n+1)!}\) | \(\sum_{n=0}^{\infty} \frac{(-1)^n \Gamma(\rho+2n+1)}{(2n+1)! s^{\rho+2n+1}}\) | \(\Gamma(\rho)\sin(\pi\rho/2),\ 0<\rho<1\) |
    | \(t^m f(t)\) | \(t^m\sum a_n t^n = \sum a_n t^{n+m}\) | \(\sum a_n \Gamma(\rho+n+m)/s^{\rho+n+m}\) | \(F(\rho+m)\) |
    """)

# ============================================================
# SECTION 5: Fractional Derivative Link
# ============================================================
elif section == "Fractional Derivative Link":
    st.header("5. Fractional Derivative Link")

    st.markdown(r"""
    We note the key operator identity:
    """)
    st.latex(r"(-\partial_s)^n\left(\frac{1}{s}\right) = \frac{n!}{s^{n+1}}")

    st.markdown(r"""
    Fractional calculus extends this to arbitrary \(\alpha\):
    """)
    st.latex(r"(-\partial_s)^\alpha\left(\frac{1}{s}\right) = \frac{\Gamma(\alpha+1)}{s^{\alpha+1}},\quad \Re(\alpha)>-1")

    st.subheader("Key Implication")
    st.markdown(r"""
    By setting \(\alpha = n + \rho - 1\), we obtain
    """)
    st.latex(r"(-\partial_s)^{n+\rho-1}\left(\frac{1}{s}\right) = \frac{\Gamma(n+\rho)}{s^{n+\rho}}")
    st.markdown("This is exactly the Mellin planting law.")

    st.subheader("Summary: Unified Operator Planting")

    st.markdown("The parameter \(\alpha\) denotes the planting rank, which may be integer, fractional, or bilateral, depending on the underlying transform.")

    st.markdown("""
    | Transform | Rank \(\alpha\) | Operator Planting Form |
    |-----------|---------------|----------------------|
    | Laplace | \(\alpha = n\) | \((-\partial_s)^\alpha(1/s) = \Gamma(\alpha+1)/s^{\alpha+1}\) |
    | Fourier | \(\alpha = n\) | \((-\partial_s)^\alpha(1/s) + (\partial_s)^\alpha(1/\bar{s}) = \Gamma(\alpha+1)(1/s^{\alpha+1} + (-1)^\alpha/\bar{s}^{\alpha+1})\) |
    | Mellin | \(\alpha = n+\rho-1\) | \((-\partial_s)^\alpha(1/s) = \Gamma(n+\rho)/s^{n+\rho}\) |
    | Hankel | \(\alpha = n+\nu\) | \((-\partial_s)^\alpha(1/s) = \Gamma(n+\nu+1)/s^{n+\nu+1}\) |
    """)

    st.markdown("**Laplace planting.** This corresponds to an integer planting rank")
    st.latex(r"\alpha = n,")
    st.markdown("acting on the kernel \(1/s\), yielding")
    st.latex(r"(-\partial_s)^\alpha(1/s) = \Gamma(\alpha+1)/s^{\alpha+1}.")

    st.markdown("**Fourier planting.** This appears as the bilateral extension of the Laplace framework, encoded through the conjugate pair \((s,\bar{s})\) with")
    st.latex(r"s = \sigma + i\omega.")
    st.markdown("The planting rank remains integer,")
    st.latex(r"\alpha = n,")
    st.markdown("acting simultaneously on both kernels \((\sigma+i\omega)^{-1}\) and \((\sigma-i\omega)^{-1}\).")
    st.latex(r"(-\partial_s)^\alpha(1/s) + (\partial_s)^\alpha(1/\bar{s}) = \Gamma(\alpha+1)(1/s^{\alpha+1} + (-1)^\alpha/\bar{s}^{\alpha+1})")

    st.markdown("**Mellin planting.** This extends the planting rank to fractional and complex values,")
    st.latex(r"\alpha = n+\rho-1,")
    st.markdown("acting on the kernel \(1/s\), which produces")
    st.latex(r"(-\partial_s)^\alpha(1/s) = \Gamma(n+\rho)/s^{n+\rho}.")

    st.markdown("**Hankel planting.** This introduces a radial shift governed by the Bessel order \(\nu\), with planting rank")
    st.latex(r"\alpha = n+\nu,")
    st.markdown("again acting on the kernel \(1/s\), leading to")
    st.latex(r"(-\partial_s)^\alpha(1/s) = \Gamma(n+\nu+1)/s^{n+\nu+1}.")

# ============================================================
# NEW SECTIONS TO ADD TO EXISTING CODE
# Place these sections after the "Fractional Derivative Link" section
# and before the sidebar navigation
# ============================================================

# ============================================================
# SECTION 8: The Laplace–Gamma Kernel
# ============================================================
def render_laplace_gamma():
    st.header("8. The Laplace–Gamma Kernel")
    
    st.markdown(r"""
    The Laplace–Gamma kernel is defined as
    """)
    st.latex(r"G(s) = s^{-\rho}, \quad \rho > 0.")
    
    st.markdown(r"""
    Hence, by applying the planted differential operator series, we reconstruct the integral form directly from the kernel's planted hierarchy. The reconstruction identity takes the form:
    """)
    st.latex(r"\sum_{n=0}^{\infty} (-1)^n a_n D^n \left( \frac{1}{s^\rho} \right) = \frac{1}{\Gamma(\rho)} \int_0^\infty t^{\rho-1} f(t) e^{-st} dt.")
    
    st.subheader("1. Fundamental Kernel")
    st.markdown(r"""
    The Laplace–Gamma kernel is defined by
    """)
    st.latex(r"G(s) = s^{-\rho}, \quad \rho > 0.")
    
    st.subheader("2. Differential Planting Law")
    st.markdown(r"""
    According to the differential–planting principle, each planted derivative of the kernel satisfies
    """)
    st.latex(r"(-\partial_s)^n G(s) = (\rho)_n s^{-\rho-n},")
    st.markdown(r"""
    where \( (\rho)_n = \rho(\rho+1)\cdots(\rho+n-1) \) is the rising Pochhammer symbol, with \( (\rho)_0 = 1 \). This expresses the recursive structure of the planted derivatives.
    """)
    
    st.subheader("3. Definition on a Taylor Series")
    st.markdown(r"""
    Let \( f(t) \) be analytic near \( t=0 \) with the expansion
    """)
    st.latex(r"f(t) = \sum_{n=0}^{\infty} a_n t^n.")
    st.markdown(r"""
    The Laplace–Gamma type of \( f \) is defined by planting these coefficients onto the kernel's differential hierarchy:
    """)
    st.latex(r"\mathcal{L}_\Gamma\{f\}(s) = \sum_{n=0}^{\infty} a_n (-\partial_s)^n G(s) = \sum_{n=0}^{\infty} a_n (\rho)_n s^{-\rho-n}.")
    st.markdown("This is the foundational series form of the Laplace–Gamma transform.")
    
    st.subheader("4. Conversion to an Integral Form")
    st.markdown(r"""
    Using the Laplace–Gamma identity
    """)
    st.latex(r"s^{-\rho-n} = \frac{1}{\Gamma(\rho+n)} \int_0^\infty e^{-st} t^{\rho+n-1} dt,")
    st.markdown(r"""
    we substitute this representation into the series:
    """)
    st.latex(r"\mathcal{L}_\Gamma\{f\}(s) = \sum_{n=0}^{\infty} a_n (\rho)_n \frac{1}{\Gamma(\rho+n)} \int_0^\infty e^{-st} t^{\rho+n-1} dt.")
    st.latex(r"(\rho)_n = \frac{\Gamma(\rho+n)}{\Gamma(\rho)}")
    st.latex(r"\mathcal{L}_\Gamma\{f\}(s) = \frac{1}{\Gamma(\rho)} \int_0^\infty e^{-st} t^{\rho-1} \sum_{n=0}^{\infty} a_n t^n dt.")
    
    st.subheader("5. Internal Structural Definition")
    st.latex(r"\mathcal{L}_\Gamma\{f\}(s) = \frac{1}{\Gamma(\rho)} \int_0^\infty e^{-st} t^{\rho-1} f(t) dt.")
    st.markdown(r"""
    This is classical Mellin transform of \( \mathcal{M}\{f(t)e^{-st}\}(\rho) \)
    """)
    st.latex(r"\mathcal{M}\{f(t)e^{-st}\}(\rho) = \int_0^\infty f(t) e^{-st} t^{\rho-1} dt,")
    st.latex(r"\mathcal{L}_\Gamma\{f\}(s) = \frac{1}{\Gamma(\rho)} \mathcal{M}\{f(t)e^{-st}\}(\rho)")
    
    st.subheader("Interactive Laplace–Gamma Calculator")
    
    rho_val = st.slider("Select ρ (rho) value", min_value=0.1, max_value=5.0, value=1.0, step=0.1, key="rho_slider")
    s_val = st.slider("Select s value", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="s_slider_gamma")
    
    st.latex(rf"\mathcal{{L}}_\Gamma{{(1)}}({s_val}) = {s_val}^{{-{rho_val}}} = {s_val**(-rho_val):.4f}")
    
    st.markdown("**Example: \( f(t) = e^{-t} \)**")
    st.latex(rf"\mathcal{{L}}_\Gamma{{(e^{{-t}})}}({s_val}) = ({s_val}+1)^{{-{rho_val}}} = {(s_val+1)**(-rho_val):.4f}")

# ============================================================
# SECTION 8.1: Logarithmic Extension of the Generated Kernel
# ============================================================
def render_logarithmic_kernel():
    st.header("8.1 Logarithmic Extension of the Generated Kernel")
    
    st.markdown(r"""
    The logarithmic seed is
    """)
    st.latex(r"G(s) = \ln(s), \quad s > 0.")
    st.markdown(r"""
    For integer \( n \geq 1 \),
    """)
    st.latex(r"\frac{d^n}{ds^n} \ln(s) = (-1)^{n-1} (n-1)! \cdot s^{-n}.")
    st.markdown(r"""
    Hence the planted \( n \)-th derivative of the seed is
    """)
    st.latex(r"(-\partial_s)^n \ln(s) = -(n-1)! \cdot s^{-n} \quad (n \geq 1), \quad (-\partial_s)^0 \ln(s) = \ln(s).")
    
    st.subheader("Logarithmic Extension")
    st.markdown(r"""
    Consider the generated kernel representation
    """)
    st.latex(r"\sum_{n=0}^{\infty} a_n (-1)^n D^n \left( \frac{1}{s} \right) = \mathcal{L}\{f(s)\}.")
    st.markdown(r"""
    Replacing the kernel \( \frac{1}{s} \to \ln s \), we obtain the logarithmic extension
    """)
    st.latex(r"a_0 \ln s + \sum_{n=1}^{\infty} a_n (-1)^{n-1} D^n (\ln s).")
    st.markdown(r"""
    Using
    """)
    st.latex(r"D^n (\ln s) = (-1)^{n-1} \frac{(n-1)!}{s^n},")
    st.markdown(r"""
    we get
    """)
    st.latex(r"(-1)^{n-1} D^n (\ln s) = \frac{(n-1)!}{s^n}.")
    st.markdown(r"""
    Therefore,
    """)
    st.latex(r"a_0 \ln s + \sum_{n=1}^{\infty} a_n \frac{(n-1)!}{s^n}.")
    st.markdown(r"""
    Since
    """)
    st.latex(r"\mathcal{L}\{t^{n-1}\} = \frac{(n-1)!}{s^n},")
    st.markdown(r"""
    it follows that
    """)
    st.latex(r"a_0 \ln s + \sum_{n=1}^{\infty} a_n \mathcal{L}\{t^{n-1}\} = a_0 \ln s + \mathcal{L}\left\{ \frac{f(t) - a_0}{t} \right\}.")
    st.markdown(r"""
    Hence, the replacement of the kernel \( 1/s \) by the logarithmic kernel \( \ln s \) transforms the generated representation into a Laplace integral containing the factor \( 1/t \).
    """)

# ============================================================
# SECTION 8.2: Example: Logarithmic Kernel for sin t
# ============================================================
def render_logarithmic_example():
    st.header("8.2 Example: Logarithmic Kernel Representation for \(\sin t\)")
    
    st.markdown(r"""
    Consider the function
    """)
    st.latex(r"f(t) = \sin t.")
    st.markdown(r"""
    Using its Maclaurin expansion,
    """)
    st.latex(r"\sin t = \sum_{k=0}^{\infty} \frac{(-1)^k t^{2k+1}}{(2k+1)!},")
    st.markdown(r"""
    the corresponding coefficients \( a_n \) are substituted into the logarithmic kernel representation.
    """)
    
    st.latex(r"\sum_{k=0}^{\infty} \frac{(-1)^k}{(2k+1)!} \frac{(2k)!}{s^{2k+1}} = \sum_{k=0}^{\infty} \frac{(-1)^k}{(2k+1)} \frac{1}{s^{2k+1}}.")
    
    st.markdown(r"""
    Using the Taylor expansion of the inverse tangent function,
    """)
    st.latex(r"\tan^{-1}(z) = \sum_{k=0}^{\infty} \frac{(-1)^k z^{2k+1}}{2k+1},")
    st.markdown(r"""
    with \( z = \frac{1}{s} \), the generated series becomes
    """)
    st.latex(r"\tan^{-1}\left(\frac{1}{s}\right).")
    
    st.markdown(r"""
    On the other hand, the logarithmic kernel extension gives
    """)
    st.latex(r"\mathcal{L}\left\{ \frac{f(t)}{t} \right\} = \mathcal{L}\left\{ \frac{\sin t}{t} \right\},")
    st.markdown(r"""
    therefore,
    """)
    st.latex(r"\int_{0}^{\infty} \frac{\sin t}{t} e^{-st} dt = \tan^{-1}\left(\frac{1}{s}\right).")
    
    st.markdown(r"""
    This example demonstrates that replacing the kernel \( \frac{1}{s} \) by \( \ln s \) transforms the generated function \( f(t) \) into the normalized kernel \( \frac{f(t)}{t} \).
    """)

# ============================================================
# SECTION 9.1: Inverse-Kernel Duality Framework
# ============================================================
def render_inverse_kernel_duality():
    st.header("9.1 The Inverse-Kernel Duality Framework")
    
    st.markdown(r"""
    The core advancement of this generalized framework is the establishment of the **Inverse-Kernel Duality**. Instead of relying on a fixed foundational anchor, we allow the planting mechanism to act upon a dynamically generated kernel \( G(s) \), which encapsulates its own internal analytical structure.
    """)
    
    st.subheader("Theorem 9.1 (Inverse-Kernel Duality)")
    st.markdown(r"""
    Let \( G(s) \) be an operator-type transform generated by a continuous spatial weight function \( g(t) \) via the classical Laplace integral:
    """)
    st.latex(r"G(s) = \mathcal{L}\{g(t)\}(s) = \int_0^\infty e^{-st} g(t) dt \tag{191}")
    
    st.markdown(r"""
    Let \( f(t) = \sum_{n=0}^{\infty} a_n t^n \) be an analytic function in a neighborhood of \( t = 0 \) with Maclaurin coefficients \( a_n \). If the transform operator \( \mathcal{T} \) is defined purely through the differential planting series:
    """)
    st.latex(r"\mathcal{T}\{f\}(s) := \sum_{n=0}^{\infty} a_n (-D_s)^n G(s) \tag{192}")
    st.markdown(r"""
    where \( D_s = \frac{d}{ds} \), then \( \mathcal{T}\{f\}(s) \) admits the equivalent weighted integral representation:
    """)
    st.latex(r"\mathcal{T}\{f\}(s) = \int_0^\infty e^{-st} f(t) g(t) dt \tag{193}")
    st.markdown(r"""
    where \( g(x) = \mathcal{L}^{-1}\{G(s)\}(t) \) is the structural inverse kernel of the transform.
    """)
    
    st.subheader("Proof")
    st.markdown(r"""
    Consider the \( n \)-th order derivative of the generated kernel \( G(s) \):
    """)
    st.latex(r"G(s) = \int_0^\infty e^{-st} g(t) dt")
    st.latex(r"D_s^n G(s) = \int_0^\infty (-1)^n t^n e^{-st} g(t) dt \tag{194}")
    st.markdown(r"""
    To ensure sign absorption and clean algebraic mapping, we isolate the planted differential rank:
    """)
    st.latex(r"(-D_s)^n G(s) = \int_0^\infty t^n e^{-st} g(t) dt \tag{195}")
    
    st.markdown(r"""
    Multiplying both sides by \( a_n \) and summing:
    """)
    st.latex(r"\sum_{n=0}^{\infty} a_n (-D_s)^n G(s) = \sum_{n=0}^{\infty} a_n \int_0^\infty t^n e^{-st} g(t) dt \tag{196}")
    
    st.markdown(r"""
    Assuming uniform convergence, we interchange summation and integration:
    """)
    st.latex(r"\mathcal{T}\{f\}(s) = \int_0^\infty e^{-st} g(t) \left( \sum_{n=0}^{\infty} a_n t^n \right) dt \tag{197}")
    st.markdown(r"""
    Recognizing the internal series as \( f(t) \):
    """)
    st.latex(r"\mathcal{T}\{f\}(s) = \int_0^\infty e^{-st} f(t) g(t) dt \tag{198}")
    st.markdown("This completes the proof.")
    
    st.subheader("Examples")
    st.markdown(r"""
    **Example 9.2.** Let \( G(s) = \frac{s}{s^2 + b^2} \), \( g(t) = \cos(bt) \). Then
    """)
    st.latex(r"\sum_{n=0}^{\infty} (-1)^n a_n D^n \left( \frac{s}{s^2 + b^2} \right) = \int_0^\infty f(t) e^{-st} \cos(bt) dt")
    
    st.markdown(r"""
    **Example 9.3.** Let \( G(s) = \frac{b}{s^2 + b^2} \), \( g(t) = \sin(bt) \). Then
    """)
    st.latex(r"\sum_{n=0}^{\infty} (-1)^n a_n D^n \left( \frac{b}{s^2 + b^2} \right) = \int_0^\infty f(t) e^{-st} \sin(bt) dt")

# ============================================================
# SECTION 9.2: Using Inverse-Kernel Duality (sin and cos transforms)
# ============================================================
def render_inverse_kernel_transforms():
    st.header("9.2 Using the Inverse-Kernel Duality and Proving the Transform Results")
    
    st.markdown(r"""
    ### Sine Transform
    """)
    st.latex(r"\mathcal{L}\{t^n \sin(bt)\}(s) = \frac{n!}{r^{n+1}} \sin((n+1)\theta)")
    st.markdown(r"""
    where
    """)
    st.latex(r"r = \sqrt{s^2 + b^2}, \quad \theta = \arctan\left(\frac{b}{s}\right).")
    
    st.markdown(r"""
    ### Classical Mellin for Sine
    """)
    st.latex(r"\int_0^\infty t^{\rho-1} \sin t \, dt = \Gamma(\rho) \sin\left(\frac{\pi \rho}{2}\right), \quad 0 < \Re(\rho) < 1.")
    
    st.markdown(r"""
    ### Cosine Transform
    """)
    st.latex(r"\mathcal{L}\{t^n \cos(bt)\}(s) = \frac{n!}{r^{n+1}} \cos((n+1)\theta)")
    
    st.markdown(r"""
    ### Classical Mellin for Cosine
    """)
    st.latex(r"\int_0^\infty t^{\rho-1} \cos t \, dt = \Gamma(\rho) \cos\left(\frac{\pi \rho}{2}\right), \quad 0 < \Re(\rho) < 1.")
    
    st.subheader("Derivation for Sine Kernel")
    st.markdown(r"""
    Starting from the formula
    """)
    st.latex(r"\mathcal{T}\{f(t)\}(s) = \sum_{n=0}^{\infty} (-1)^n a_n D^n \left( \frac{1}{s^2 + 1} \right) = \int_0^\infty e^{-st} f(t) \sin t \, dt.")
    st.markdown(r"""
    Using the algebraic decomposition
    """)
    st.latex(r"\frac{1}{s^2 + 1} = \frac{1}{2i} \left( \frac{1}{s - i} - \frac{1}{s + i} \right),")
    st.markdown(r"""
    and the polar representation \( s \pm i = re^{\pm i\theta} \), we obtain
    """)
    st.latex(r"D^n G(s) = (-1)^n n! \frac{\sin((n+1)\theta)}{r^{n+1}}.")
    st.markdown(r"""
    For \( f(t) = t^n \), this yields
    """)
    st.latex(r"\frac{n! \sin((n+1)\theta)}{r^{n+1}} = \int_0^\infty e^{-st} t^n \sin t \, dt = \mathcal{L}\{t^n \sin(bt)\}(s).")
    
    st.subheader("Derivation for Cosine Kernel")
    st.markdown(r"""
    Starting from
    """)
    st.latex(r"\mathcal{T}\{f(t)\}(s) = \sum_{n=0}^{\infty} (-1)^n a_n D^n \left( \frac{s}{s^2 + 1} \right) = \int_0^\infty e^{-st} f(t) \cos t \, dt.")
    st.markdown(r"""
    Using the decomposition
    """)
    st.latex(r"\frac{s}{s^2 + 1} = \frac{1}{2} \left( \frac{1}{s - i} + \frac{1}{s + i} \right),")
    st.markdown(r"""
    we obtain
    """)
    st.latex(r"D^n G(s) = (-1)^n n! \frac{\cos((n+1)\theta)}{r^{n+1}}.")
    st.markdown(r"""
    For \( f(t) = t^n \):
    """)
    st.latex(r"\frac{n! \cos((n+1)\theta)}{r^{n+1}} = \int_0^\infty e^{-st} t^n \cos t \, dt = \mathcal{L}\{t^n \cos(bt)\}(s).")

# ============================================================
# SECTION 9.4: Kernel Multiplication and Boundary Contribution
# ============================================================
def render_kernel_multiplication():
    st.header("9.4 Kernel Multiplication and Boundary Contribution")
    
    st.markdown(r"""
    **Lemma 9.4** (Kernel Multiplication and Boundary Contribution). Let
    """)
    st.latex(r"G(s) = \int_0^\infty e^{-st} g(t) dt \tag{203}")
    st.markdown(r"""
    be an inverse-kernel representation. Then multiplication of the kernel by \( s \) satisfies:
    """)
    st.latex(r"sG(s) = g(0) + \int_0^\infty e^{-st} g'(t) dt, \tag{204}")
    st.markdown(r"""
    and hence, in the sense of inverse kernels,
    """)
    st.latex(r"\mathcal{T}^{-1}\{sG(s)\}(t) = g'(t) + g(0)\delta(t). \tag{205}")
    
    st.markdown(r"""
    In particular, the boundary contribution depends exclusively on the value of the original inverse kernel \( g(t) \) at the origin.
    
    At \( g(0) = 0 \):
    """)
    st.latex(r"sG(s) = \int_0^\infty e^{-st} g'(t) dt, \tag{206}")
    st.latex(r"s^2G(s) = \int_0^\infty e^{-st} g''(t) dt, \tag{207}")
    st.latex(r"s^nG(s) = \int_0^\infty e^{-st} g^{(n)}(t) dt, \tag{208}")
    st.latex(r"\mathcal{T}^{-1}\{s^nG(s)\}(t) = g^{(n)}(t). \tag{209}")
    
    st.markdown(r"""
    **Conclusion.** Multiplication by \( s \) injects differentiation into the inverse kernel, while boundary contributions arise if and only if the original inverse kernel is nonzero at the origin.
    """)

# ============================================================
# SECTION 9.5: Kernel Division and Integral Injection
# ============================================================
def render_kernel_division():
    st.header("9.5 Kernel Division and Integral Injection")
    
    st.markdown(r"""
    **Lemma 9.5** (Kernel Division and Integral Injection). Let
    """)
    st.latex(r"G(s) = \int_0^\infty e^{-st} g(t) dt \tag{210}")
    st.markdown(r"""
    be an inverse-kernel representation. Define:
    """)
    st.latex(r"H(s) := \frac{G(s)}{s}, \quad \Re(s) > 0. \tag{211}")
    st.markdown(r"""
    Then \( H \) admits the inverse-kernel representation:
    """)
    st.latex(r"H(s) = \int_0^\infty e^{-st} h(t) dt, \quad \text{where } h(t) = \int_0^t g(u) du. \tag{212}")
    st.markdown(r"""
    Equivalently,
    """)
    st.latex(r"\mathcal{T}^{-1}\left\{ \frac{G(s)}{s} \right\} (t) = \int_0^t g(u) du. \tag{213}")
    
    st.markdown(r"""
    More generally, for any integer \( k \geq 1 \):
    """)
    st.latex(r"\mathcal{T}^{-1}\left\{ \frac{G(s)}{s^k} \right\} (t) = \int_0^t \int_0^{u_1} \cdots \int_0^{u_{k-1}} g(u_k) du_k \cdots du_1, \tag{214}")
    st.markdown(r"""
    i.e., division by \( s^k \) injects \( k \)-fold integration into the inverse kernel.
    """)
    
    st.markdown(r"""
    **Remark.** Unlike multiplication by \( s \), division by \( s \) produces no boundary term: there is no integration by parts, hence no evaluation at \( t = 0 \) or \( t = \infty \).
    """)

# ============================================================
# SECTION 9.6: Cauchy Reduction Principle
# ============================================================
def render_cauchy_reduction():
    st.header("9.6 The Cauchy Reduction Principle and Fractional Integral Injection")
    
    st.markdown(r"""
    **Theorem 9.6** (Cauchy Reduction of the Operator Kernel). Let \( k \in \mathbb{N} \) and let \( g(t) \) be a locally integrable function on \( [0, \infty) \). The \( k \)-fold repeated integral mapping of the inverse kernel can be rigorously reduced to a single continuous integral weighted by a polynomial kernel:
    """)
    st.latex(r"\int_0^t \int_0^{u_1} \cdots \int_0^{u_{k-1}} g(u_k) du_k \cdots du_1 = \frac{1}{(k-1)!} \int_0^t (t-\tau)^{k-1} g(\tau) d\tau")
    
    st.markdown(r"""
    Recognizing that \( (m-1)! \cdot m = m! \), the expression simplifies to:
    """)
    st.latex(r"I^{m+1}[g](t) = \frac{1}{m!} \int_0^t (t-\tau)^m g(\tau) d\tau")
    
    st.markdown(r"""
    This confirms that the relation holds true for \( k = m+1 \), completing the rigorous inductive proof.
    """)

# ============================================================
# SECTION 9.7: Fractional Integral Injection
# ============================================================
def render_fractional_integral():
    st.header("9.7 Generalization to Continuous Fractional Calculus Domains")
    
    st.markdown(r"""
    **Definition 9.7** (Fractional Integral Injection Operator). Let \( \alpha \in \mathbb{R}^+ \) be an arbitrary positive fractional rank. The division of a generalized generated kernel \( G(s) \) by the continuous fractional power \( s^\alpha \) induces a continuous convolution weight into the inverse-kernel spatial domain:
    """)
    st.latex(r"\mathcal{T}^{-1}\left\{ \frac{G(s)}{s^\alpha} \right\} (t) = \frac{1}{\Gamma(\alpha)} \int_0^t (t-\tau)^{\alpha-1} g(\tau) d\tau")
    st.markdown(r"""
    where \( \Gamma(\alpha) = \int_0^\infty t^{\alpha-1} e^{-t} dt \).
    """)
    
    st.latex(r"\mathcal{T}\{f\}(s) := \sum_{n=0}^{\infty} a_n (-D_s)^n \frac{G(s)}{s^\alpha} = \int_0^\infty e^{-st} f(t) \frac{1}{\Gamma(\alpha)} \int_0^t (t-\tau)^{\alpha-1} g(\tau) d\tau dt")
    
    st.markdown(r"""
    If the framework is anchored to the foundational baseline where \( G(s) = 1/s \) (and consequently \( g(t) = 1 \)), the expression yields:
    """)
    st.latex(r"\mathcal{T}^{-1}\left\{ \frac{1}{s^{\alpha+1}} \right\} (t) = \frac{1}{\Gamma(\alpha)} \int_0^t (t-\tau)^{\alpha-1} (1) d\tau = \frac{t^\alpha}{\Gamma(\alpha+1)}")
    
    st.markdown(r"""
    This perfectly aligns with classical fractional calculus definitions, demonstrating that our generalized core embeds standard fractional integration as a simple boundary restriction.
    """)

# ============================================================
# SECTION 9.8: General Rational Operator and Convolution
# ============================================================
def render_rational_operator():
    st.header("9.8 The General Rational Operator and Convolution Domain Integration")
    
    st.markdown(r"""
    **Theorem 9.8** (The Generalized Convolution Kernel Identity). Let \( G(s) = \int_0^\infty e^{-st} g(t) dt \) be a continuous inverse-kernel representation. Let \( P(s) \) be an algebraic polynomial in the transform parameter \( s \) such that its reciprocal admits a well-defined spatial weight function \( h(t) = \mathcal{L}^{-1}\left\{ \frac{1}{P(s)} \right\}(t) \). Define the compound rational transform operator \( W(s) \) as:
    """)
    st.latex(r"W(s) := \frac{G(s)}{P(s)} \tag{224}")
    st.markdown(r"""
    Then the structural inversion of \( W(s) \) maps directly onto a closed spatial convolution integral:
    """)
    st.latex(r"\mathcal{T}^{-1}\left\{ \frac{G(s)}{P(s)} \right\} (t) = \int_0^t h(t-\tau) g(\tau) d\tau \tag{225}")
    
    st.markdown(r"""
    **Corollary 9.9** (Compound Rational Operator Law). Let \( n \in \mathbb{N}_0 \) and let \( P(s) \) be a transform domain polynomial operator generating the inverse mapping response \( h(t) \):
    """)
    st.latex(r"\mathcal{T}^{-1}\left\{ \frac{G(s)}{s^n P(s)} \right\} (t) = \frac{1}{(n-1)!} \int_0^t (t-\tau)^{n-1} \left( \int_0^\tau h(\tau-z) g(z) dz \right) d\tau \tag{233}")
    
    st.markdown(r"""
    Alternatively, invoking the associativity and commutativity properties of spatial convolution:
    """)
    st.latex(r"\mathcal{T}^{-1}\left\{ \frac{G(s)}{s^n P(s)} \right\} (t) = \int_0^t K_{n,P}(t-\tau) g(\tau) d\tau \tag{234}")
    st.markdown(r"""
    where \( K_{n,P}(t) = \mathcal{L}^{-1}\left\{ \frac{1}{s^n P(s)} \right\}(t) \) represents the total compound baseline response function.
    """)

# ============================================================
# SECTION 9.9: Grand Unified Framework Mapping Table (Table 5)
# ============================================================
def render_grand_table():
    st.header("9.9 The Grand Unified Framework Mapping Table")
    
    st.markdown(r"""
    The following mapping table summarizes the structural transformations induced by all evaluated operator configurations in the \( s \)-domain alongside their precise spatial representations in the \( t \)-domain.
    """)
    
    st.markdown("### Table 5: Comprehensive Operational Mapping of the Generalized Kernel Framework")
    
    st.markdown("""
    | s-Domain Operator | t-Domain Spatial Representation | Analytical Classification |
    |-------------------|--------------------------------|---------------------------|
    | \( \mathcal{T}^{-1}\{G(s)\} \) | \( g(t) \) | Foundational Kernel |
    | \( \mathcal{T}^{-1}\left\{\frac{G(s)}{s}\right\} \) | \( \int_0^t g(u) du \) | Single Integral Injection |
    | \( \mathcal{T}^{-1}\left\{\frac{G(s)}{s^n}\right\} \) | \( \frac{1}{(n-1)!} \int_0^t (t-\tau)^{n-1} g(\tau) d\tau \) | Repeated Integral Injection |
    | \( \mathcal{T}^{-1}\{G(s-a)\} \) | \( e^{at} g(t) \) | Exponential Shift |
    | \( \mathcal{T}^{-1}\{1\} \) | \( \delta(t) \) | Dirac Delta |
    | \( \mathcal{T}^{-1}\left\{\frac{1}{s-a}\right\} \) | \( e^{at} \) | Pure Exponential |
    | \( \mathcal{T}^{-1}\left\{\frac{1}{(s-a)^n}\right\} \) | \( \frac{t^{n-1}}{(n-1)!} e^{at} \) | Higher-Order Shifted Monomial |
    | \( \mathcal{T}^{-1}\left\{\frac{1}{s(s-a)^n}\right\} \) | \( \frac{1}{(n-1)!} \int_0^t u^{n-1} e^{au} du \) | Neutral Base Shifted Monomial |
    | \( \mathcal{T}^{-1}\left\{\frac{G(s)}{s^2+a^2}\right\} \) | \( \frac{1}{a} \int_0^t \sin(a(t-\tau)) g(\tau) d\tau \) | Trigonometric Modulated Response |
    | \( \mathcal{T}^{-1}\left\{\frac{G(s)}{s^2-a^2}\right\} \) | \( \frac{1}{a} \int_0^t \sinh(a(t-\tau)) g(\tau) d\tau \) | Hyperbolic Sinh Modulated Kernel |
    | \( \mathcal{T}^{-1}\left\{\frac{sG(s)}{s^2+a^2}\right\} \) | \( \int_0^t \cos(a(t-\tau)) g(\tau) d\tau \) | Trigonometric Modulated Response |
    | \( \mathcal{T}^{-1}\left\{\frac{sG(s)}{s^2-a^2}\right\} \) | \( \int_0^t \cosh(a(t-\tau)) g(\tau) d\tau \) | Hyperbolic Cosh Operator Layer |
    | \( \mathcal{T}^{-1}\left\{\frac{G(s)}{s^n(s+a)}\right\} \) | \( \frac{1}{(n-1)!} \int_0^t (t-\tau)^{n-1} e^{-a(t-\tau)} g(\tau) d\tau \) | Exponential-Monomial Composite |
    | \( \mathcal{T}^{-1}\left\{\frac{G(s)}{s^n(s^2+a^2)}\right\} \) | \( \frac{1}{a(n-1)!} \int_0^t (t-\tau)^{n-1} \sin(a(t-\tau)) g(\tau) d\tau \) | Trigonometric-Monomial Extension |
    | \( \mathcal{T}^{-1}\left\{\frac{G(s)}{P(s)}\right\} \) | \( \int_0^t h(t-\tau) g(\tau) d\tau \) | General Dual-Kernel Convolution |
    | \( \mathcal{T}^{-1}\left\{\frac{G(s)}{s^n P(s)}\right\} \) | \( \int_0^t K_{n,P}(t-\tau) g(\tau) d\tau \) | Higher-Order Polynomial Regularization |
    | \( \mathcal{T}^{-1}\left\{\frac{G(s)}{s^\alpha}\right\} \) | \( \frac{1}{\Gamma(\alpha)} \int_0^t (t-\tau)^{\alpha-1} g(\tau) d\tau \) | Continuous Fractional Domain |
    """)

# ============================================================
# SECTION 9.10: Kernel-Based Coefficient Extraction (Partial Fractions)
# ============================================================
def render_coefficient_extraction():
    st.header("9.10 Kernel-Based Coefficient Extraction Without Partial Fractions")
    
    st.markdown(r"""
    In classical inverse Laplace problems, rational functions are usually simplified using partial fraction decomposition (PFD). This algebraic approach requires finding unknown constants through tedious simultaneous equations and polynomial factorization.
    
    The General Kernel framework provides an entirely alternative viewpoint: instead of decomposing the transform expression first, the required decomposition components emerge naturally from the structural properties of the generated spatial kernel.
    """)
    
    st.subheader("Example 1: \( F(s) = \frac{1}{s(s+1)^2} \)")
    st.markdown(r"""
    The baseline factor \( \frac{1}{s} \) corresponds to \( g(t) = 1 \), while the shifting operator \( P(s) = s+1 \) generates \( K_P(t) = e^{-t} \).
    """)
    st.latex(r"f(t) = \mathcal{L}^{-1}\left\{ \frac{1}{s(s+1)^2} \right\} = \int_0^t (t-\tau)e^{-(t-\tau)}(1) d\tau")
    st.latex(r"f(t) = 1 - (t+1)e^{-t}")
    st.markdown(r"""
    Forward transform:
    """)
    st.latex(r"\frac{1}{s(s+1)^2} = \frac{1}{s} - \frac{1}{s+1} - \frac{1}{(s+1)^2}")
    st.markdown("Coefficients: \( A = 1, B = -1, C = -1 \)")
    
    st.subheader("Example 2: \( F(s) = \frac{1}{(s-2)(s-3)} \)")
    st.markdown(r"""
    Using dual-kernel convolution:
    """)
    st.latex(r"f(t) = \int_0^t e^{3(t-\tau)} e^{2\tau} d\tau = e^{3t} - e^{2t}")
    st.latex(r"\frac{1}{(s-2)(s-3)} = \frac{1}{s-3} - \frac{1}{s-2}")
    st.markdown("Coefficients: \( A = -1, B = 1 \)")
    
    st.subheader("Example 3: \( H(s) = \frac{1}{s^2(s^2+1)} \)")
    st.markdown(r"""
    Using trigonometric-monomial structure:
    """)
    st.latex(r"h(t) = \int_0^t (t-\tau)\sin(t-\tau) d\tau = \sin(t) - t\cos(t)")
    st.latex(r"\frac{1}{s^2(s^2+1)} = \frac{1}{s^2} - \frac{1}{s^2+1}")
    
    st.subheader("Example 4: \( F(s) = \frac{1}{s^2(s-1)(s^2+1)} \)")
    st.markdown(r"""
    Base Kernel Weight: \( g(t) = t \)
    System Response: \( h(t) = \frac{1}{2}(e^t - \sin t - \cos t) \)
    """)
    st.latex(r"f(t) = \frac{1}{2}e^t - t + \frac{1}{2}\sin t - \frac{1}{2}\cos t")
    st.latex(r"\frac{1}{s^2(s-1)(s^2+1)} = \frac{1}{2(s-1)} - \frac{1}{s^2} + \frac{1}{2(s^2+1)} - \frac{s}{2(s^2+1)}")
    
    st.subheader("Example 5: \( F(s) = \frac{1}{(s-1)(s-3)(s-4)} \)")
    st.markdown(r"""
    Base Kernel: \( g(t) = e^t \)
    System Response: \( h(t) = e^{4t} - e^{3t} \)
    """)
    st.latex(r"f(t) = \frac{1}{6}e^t - \frac{1}{2}e^{3t} + \frac{1}{3}e^{4t}")
    st.latex(r"\frac{1}{(s-1)(s-3)(s-4)} = \frac{1}{6(s-1)} - \frac{1}{2(s-3)} + \frac{1}{3(s-4)}")
    st.markdown("Coefficients: \( A = \frac{1}{6}, B = -\frac{1}{2}, C = \frac{1}{3} \)")
    
    st.subheader("Example 6: \( F(s) = \frac{1}{s(s+1)^5} \)")
    st.markdown(r"""
    Base Kernel: \( g(t) = 1 \)
    System Response: \( h(t) = \frac{t^4}{24}e^{-t} \)
    """)
    st.latex(r"f(t) = 1 - e^{-t}\left(1 + t + \frac{t^2}{2} + \frac{t^3}{6} + \frac{t^4}{24}\right)")
    st.latex(r"\frac{1}{s(s+1)^5} = \frac{1}{s} - \frac{1}{s+1} - \frac{1}{(s+1)^2} - \frac{1}{(s+1)^3} - \frac{1}{(s+1)^4} - \frac{1}{(s+1)^5}")
    st.markdown("Coefficients: \( A = 1, B = -1, C = -1, D = -1, E = -1, F = -1 \)")

# ============================================================
# Add these to the sidebar navigation
# ============================================================
# In the sidebar radio, add these options:
# "Laplace–Gamma Kernel",
# "Logarithmic Kernel",
# "Inverse-Kernel Duality",
# "Kernel Multiplication & Division",
# "Cauchy Reduction & Fractional Integral",
# "Grand Unified Mapping Table",
# "Coefficient Extraction (Partial Fractions)"

# ============================================================
# Add these to the section rendering
# ============================================================
# elif section == "Laplace–Gamma Kernel":
#     render_laplace_gamma()
#     render_logarithmic_kernel()
#     render_logarithmic_example()
# elif section == "Inverse-Kernel Duality":
#     render_inverse_kernel_duality()
#     render_inverse_kernel_transforms()
# elif section == "Kernel Multiplication & Division":
#     render_kernel_multiplication()
#     render_kernel_division()
# elif section == "Cauchy Reduction & Fractional Integral":
#     render_cauchy_reduction()
#     render_fractional_integral()
#     render_rational_operator()
# elif section == "Grand Unified Mapping Table":
#     render_grand_table()
# elif section == "Coefficient Extraction (Partial Fractions)":
#     render_coefficient_extraction()
