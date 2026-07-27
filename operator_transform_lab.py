# ============================================================
# app.py
# Operator Transform Laboratory - Complete Version
# Laplace (x→t) + Fourier (original) + Mellin (new from screenshots)
# ============================================================

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
nu = sp.Symbol("nu", real=True)
m = sp.Symbol("m", integer=True, nonnegative=True)

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
    t_local = sp.Symbol("t", positive=True, real=True)
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
            "plant_steps": [
                r"MT\{e^{-t}\}(\rho,s)=\sum_{n=0}^{\infty}\frac{(-1)^n}{n!}\frac{\Gamma(\rho+n)}{s^{\rho+n}},\qquad s>0",
                r"\Gamma(\rho+n)=\Gamma(\rho)(\rho)_n,\qquad (\rho)_n=\frac{\Gamma(\rho+n)}{\Gamma(\rho)}",
                r"MT\{e^{-t}\}(\rho,s)=\Gamma(\rho)s^{-\rho}\sum_{n=0}^{\infty}\frac{(\rho)_n}{n!}\left(-\frac{1}{s}\right)^n",
                r"(1+z)^{-\rho}=\sum_{n=0}^{\infty}\frac{(\rho)_n}{n!}(-z)^n",
                r"\sum_{n=0}^{\infty}\frac{(\rho)_n}{n!}\left(-\frac{1}{s}\right)^n=\left(1+\frac{1}{s}\right)^{-\rho}",
                r"MT\{e^{-t}\}(\rho,s)=\Gamma(\rho)s^{-\rho}\left(1+\frac{1}{s}\right)^{-\rho}",
                r"=\frac{\Gamma(\rho)}{(s+1)^{\rho}}",
            ],
            "closed_text": r"MT\{e^{-t}\}(\rho,s)=\frac{\Gamma(\rho)}{(s+1)^{\rho}}",
            "classical_text": r"\lim_{s\to0^+}MT\{e^{-t}\}(\rho,s)=\Gamma(\rho)",
            "extra_note": r"\int_0^\infty e^{-t}t^{\rho-1}\,dt=\Gamma(\rho)"
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
            "plant_steps": [
                r"MT\{e^{-at}\}(\rho,s)=\sum_{n=0}^{\infty}\frac{(-a)^n}{n!}\frac{\Gamma(\rho+n)}{s^{\rho+n}},\qquad s>0",
                r"\Gamma(\rho+n)=\Gamma(\rho)(\rho)_n",
                r"MT\{e^{-at}\}(\rho,s)=\Gamma(\rho)s^{-\rho}\sum_{n=0}^{\infty}\frac{(\rho)_n}{n!}\left(-\frac{a}{s}\right)^n",
                r"(1+z)^{-\rho}=\sum_{n=0}^{\infty}\frac{(\rho)_n}{n!}(-z)^n",
                r"\sum_{n=0}^{\infty}\frac{(\rho)_n}{n!}\left(-\frac{a}{s}\right)^n=\left(1+\frac{a}{s}\right)^{-\rho}",
                r"MT\{e^{-at}\}(\rho,s)=\Gamma(\rho)s^{-\rho}\left(1+\frac{a}{s}\right)^{-\rho}",
                r"=\frac{\Gamma(\rho)}{(s+a)^{\rho}}",
            ],
            "closed_text": r"MT\{e^{-at}\}(\rho,s)=\frac{\Gamma(\rho)}{(s+a)^{\rho}}",
            "classical_text": r"\lim_{s\to0^+}MT\{e^{-at}\}(\rho,s)=a^{-\rho}\Gamma(\rho)",
            "extra_note": r"\int_0^\infty e^{-at}t^{\rho-1}\,dt=a^{-\rho}\Gamma(\rho)"
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
            "plant_steps": [
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
            ],
            "closed_text": r"MT\{e^{it}\}(\rho,s)=\frac{\Gamma(\rho)}{(s-i)^{\rho}}",
            "classical_text": r"\lim_{s\to0^+}MT\{e^{it}\}(\rho,s)=\Gamma(\rho)e^{i\pi\rho/2}",
            "extra_note": r"\int_0^\infty t^{\rho-1}e^{-(s-i)t}\,dt=\frac{\Gamma(\rho)}{(s-i)^{\rho}}"
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
            "plant_steps": [
                r"MT\{\cos t\}(\rho,s)=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n)!}\frac{\Gamma(\rho+2n)}{s^{\rho+2n}}",
                r"\Gamma(\rho+2n)=\Gamma(\rho)2^{2n}\left(\frac{\rho}{2}\right)_n\left(\frac{\rho+1}{2}\right)_n",
                r"(2n)!=2^{2n}\left(\frac12\right)_n n!",
                r"MT\{\cos t\}(\rho,s)=\Gamma(\rho)s^{-\rho}\,{}_2F_1\!\left(\frac{\rho}{2},\frac{\rho+1}{2};\frac12;-\frac{1}{s^2}\right)",
                r"\text{Equivalently, }MT\{\cos t\}(\rho,s)=\Gamma(\rho)(s^2+1)^{-\rho/2}\cos\!\left(\rho\arctan\!\frac{1}{s}\right)",
            ],
            "closed_text": r"MT\{\cos t\}(\rho,s)=\Gamma(\rho)(s^2+1)^{-\rho/2}\cos\!\left(\rho\arctan\!\frac{1}{s}\right)",
            "classical_text": r"\lim_{s\to0^+}MT\{\cos t\}(\rho,s)=\Gamma(\rho)\cos\!\left(\frac{\pi\rho}{2}\right),\qquad 0<\rho<1",
            "extra_note": r"\int_0^\infty t^{\rho-1}\cos t\,dt=\Gamma(\rho)\cos\!\left(\frac{\pi\rho}{2}\right),\qquad 0<\rho<1"
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
            "plant_steps": [
                r"MT\{\sin t\}(\rho,s)=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n+1)!}\frac{\Gamma(\rho+2n+1)}{s^{\rho+2n+1}}",
                r"\Gamma(\rho+2n+1)=\Gamma(\rho+1)2^{2n}\left(\frac{\rho+1}{2}\right)_n\left(\frac{\rho+2}{2}\right)_n",
                r"(2n+1)!=2^{2n}\left(\frac32\right)_n n!",
                r"MT\{\sin t\}(\rho,s)=\Gamma(\rho+1)s^{-(\rho+1)}\,{}_2F_1\!\left(\frac{\rho+1}{2},\frac{\rho+2}{2};\frac32;-\frac{1}{s^2}\right)",
                r"\text{Equivalently, }MT\{\sin t\}(\rho,s)=\Gamma(\rho)(s^2+1)^{-\rho/2}\sin\!\left(\rho\arctan\!\frac{1}{s}\right)",
            ],
            "closed_text": r"MT\{\sin t\}(\rho,s)=\Gamma(\rho)(s^2+1)^{-\rho/2}\sin\!\left(\rho\arctan\!\frac{1}{s}\right)",
            "classical_text": r"\lim_{s\to0^+}MT\{\sin t\}(\rho,s)=\Gamma(\rho)\sin\!\left(\frac{\pi\rho}{2}\right),\qquad 0<\rho<1",
            "extra_note": r"\int_0^\infty t^{\rho-1}\sin t\,dt=\Gamma(\rho)\sin\!\left(\frac{\pi\rho}{2}\right),\qquad 0<\rho<1"
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
    ],
)


# ============================================================
# Header
# ============================================================
st.title("Operator Transform Laboratory")
st.subheader("Kernel Geometry and Planted Differential Structures")
st.caption("Interactive version for Laplace, Fourier, and Mellin transforms")
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
        key="laplace_case",
    )
    render_case(cases[selected_case])

    st.divider()


# ============================================================
# SECTION: Fourier Transform
# ============================================================
elif section == "Operator Fourier Transform":
    st.header("3. Operator Fourier Transform")

    st.markdown(r"""
    In this section we extend the operator-based Laplace framework to the oscillatory Fourier domain. The key structural point is that the bilateral Fourier transform naturally produces a symmetric pair of complex Laplace kernels rather than a single one-sided kernel.

    The regulator $e^{-\sigma |t|}$, with $\sigma>0$, is not an artificial addition.

    It appears as the minimal symmetric damping required to make both half-axes integrable at once.

    This yields a regulated Fourier–Laplace operator transform, and the classical Fourier transform is recovered in the distributional limit $\sigma \to 0^+$.
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
        key="fourier_case"
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
# SECTION: Mellin Transform
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

    st.subheader("6.1 Differential Derivation of the Mellin Integral Form (Planting Method)")

    st.markdown("### 1. Differential Planting Foundation")
    st.markdown(r"""
    Let $f(t)$ be an analytic function expanded as a Maclaurin series:
    """)
    st.latex(r"f(t)=\sum_{n=0}^{\infty} a_n t^n")

    st.markdown(r"""
    The Mellin transform is defined in its differential planting form by
    """)
    st.latex(r"""
    MT\{f\}(\rho,s)=\sum_{n=0}^{\infty} a_n (-\partial_s)^{\rho+n-1}\!\left(\frac{1}{s}\right),
    \qquad s>0
    """)

    st.markdown(r"""
    Each term $a_n t^n$ of the series is planted as a fractional derivative of order
    $(\rho+n-1)$ applied to the seed function $(1/s)$.
    """)

    st.subheader("6.2 Fractional Derivative Identity")
    st.markdown(r"""
    For $(\Re(\alpha)>-1)$, the fractional derivative of $(1/s)$ satisfies
    """)
    st.latex(r"""
    (-\partial_s)^\alpha\!\left(\frac{1}{s}\right)=\frac{\Gamma(\alpha+1)}{s^{\alpha+1}}
    """)

    st.markdown(r"Substituting $(\alpha=\rho+n-1)$ gives")
    st.latex(r"""
    (-\partial_s)^{\rho+n-1}\!\left(\frac{1}{s}\right)=\frac{\Gamma(\rho+n)}{s^{\rho+n}}
    """)

    st.markdown(r"Hence the differential planting law becomes")
    st.latex(r"""
    MT\{f\}(\rho,s)=\sum_{n=0}^{\infty} a_n \frac{\Gamma(\rho+n)}{s^{\rho+n}}
    """)

    st.subheader("3. Transition to the Integral Representation")
    st.markdown(r"Using the Gamma integral identity")
    st.latex(r"""
    \frac{\Gamma(\rho+n)}{s^{\rho+n}}=\int_0^\infty t^{\rho+n-1}e^{-st}\,dt,
    \qquad s>0
    """)

    st.markdown(r"we substitute this into the planted series:")
    st.latex(r"""
    MT\{f\}(\rho,s)=\sum_{n=0}^{\infty} a_n \int_0^\infty t^{\rho+n-1}e^{-st}\,dt
    """)

    st.subheader("4. Interchanging Sum and Integral")
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

    st.subheader("5. Classical Mellin Limit")
    st.markdown(r"Taking the limit as $(s\to0^+)$ removes the exponential regulator:")
    st.latex(r"""
    \lim_{s\to0^+} MT\{f\}(\rho,s)=\int_0^\infty f(t)t^{\rho-1}\,dt
    """)

    st.markdown(r"""
    which is precisely the classical Mellin transform.

    Perfect agreement between the planted differential form and the classical integral form
    confirms the internal consistency.
    """)

    st.subheader("7. Interpretation")
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

    st.subheader("6.3 Integral form")
    st.markdown(r"The classical Mellin transform is defined as")
    st.latex(r"""
    \mathcal{M}\{f(t)\}(\rho)=\int_0^\infty f(t)t^{\rho-1}\,dt
    """)
    st.markdown(r"valid whenever the integral converges.")
    st.markdown(r"In the operator framework, we introduce an exponential regulator:")
    st.latex(r"""
    MT\{f(t)\}(\rho,s)=\int_0^\infty f(t)t^{\rho-1}e^{-st}\,dt,\qquad s>0
    """)

    st.subheader("6.4 Series form (planting law)")
    st.markdown(r"If $(f(t))$ has a Maclaurin expansion")
    st.latex(r"f(t)=\sum_{n=0}^{\infty} a_n t^n")
    st.markdown(r"then the transform is obtained by planting each term:")
    st.latex(r"""
    MT\{f\}(\rho,s)=\sum_{n=0}^{\infty} a_n \frac{\Gamma(\rho+n)}{s^{\rho+n}}
    """)
    st.markdown(r"""
    This is the series planting law: each \(t^n\) contributes a gamma factor \((\Gamma(\rho+n))\)
    and a regulator \((s^{-(\rho+n)})\).
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
        key="mellin_case"
    )

    rho_sym = sp.Symbol("rho", real=True)
    s_sym = sp.Symbol("s", positive=True, real=True)
    t_sym = sp.Symbol("t", positive=True, real=True)
    a_sym = sp.Symbol("a", positive=True, real=True)

    params = {"a": 1.0}
    data = mellin_transform_data(mellin_case, params)

    st.markdown("### Function")
    st.latex(r"f(t)=" + sp.latex(data["f"]))

    st.markdown("### Series")
    st.latex(data["series"])

    st.markdown("### Coefficients")
    st.latex(data["coeff"])

    st.markdown("### Rank")
    st.latex(data["rank"])

    st.markdown("### Plant & Sum")
    for step in data.get("plant_steps", []):
        st.latex(step)

    st.markdown("### Closed Regulated Form")
    st.latex(data.get("closed_text", data["planted"]))

    st.markdown("### Classical Mellin Limit")
    st.latex(data.get("classical_text", r"\lim_{s\to0^+} MT\{f\}(\rho,s) = " + sp.latex(data["classical"])))

    if data.get("extra_note"):
        st.markdown("### Cross-check / Remark")
        st.latex(data["extra_note"])

    # ============================================================
    # 6.10 Power Multiplication Property
    # ============================================================
    st.divider()
    st.subheader("6.10 Power Multiplication Property in the Generated Kernel Method")

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
    This property shows that multiplication by \(t^m\) produces a shift in the Gamma index and the power of the kernel variable \(s\).
    """)

    st.markdown("#### Example: \(f(t) = e^{-t}\)")
    st.latex(r"\mathcal{MT}_{(\rho,s)}\{t^m e^{-t}\} = \frac{\Gamma(\rho + m)}{(s + 1)^{\rho + m}}")
    st.latex(r"\lim_{s\to0^+} \mathcal{MT}_{(\rho,s)}\{t^m e^{-t}\} = \Gamma(\rho + m)")

    st.markdown(r"""
    which agrees with the classical Mellin transform result
    """)
    st.latex(r"\int_0^{\infty} t^{\rho + m - 1} e^{-t} dt = \Gamma(\rho + m)")

    # ============================================================
    # Table 3: Comparison Table
    # ============================================================
    st.divider()
    st.subheader("Table 3: Comparison Table")

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
    # 7. Fractional Derivative Link
    # ============================================================
    st.divider()
    st.header("7. Fractional Derivative Link")

    st.markdown(r"""
    We note the key operator identity:
    """)
    st.latex(r"(-\partial_s)^n\left(\frac{1}{s}\right) = \frac{n!}{s^{n+1}}")

    st.markdown(r"""
    Fractional calculus extends this to arbitrary \(\alpha\):
    """)
    st.latex(r"(-\partial_s)^\alpha\left(\frac{1}{s}\right) = \frac{\Gamma(\alpha+1)}{s^{\alpha+1}},\quad \Re(\alpha)>-1")

    st.subheader("7.1 Key Implication")
    st.markdown(r"""
    By setting \(\alpha = n + \rho - 1\), we obtain
    """)
    st.latex(r"(-\partial_s)^{n+\rho-1}\left(\frac{1}{s}\right) = \frac{\Gamma(n+\rho)}{s^{n+\rho}}")
    st.markdown("This is exactly the Mellin planting law.")

    # ============================================================
    # Table 4: Unified Operator Planting
    # ============================================================
    st.subheader("Summary: Unified Operator Planting")

    st.markdown("""
    The parameter \(\alpha\) denotes the planting rank, which may be integer, fractional, or bilateral, depending on the underlying transform.
    """)

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
