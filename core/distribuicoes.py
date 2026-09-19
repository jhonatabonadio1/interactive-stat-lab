"""
Distribuições de probabilidade teóricas — Módulo 4.

Todas as densidades, funções de probabilidade e acumuladas são escritas
à mão a partir da definição. De `math` usamos apenas funções matemáticas
elementares e especiais (exp, log, sqrt, erf, lgamma), nunca uma função
estatística pronta do SciPy.

Sobre `math.lgamma`
-------------------
Vários cálculos exigem fatoriais. Calcular 977! diretamente produz um
inteiro gigante e estoura a faixa do float na divisão seguinte. A saída
é trabalhar em escala logarítmica usando

    ln(k!) = ln Γ(k+1) = lgamma(k+1)

e exponenciar só no final. Γ é a função gama, a extensão contínua do
fatorial — uma função matemática especial, não uma rotina estatística.
"""

import math

from core.minhastats import (
    _validar,
    desvio_padrao,
    maximo,
    media,
    minimo,
    variancia,
)

__all__ = [
    "pdf_normal",
    "cdf_normal",
    "pmf_binomial",
    "pmf_poisson",
    "pdf_uniforme",
    "cdf_uniforme",
    "pdf_exponencial",
    "cdf_exponencial",
    "estimar_normal",
    "estimar_poisson",
    "estimar_exponencial",
    "estimar_uniforme",
    "estimar_binomial",
    "DISTRIBUICOES_CONTINUAS",
    "DISTRIBUICOES_DISCRETAS",
]

SQRT_2PI = math.sqrt(2.0 * math.pi)
SQRT_2 = math.sqrt(2.0)


# ---------------------------------------------------------------------------
# Normal
# ---------------------------------------------------------------------------

def pdf_normal(x, mu, sigma):
    """Densidade da distribuição Normal N(μ, σ²).

                     1              ⎡   (x − μ)²  ⎤
        f(x) = ───────────── · exp  ⎢ − ───────── ⎥
                σ·√(2π)             ⎣     2σ²     ⎦

    Curva em sino, simétrica em torno de μ, com pontos de inflexão em
    μ ± σ. É uma DENSIDADE: f(x) não é probabilidade (pode passar de 1);
    probabilidade é a área sob a curva.
    """
    if sigma <= 0:
        raise ValueError(f"sigma precisa ser positivo, recebeu {sigma}")

    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * SQRT_2PI)


def cdf_normal(x, mu, sigma):
    """Função de distribuição acumulada da Normal: F(x) = P(X ≤ x).

    A integral da densidade normal não tem primitiva elementar, mas se
    escreve exatamente em termos da função erro:

        F(x) = ½ · [ 1 + erf( (x − μ) / (σ·√2) ) ]

    onde erf(z) = (2/√π) · ∫₀^z e^(−t²) dt, disponível em `math.erf`.
    """
    if sigma <= 0:
        raise ValueError(f"sigma precisa ser positivo, recebeu {sigma}")

    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * SQRT_2)))


def estimar_normal(dados):
    """Estima (μ, σ) de uma Normal pelos dados.

    Os estimadores de máxima verossimilhança da Normal são exatamente a
    média e o desvio padrão amostrais, então basta reusar o Módulo 1:

        μ̂ = x̄        σ̂ = s
    """
    return {"mu": media(dados), "sigma": desvio_padrao(dados, amostral=True)}


# ---------------------------------------------------------------------------
# Binomial
# ---------------------------------------------------------------------------

def pmf_binomial(k, n, p):
    """Função de probabilidade da Binomial B(n, p): P(X = k).

        P(X = k) = C(n,k) · p^k · (1−p)^(n−k),    k = 0, 1, ..., n

    Modela o número de sucessos em n ensaios de Bernoulli independentes,
    todos com a mesma probabilidade p de sucesso.

    Calculamos em escala logarítmica para que n grande não estoure:

        ln P = ln C(n,k) + k·ln(p) + (n−k)·ln(1−p)
        ln C(n,k) = lgamma(n+1) − lgamma(k+1) − lgamma(n−k+1)
    """
    if n < 0 or int(n) != n:
        raise ValueError(f"n precisa ser inteiro não negativo, recebeu {n}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p precisa estar em [0, 1], recebeu {p}")

    n = int(n)
    if k < 0 or k > n or int(k) != k:
        return 0.0
    k = int(k)

    # Casos de borda: ln(0) é indefinido, então resolvemos por definição.
    if p == 0.0:
        return 1.0 if k == 0 else 0.0
    if p == 1.0:
        return 1.0 if k == n else 0.0

    log_coeficiente = (
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    )
    log_probabilidade = (
        log_coeficiente + k * math.log(p) + (n - k) * math.log1p(-p)
    )
    return math.exp(log_probabilidade)


def estimar_binomial(dados, n_ensaios):
    """Estima p de uma Binomial com n conhecido.

    Como E[X] = n·p, igualar a média teórica à média amostral dá o
    estimador pelo método dos momentos:

        p̂ = x̄ / n
    """
    if n_ensaios <= 0:
        raise ValueError(f"n_ensaios precisa ser positivo, recebeu {n_ensaios}")

    p = media(dados) / n_ensaios
    if not 0.0 <= p <= 1.0:
        raise ValueError(
            f"p̂ = {p:.4f} fora de [0, 1]: n_ensaios={n_ensaios} é pequeno "
            f"demais para estes dados"
        )
    return {"n": int(n_ensaios), "p": p}


# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------

def pmf_poisson(k, lam):
    """Função de probabilidade da Poisson P(λ): P(X = k).

                   e^(−λ) · λ^k
        P(X = k) = ─────────────,    k = 0, 1, 2, ...
                        k!

    Modela contagens de eventos em um intervalo fixo de tempo ou espaço,
    supondo taxa média λ constante e eventos independentes. Propriedade
    característica: E[X] = Var[X] = λ. Quando a variância observada
    excede muito a média, há SOBREDISPERSÃO e a Poisson não serve — é
    exatamente o que acontece com a variável `cnt` deste dataset, e a
    discussão está no RELATORIO.md.

    Em escala logarítmica, para suportar λ e k grandes:

        ln P = −λ + k·ln(λ) − lgamma(k+1)
    """
    if lam <= 0:
        raise ValueError(f"lambda precisa ser positivo, recebeu {lam}")
    if k < 0 or int(k) != k:
        return 0.0

    k = int(k)
    log_probabilidade = -lam + k * math.log(lam) - math.lgamma(k + 1)

    # Caudas muito distantes de λ dão logaritmos fortemente negativos;
    # exp() devolveria 0.0 de qualquer forma, mas evitamos o underflow.
    if log_probabilidade < -745.0:
        return 0.0
    return math.exp(log_probabilidade)


def estimar_poisson(dados):
    """Estima λ de uma Poisson.

    Como E[X] = λ, o estimador de máxima verossimilhança é a própria
    média amostral:

        λ̂ = x̄

    Devolvemos junto o índice de dispersão Var/média, que vale 1 para uma
    Poisson genuína e serve de diagnóstico rápido do ajuste.
    """
    lam = media(dados)
    if lam <= 0:
        raise ValueError("λ̂ ≤ 0: a Poisson exige dados de contagem positivos")

    variancia_amostral = variancia(dados, amostral=True)
    return {
        "lambda": lam,
        "indice_dispersao": variancia_amostral / lam,
    }


# ---------------------------------------------------------------------------
# Uniforme contínua
# ---------------------------------------------------------------------------

def pdf_uniforme(x, a, b):
    """Densidade da Uniforme contínua U(a, b).

               ⎧ 1/(b−a),  se a ≤ x ≤ b
        f(x) = ⎨
               ⎩ 0,        caso contrário

    Todos os valores do intervalo são igualmente prováveis; a densidade
    é um retângulo de altura 1/(b−a).
    """
    if b <= a:
        raise ValueError(f"exige-se a < b, recebeu a={a}, b={b}")

    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0


def cdf_uniforme(x, a, b):
    """Acumulada da Uniforme contínua.

               ⎧ 0,            x < a
        F(x) = ⎨ (x−a)/(b−a),  a ≤ x ≤ b
               ⎩ 1,            x > b
    """
    if b <= a:
        raise ValueError(f"exige-se a < b, recebeu a={a}, b={b}")

    if x < a:
        return 0.0
    if x > b:
        return 1.0
    return (x - a) / (b - a)


def estimar_uniforme(dados):
    """Estima (a, b) de uma Uniforme pelos extremos observados.

        â = x_mín        b̂ = x_máx

    São os estimadores de máxima verossimilhança. Note que eles são
    enviesados para dentro: a amostra nunca ultrapassa os limites
    verdadeiros, então â ≥ a e b̂ ≤ b sempre.
    """
    valores = _validar(dados, minimo_exigido=2)
    a = minimo(valores)
    b = maximo(valores)
    if b <= a:
        raise ValueError("dados constantes: não é possível ajustar uma Uniforme")
    return {"a": a, "b": b}


# ---------------------------------------------------------------------------
# Exponencial
# ---------------------------------------------------------------------------

def pdf_exponencial(x, lam):
    """Densidade da Exponencial Exp(λ).

               ⎧ λ · e^(−λx),  x ≥ 0
        f(x) = ⎨
               ⎩ 0,            x < 0

    Modela o tempo de espera até o próximo evento de um processo de
    Poisson de taxa λ. É a única distribuição contínua sem memória:
    P(X > s+t | X > s) = P(X > t).
    """
    if lam <= 0:
        raise ValueError(f"lambda precisa ser positivo, recebeu {lam}")

    if x < 0:
        return 0.0
    return lam * math.exp(-lam * x)


def cdf_exponencial(x, lam):
    """Acumulada da Exponencial: F(x) = 1 − e^(−λx) para x ≥ 0."""
    if lam <= 0:
        raise ValueError(f"lambda precisa ser positivo, recebeu {lam}")

    if x < 0:
        return 0.0
    return 1.0 - math.exp(-lam * x)


def estimar_exponencial(dados):
    """Estima λ de uma Exponencial.

    Como E[X] = 1/λ, o estimador de máxima verossimilhança é o inverso
    da média amostral:

        λ̂ = 1 / x̄
    """
    x_barra = media(dados)
    if x_barra <= 0:
        raise ValueError(
            "a Exponencial exige média positiva; verifique se há valores negativos"
        )
    return {"lambda": 1.0 / x_barra, "media": x_barra}


# ---------------------------------------------------------------------------
# Catálogo consumido pela interface
# ---------------------------------------------------------------------------

DISTRIBUICOES_CONTINUAS = {
    "Normal": {
        "estimar": estimar_normal,
        "densidade": lambda x, par: pdf_normal(x, par["mu"], par["sigma"]),
        "formula": "f(x) = 1/(σ√(2π)) · exp(−(x−μ)²/(2σ²))",
        "parametros": lambda par: f"μ = {par['mu']:.4f},  σ = {par['sigma']:.4f}",
        "suporte": "todos os reais",
    },
    "Exponencial": {
        "estimar": estimar_exponencial,
        "densidade": lambda x, par: pdf_exponencial(x, par["lambda"]),
        "formula": "f(x) = λ · e^(−λx),  x ≥ 0",
        "parametros": lambda par: f"λ = {par['lambda']:.6f}  (média = {par['media']:.4f})",
        "suporte": "valores não negativos",
    },
    "Uniforme": {
        "estimar": estimar_uniforme,
        "densidade": lambda x, par: pdf_uniforme(x, par["a"], par["b"]),
        "formula": "f(x) = 1/(b−a),  a ≤ x ≤ b",
        "parametros": lambda par: f"a = {par['a']:.4f},  b = {par['b']:.4f}",
        "suporte": "intervalo [a, b]",
    },
}

DISTRIBUICOES_DISCRETAS = {
    "Poisson": {
        "estimar": estimar_poisson,
        "probabilidade": lambda k, par: pmf_poisson(k, par["lambda"]),
        "formula": "P(X = k) = e^(−λ)·λ^k / k!",
        "parametros": lambda par: (
            f"λ = {par['lambda']:.4f}  "
            f"(índice de dispersão Var/λ = {par['indice_dispersao']:.2f})"
        ),
        "suporte": "inteiros não negativos",
    },
}
