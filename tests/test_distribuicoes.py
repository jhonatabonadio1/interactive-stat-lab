"""Validação do Módulo 4 — distribuições teóricas contra scipy.stats."""

import math

import numpy as np
import pytest
from scipy import stats

from core import distribuicoes as dist
from conftest import assert_proximo


# ---------------------------------------------------------------------------
# Normal
# ---------------------------------------------------------------------------

PARAMETROS_NORMAL = [
    (0.0, 1.0),        # normal padrão
    (50.0, 12.0),      # escala do dataset
    (-3.5, 0.25),      # média negativa, desvio pequeno
    (189.46, 181.39),  # média e desvio reais de `cnt`
]


@pytest.mark.parametrize("mu, sigma", PARAMETROS_NORMAL)
@pytest.mark.parametrize("desvios", [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0])
def test_pdf_normal_confere_com_scipy(mu, sigma, desvios):
    x = mu + desvios * sigma
    assert_proximo(
        dist.pdf_normal(x, mu, sigma),
        float(stats.norm.pdf(x, loc=mu, scale=sigma)),
        contexto=f"pdf_normal(x={x}, μ={mu}, σ={sigma})",
    )


@pytest.mark.parametrize("mu, sigma", PARAMETROS_NORMAL)
@pytest.mark.parametrize("desvios", [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0])
def test_cdf_normal_confere_com_scipy(mu, sigma, desvios):
    x = mu + desvios * sigma
    assert_proximo(
        dist.cdf_normal(x, mu, sigma),
        float(stats.norm.cdf(x, loc=mu, scale=sigma)),
        contexto=f"cdf_normal(x={x}, μ={mu}, σ={sigma})",
    )


def test_densidade_normal_e_simetrica_em_torno_da_media():
    mu, sigma = 10.0, 3.0
    for delta in (0.5, 1.0, 2.5, 7.0):
        assert dist.pdf_normal(mu - delta, mu, sigma) == pytest.approx(
            dist.pdf_normal(mu + delta, mu, sigma)
        )


def test_cdf_normal_na_media_vale_meio():
    assert dist.cdf_normal(7.0, 7.0, 2.0) == pytest.approx(0.5)


def test_regra_empirica_68_95_99():
    """Um intervalo de ±k·σ deve conter as proporções conhecidas."""
    esperados = {1: 0.6827, 2: 0.9545, 3: 0.9973}
    for k, esperado in esperados.items():
        area = dist.cdf_normal(k, 0.0, 1.0) - dist.cdf_normal(-k, 0.0, 1.0)
        assert area == pytest.approx(esperado, abs=1e-4)


def test_densidade_normal_integra_um():
    """Integração numérica por trapézios sobre ±10σ."""
    mu, sigma = 5.0, 2.0
    grade = np.linspace(mu - 10 * sigma, mu + 10 * sigma, 200001)
    valores = [dist.pdf_normal(float(x), mu, sigma) for x in grade]
    area = np.trapezoid(valores, grade)
    assert float(area) == pytest.approx(1.0, abs=1e-6)


def test_sigma_invalido_levanta_erro():
    with pytest.raises(ValueError):
        dist.pdf_normal(0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        dist.cdf_normal(0.0, 0.0, -1.0)


# ---------------------------------------------------------------------------
# Binomial
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, p", [(10, 0.5), (20, 0.3), (100, 0.07), (500, 0.9), (1000, 0.5)])
def test_pmf_binomial_confere_com_scipy(n, p):
    for k in range(0, n + 1, max(1, n // 25)):
        assert_proximo(
            dist.pmf_binomial(k, n, p),
            float(stats.binom.pmf(k, n, p)),
            rel_tol=1e-9,
            abs_tol=1e-15,
            contexto=f"pmf_binomial(k={k}, n={n}, p={p})",
        )


@pytest.mark.parametrize("n, p", [(10, 0.5), (50, 0.2), (200, 0.65)])
def test_probabilidades_binomiais_somam_um(n, p):
    total = sum(dist.pmf_binomial(k, n, p) for k in range(n + 1))
    assert total == pytest.approx(1.0, abs=1e-10)


def test_binomial_fora_do_suporte_vale_zero():
    assert dist.pmf_binomial(-1, 10, 0.5) == 0.0
    assert dist.pmf_binomial(11, 10, 0.5) == 0.0
    assert dist.pmf_binomial(2.5, 10, 0.5) == 0.0


def test_binomial_degenerada():
    assert dist.pmf_binomial(0, 10, 0.0) == 1.0
    assert dist.pmf_binomial(3, 10, 0.0) == 0.0
    assert dist.pmf_binomial(10, 10, 1.0) == 1.0
    assert dist.pmf_binomial(9, 10, 1.0) == 0.0


def test_binomial_com_n_grande_nao_estoura():
    """Em escala linear, C(5000, 2500) estouraria o float."""
    valor = dist.pmf_binomial(2500, 5000, 0.5)
    assert valor > 0.0
    assert math.isfinite(valor)
    assert valor == pytest.approx(float(stats.binom.pmf(2500, 5000, 0.5)), rel=1e-9)


def test_parametros_binomiais_invalidos():
    with pytest.raises(ValueError):
        dist.pmf_binomial(1, 10, 1.5)
    with pytest.raises(ValueError):
        dist.pmf_binomial(1, -5, 0.5)


# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lam", [0.1, 1.0, 4.5, 25.0, 189.46, 1000.0])
def test_pmf_poisson_confere_com_scipy(lam):
    # Varre uma faixa em torno de λ, onde está praticamente toda a massa.
    inicio = max(0, int(lam - 5 * math.sqrt(lam)))
    fim = int(lam + 5 * math.sqrt(lam)) + 2

    for k in range(inicio, fim):
        assert_proximo(
            dist.pmf_poisson(k, lam),
            float(stats.poisson.pmf(k, lam)),
            rel_tol=1e-9,
            abs_tol=1e-15,
            contexto=f"pmf_poisson(k={k}, λ={lam})",
        )


@pytest.mark.parametrize("lam", [0.5, 3.0, 20.0, 189.46])
def test_probabilidades_poisson_somam_um(lam):
    limite = int(lam + 12 * math.sqrt(lam)) + 30
    total = sum(dist.pmf_poisson(k, lam) for k in range(limite))
    assert total == pytest.approx(1.0, abs=1e-9)


def test_poisson_com_lambda_grande_nao_estoura():
    """977! em escala linear é impraticável; o cálculo é feito em log."""
    valor = dist.pmf_poisson(977, 189.46)
    assert math.isfinite(valor)
    assert valor >= 0.0


def test_poisson_fora_do_suporte_vale_zero():
    assert dist.pmf_poisson(-1, 5.0) == 0.0
    assert dist.pmf_poisson(2.7, 5.0) == 0.0


def test_lambda_invalido_levanta_erro():
    with pytest.raises(ValueError):
        dist.pmf_poisson(1, 0.0)
    with pytest.raises(ValueError):
        dist.pmf_poisson(1, -2.0)


# ---------------------------------------------------------------------------
# Uniforme
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a, b", [(0.0, 1.0), (-5.0, 5.0), (0.02, 0.98)])
@pytest.mark.parametrize("fracao", [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
def test_pdf_uniforme_confere_com_scipy(a, b, fracao):
    x = a + fracao * (b - a)
    assert_proximo(
        dist.pdf_uniforme(x, a, b),
        float(stats.uniform.pdf(x, loc=a, scale=b - a)),
        abs_tol=1e-12,
        contexto=f"pdf_uniforme(x={x}, a={a}, b={b})",
    )


@pytest.mark.parametrize("a, b", [(0.0, 1.0), (-5.0, 5.0), (2.0, 3.5)])
@pytest.mark.parametrize("fracao", [-0.5, 0.0, 0.3, 0.7, 1.0, 1.4])
def test_cdf_uniforme_confere_com_scipy(a, b, fracao):
    x = a + fracao * (b - a)
    assert_proximo(
        dist.cdf_uniforme(x, a, b),
        float(stats.uniform.cdf(x, loc=a, scale=b - a)),
        abs_tol=1e-12,
        contexto=f"cdf_uniforme(x={x}, a={a}, b={b})",
    )


def test_densidade_uniforme_e_constante_no_suporte():
    altura = dist.pdf_uniforme(0.5, 0.0, 4.0)
    for x in (0.0, 1.0, 2.0, 3.9, 4.0):
        assert dist.pdf_uniforme(x, 0.0, 4.0) == pytest.approx(altura)
    assert altura == pytest.approx(0.25)


def test_uniforme_com_limites_invertidos_levanta_erro():
    with pytest.raises(ValueError):
        dist.pdf_uniforme(1.0, 5.0, 2.0)


# ---------------------------------------------------------------------------
# Exponencial
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lam", [0.1, 0.5, 1.0, 3.0, 12.7])
@pytest.mark.parametrize("x", [0.0, 0.25, 1.0, 2.5, 10.0])
def test_pdf_exponencial_confere_com_scipy(lam, x):
    assert_proximo(
        dist.pdf_exponencial(x, lam),
        float(stats.expon.pdf(x, scale=1.0 / lam)),
        abs_tol=1e-15,
        contexto=f"pdf_exponencial(x={x}, λ={lam})",
    )


@pytest.mark.parametrize("lam", [0.1, 1.0, 3.0])
@pytest.mark.parametrize("x", [0.0, 0.5, 2.0, 8.0])
def test_cdf_exponencial_confere_com_scipy(lam, x):
    assert_proximo(
        dist.cdf_exponencial(x, lam),
        float(stats.expon.cdf(x, scale=1.0 / lam)),
        abs_tol=1e-15,
        contexto=f"cdf_exponencial(x={x}, λ={lam})",
    )


def test_exponencial_e_zero_para_x_negativo():
    assert dist.pdf_exponencial(-1.0, 2.0) == 0.0
    assert dist.cdf_exponencial(-1.0, 2.0) == 0.0


def test_exponencial_nao_tem_memoria():
    """P(X > s+t | X > s) = P(X > t), a propriedade que a caracteriza."""
    lam, s, t = 1.7, 2.0, 3.0

    sobrevivencia = lambda v: 1.0 - dist.cdf_exponencial(v, lam)
    condicional = sobrevivencia(s + t) / sobrevivencia(s)

    assert condicional == pytest.approx(sobrevivencia(t))


# ---------------------------------------------------------------------------
# Estimação de parâmetros a partir dos dados
# ---------------------------------------------------------------------------

def test_estimar_normal_confere_com_scipy_fit(dados_normais):
    """scipy.stats.norm.fit usa MV (ddof=0); ajustamos a comparação."""
    parametros = dist.estimar_normal(dados_normais)

    mu_ref, sigma_ref = stats.norm.fit(dados_normais)
    n = len(dados_normais)
    # Convertemos σ de máxima verossimilhança para a versão amostral.
    sigma_amostral = sigma_ref * math.sqrt(n / (n - 1))

    assert_proximo(parametros["mu"], float(mu_ref), contexto="μ̂")
    assert_proximo(parametros["sigma"], float(sigma_amostral), contexto="σ̂")


def test_estimar_normal_recupera_os_parametros_verdadeiros():
    gerador = np.random.default_rng(11)
    amostra = gerador.normal(loc=100.0, scale=15.0, size=200000).tolist()

    parametros = dist.estimar_normal(amostra)
    assert parametros["mu"] == pytest.approx(100.0, abs=0.2)
    assert parametros["sigma"] == pytest.approx(15.0, abs=0.2)


def test_estimar_exponencial_recupera_lambda():
    gerador = np.random.default_rng(13)
    amostra = gerador.exponential(scale=1.0 / 2.5, size=200000).tolist()

    parametros = dist.estimar_exponencial(amostra)
    assert parametros["lambda"] == pytest.approx(2.5, rel=0.02)


def test_estimar_poisson_recupera_lambda_e_detecta_equidispersao():
    gerador = np.random.default_rng(17)
    amostra = gerador.poisson(lam=8.0, size=200000).astype(float).tolist()

    parametros = dist.estimar_poisson(amostra)
    assert parametros["lambda"] == pytest.approx(8.0, rel=0.02)
    # Numa Poisson genuína, Var ≈ λ, então o índice fica perto de 1.
    assert parametros["indice_dispersao"] == pytest.approx(1.0, abs=0.05)


def test_indice_de_dispersao_detecta_sobredispersao_em_cnt(cnt_real):
    """`cnt` é muito mais dispersa que uma Poisson — achado do Módulo 4."""
    parametros = dist.estimar_poisson(cnt_real)
    assert parametros["indice_dispersao"] > 100.0, (
        "esperávamos sobredispersão severa em cnt"
    )


def test_estimar_uniforme_usa_os_extremos(dados_normais):
    parametros = dist.estimar_uniforme(dados_normais)
    assert parametros["a"] == pytest.approx(float(np.min(dados_normais)))
    assert parametros["b"] == pytest.approx(float(np.max(dados_normais)))


def test_estimar_binomial_recupera_p():
    n_ensaios = 20
    gerador = np.random.default_rng(23)
    amostra = gerador.binomial(n_ensaios, 0.35, size=100000).astype(float).tolist()

    parametros = dist.estimar_binomial(amostra, n_ensaios)
    assert parametros["p"] == pytest.approx(0.35, abs=0.01)


def test_estimar_binomial_rejeita_n_pequeno_demais():
    with pytest.raises(ValueError, match="fora de"):
        dist.estimar_binomial([10.0, 12.0, 15.0], n_ensaios=5)


def test_estimar_uniforme_rejeita_dados_constantes():
    with pytest.raises(ValueError):
        dist.estimar_uniforme([3.0] * 10)


# ---------------------------------------------------------------------------
# Catálogo usado pela interface
# ---------------------------------------------------------------------------

def test_catalogo_continuo_funciona_de_ponta_a_ponta(temp_real):
    for nome, definicao in dist.DISTRIBUICOES_CONTINUAS.items():
        parametros = definicao["estimar"](temp_real)
        valor = definicao["densidade"](0.5, parametros)

        assert valor >= 0.0, f"densidade negativa em {nome}"
        assert math.isfinite(valor), f"densidade não finita em {nome}"
        assert isinstance(definicao["parametros"](parametros), str)


def test_catalogo_discreto_funciona_de_ponta_a_ponta(cnt_real):
    for nome, definicao in dist.DISTRIBUICOES_DISCRETAS.items():
        parametros = definicao["estimar"](cnt_real)
        valor = definicao["probabilidade"](100, parametros)

        assert 0.0 <= valor <= 1.0, f"probabilidade fora de [0,1] em {nome}"
        assert isinstance(definicao["parametros"](parametros), str)
