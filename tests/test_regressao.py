"""Validação do Módulo 5 — mínimos quadrados contra SciPy e NumPy."""

import numpy as np
import pytest
from scipy import stats

from core import minhastats as ms
from core.regressao import ajustar_regressao_linear
from conftest import assert_proximo


@pytest.fixture
def par_real(bike):
    """Temperatura × total de aluguéis: o par usado na demonstração do projeto."""
    return (
        bike["temp"].astype(float).tolist(),
        bike["cnt"].astype(float).tolist(),
    )


PARES_REAIS = [
    ("temp", "cnt"),
    ("atemp", "cnt"),
    ("hum", "cnt"),
    ("windspeed", "cnt"),
    ("casual", "registered"),
    ("temp", "atemp"),
]


# ---------------------------------------------------------------------------
# Coeficientes
# ---------------------------------------------------------------------------

def test_coeficientes_conferem_com_scipy_linregress(par_correlacionado):
    x, y = par_correlacionado
    modelo = ajustar_regressao_linear(x, y)
    referencia = stats.linregress(x, y)

    assert_proximo(
        modelo.b1,
        float(referencia.slope),
        contexto="b₁ divergiu de scipy.stats.linregress().slope",
    )
    assert_proximo(
        modelo.b0,
        float(referencia.intercept),
        contexto="b₀ divergiu de scipy.stats.linregress().intercept",
    )


def test_coeficientes_conferem_com_numpy_polyfit(par_correlacionado):
    x, y = par_correlacionado
    modelo = ajustar_regressao_linear(x, y)
    b1_ref, b0_ref = np.polyfit(x, y, deg=1)

    assert_proximo(modelo.b1, float(b1_ref), rel_tol=1e-8, contexto="b₁ vs np.polyfit")
    assert_proximo(modelo.b0, float(b0_ref), rel_tol=1e-8, contexto="b₀ vs np.polyfit")


@pytest.mark.parametrize("coluna_x, coluna_y", PARES_REAIS)
def test_coeficientes_conferem_em_dados_reais(bike, coluna_x, coluna_y):
    x = bike[coluna_x].astype(float).tolist()
    y = bike[coluna_y].astype(float).tolist()

    modelo = ajustar_regressao_linear(x, y)
    referencia = stats.linregress(x, y)

    assert_proximo(
        modelo.b1,
        float(referencia.slope),
        contexto=f"b₁ divergiu em ({coluna_x} → {coluna_y})",
    )
    assert_proximo(
        modelo.b0,
        float(referencia.intercept),
        contexto=f"b₀ divergiu em ({coluna_x} → {coluna_y})",
    )


# ---------------------------------------------------------------------------
# Qualidade do ajuste
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("coluna_x, coluna_y", PARES_REAIS)
def test_r2_confere_com_scipy(bike, coluna_x, coluna_y):
    x = bike[coluna_x].astype(float).tolist()
    y = bike[coluna_y].astype(float).tolist()

    modelo = ajustar_regressao_linear(x, y)
    referencia = stats.linregress(x, y)

    assert_proximo(
        modelo.r2,
        float(referencia.rvalue) ** 2,
        contexto=f"R² divergiu em ({coluna_x} → {coluna_y})",
    )


@pytest.mark.parametrize("coluna_x, coluna_y", PARES_REAIS)
def test_r_quadrado_e_igual_ao_quadrado_da_correlacao(bike, coluna_x, coluna_y):
    """Identidade exclusiva da regressão SIMPLES: R² = r²."""
    x = bike[coluna_x].astype(float).tolist()
    y = bike[coluna_y].astype(float).tolist()

    modelo = ajustar_regressao_linear(x, y)
    r = ms.correlacao_pearson(x, y)

    assert_proximo(
        modelo.r2,
        r * r,
        contexto=f"R² ≠ r² em ({coluna_x} → {coluna_y})",
    )


def test_erro_padrao_do_coeficiente_confere_com_scipy(par_correlacionado):
    x, y = par_correlacionado
    modelo = ajustar_regressao_linear(x, y)
    referencia = stats.linregress(x, y)

    assert_proximo(
        modelo.erro_padrao_b1,
        float(referencia.stderr),
        contexto="erro padrão de b₁ divergiu de scipy.stats.linregress().stderr",
    )


def test_erro_padrao_da_estimativa_confere_com_a_definicao(par_correlacionado):
    x, y = par_correlacionado
    modelo = ajustar_regressao_linear(x, y)

    coeficientes = np.polyfit(x, y, deg=1)
    residuos = np.asarray(y) - np.polyval(coeficientes, np.asarray(x))
    esperado = float(np.sqrt((residuos**2).sum() / (len(x) - 2)))

    assert_proximo(modelo.erro_padrao_estimativa, esperado, rel_tol=1e-8)


# ---------------------------------------------------------------------------
# Propriedades matemáticas do ajuste
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("coluna_x, coluna_y", PARES_REAIS)
def test_decomposicao_da_soma_de_quadrados(bike, coluna_x, coluna_y):
    """SQ_tot = SQ_reg + SQ_res — a identidade fundamental da ANOVA."""
    x = bike[coluna_x].astype(float).tolist()
    y = bike[coluna_y].astype(float).tolist()

    modelo = ajustar_regressao_linear(x, y)
    assert_proximo(
        modelo.sq_total,
        modelo.sq_regressao + modelo.sq_residual,
        rel_tol=1e-9,
        contexto=f"decomposição falhou em ({coluna_x} → {coluna_y})",
    )


def test_reta_passa_pelo_ponto_medio(par_correlacionado):
    """Consequência direta de b₀ = ȳ − b₁·x̄."""
    x, y = par_correlacionado
    modelo = ajustar_regressao_linear(x, y)

    assert_proximo(
        modelo.prever(ms.media(x)),
        ms.media(y),
        rel_tol=1e-9,
        contexto="a reta ajustada deveria passar por (x̄, ȳ)",
    )


def test_soma_dos_residuos_e_nula(par_correlacionado):
    """Com intercepto no modelo, Σe_i = 0 pela primeira equação normal."""
    x, y = par_correlacionado
    modelo = ajustar_regressao_linear(x, y)

    residuos = modelo.residuos(x, y)
    escala = max(abs(v) for v in y)
    assert abs(sum(residuos)) < 1e-9 * escala * len(x)


def test_residuos_sao_ortogonais_a_x(par_correlacionado):
    """Σ x_i·e_i = 0 pela segunda equação normal."""
    x, y = par_correlacionado
    modelo = ajustar_regressao_linear(x, y)
    residuos = modelo.residuos(x, y)

    produto_interno = sum(x[i] * residuos[i] for i in range(len(x)))
    escala = sum(abs(x[i]) for i in range(len(x))) * max(abs(v) for v in y)
    assert abs(produto_interno) < 1e-9 * escala


def test_r2_fica_entre_zero_e_um(bike):
    for coluna_x, coluna_y in PARES_REAIS:
        x = bike[coluna_x].astype(float).tolist()
        y = bike[coluna_y].astype(float).tolist()
        modelo = ajustar_regressao_linear(x, y)
        assert 0.0 <= modelo.r2 <= 1.0, f"R² fora de [0,1] em ({coluna_x}, {coluna_y})"


def test_ajuste_perfeito_em_dados_exatamente_lineares():
    x = [float(i) for i in range(50)]
    y = [2.5 * v - 8.0 for v in x]

    modelo = ajustar_regressao_linear(x, y)
    assert modelo.b1 == pytest.approx(2.5)
    assert modelo.b0 == pytest.approx(-8.0)
    assert modelo.r2 == pytest.approx(1.0)
    assert modelo.sq_residual == pytest.approx(0.0, abs=1e-18)


def test_inclinacao_negativa_em_relacao_decrescente():
    x = [float(i) for i in range(30)]
    y = [100.0 - 3.0 * v for v in x]
    modelo = ajustar_regressao_linear(x, y)
    assert modelo.b1 == pytest.approx(-3.0)
    assert modelo.r == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Predição
# ---------------------------------------------------------------------------

def test_predicao_confere_com_numpy_polyval(par_correlacionado):
    x, y = par_correlacionado
    modelo = ajustar_regressao_linear(x, y)

    coeficientes = np.polyfit(x, y, deg=1)
    pontos = [0.0, 10.0, 25.5, 50.0, 99.9]

    for ponto in pontos:
        assert_proximo(
            modelo.prever(ponto),
            float(np.polyval(coeficientes, ponto)),
            rel_tol=1e-7,
            abs_tol=1e-7,
            contexto=f"predição divergiu em x={ponto}",
        )


def test_prever_varios_equivale_a_prever_um_a_um(par_correlacionado):
    x, y = par_correlacionado
    modelo = ajustar_regressao_linear(x, y)
    pontos = [1.0, 2.0, 3.0]
    assert modelo.prever_varios(pontos) == [modelo.prever(p) for p in pontos]


def test_extrapolacao_e_sinalizada(par_real):
    x, y = par_real
    modelo = ajustar_regressao_linear(x, y)

    assert modelo.extrapola(modelo.x_maximo + 1.0)
    assert modelo.extrapola(modelo.x_minimo - 1.0)
    assert not modelo.extrapola((modelo.x_minimo + modelo.x_maximo) / 2.0)


def test_equacao_formata_o_sinal_corretamente():
    x = [float(i) for i in range(20)]

    crescente = ajustar_regressao_linear(x, [2.0 * v + 1.0 for v in x])
    assert "+" in crescente.equacao()

    decrescente = ajustar_regressao_linear(x, [-2.0 * v + 1.0 for v in x])
    assert "−" in decrescente.equacao()


# ---------------------------------------------------------------------------
# Entradas inválidas
# ---------------------------------------------------------------------------

def test_x_constante_levanta_erro():
    with pytest.raises(ValueError, match="X é constante"):
        ajustar_regressao_linear([3.0] * 10, [float(i) for i in range(10)])


def test_y_constante_levanta_erro():
    with pytest.raises(ValueError, match="Y é constante"):
        ajustar_regressao_linear([float(i) for i in range(10)], [7.0] * 10)


def test_poucos_pontos_levantam_erro():
    """Com n = 2 a reta passa exata pelos pontos e s_e teria divisor zero."""
    with pytest.raises(ValueError):
        ajustar_regressao_linear([1.0, 2.0], [3.0, 4.0])


def test_tamanhos_diferentes_levantam_erro():
    with pytest.raises(ValueError):
        ajustar_regressao_linear([1.0, 2.0, 3.0], [1.0, 2.0])
