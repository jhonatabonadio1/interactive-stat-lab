"""Validação do Módulo 1 — cada função de `core/minhastats.py` contra NumPy/SciPy.

Estratégia: para toda medida implementada à mão existe pelo menos um
teste que a compara com a função equivalente de uma biblioteca
consagrada, sobre vários conjuntos de dados (pequenos e conferíveis na
mão, grandes e aleatórios, simétricos, assimétricos, com negativos, e o
dataset real do projeto).

A tolerância numérica e o helper `assert_proximo` estão documentados em
`tests/conftest.py`.
"""

import math

import numpy as np
import pytest
from scipy import stats

from core import minhastats as ms
from conftest import assert_proximo


# Aplicado à maioria dos testes: os mesmos conjuntos, um por cenário.
CONJUNTOS = [
    "dados_simples",
    "dados_impares",
    "dados_normais",
    "dados_assimetricos",
    "dados_com_negativos",
    "cnt_real",
    "temp_real",
]

todos_os_conjuntos = pytest.mark.parametrize("nome_conjunto", CONJUNTOS)


@pytest.fixture
def conjunto(request, nome_conjunto):
    """Resolve o nome do conjunto em seus valores (indireção da parametrização)."""
    return request.getfixturevalue(nome_conjunto)


# ---------------------------------------------------------------------------
# Tendência central
# ---------------------------------------------------------------------------

@todos_os_conjuntos
def test_media_confere_com_numpy(conjunto, nome_conjunto):
    assert_proximo(
        ms.media(conjunto),
        float(np.mean(conjunto)),
        contexto=f"media() divergiu de np.mean() em '{nome_conjunto}'",
    )


@todos_os_conjuntos
def test_mediana_confere_com_numpy(conjunto, nome_conjunto):
    assert_proximo(
        ms.mediana(conjunto),
        float(np.median(conjunto)),
        contexto=f"mediana() divergiu de np.median() em '{nome_conjunto}'",
    )


def test_mediana_n_par_e_a_media_dos_dois_centrais(dados_simples):
    # Ordenado: 2, 4, 4, 4, 5, 5, 7, 9 → centrais são 4 e 5.
    assert ms.mediana(dados_simples) == pytest.approx(4.5)


def test_mediana_n_impar_e_o_elemento_central(dados_impares):
    # Ordenado: 2, 10, 21, 23, 23, 38, 38 → central é 23.
    assert ms.mediana(dados_impares) == pytest.approx(23.0)


@todos_os_conjuntos
def test_moda_confere_com_scipy(conjunto, nome_conjunto):
    """SciPy devolve só a menor moda; comparamos com o mínimo da nossa lista.

    Quando todo valor aparece uma única vez consideramos o conjunto
    amodal e devolvemos lista vazia — convenção diferente da do SciPy,
    que devolveria o menor valor. Esse caso é verificado à parte.
    """
    nossa_moda = ms.moda(conjunto)
    referencia = stats.mode(np.asarray(conjunto), keepdims=False)

    if referencia.count == 1:
        assert nossa_moda == [], (
            f"'{nome_conjunto}' é amodal (toda frequência = 1), "
            f"esperávamos lista vazia mas veio {nossa_moda}"
        )
        return

    assert nossa_moda, f"moda() devolveu lista vazia em '{nome_conjunto}'"
    assert_proximo(
        min(nossa_moda),
        float(referencia.mode),
        contexto=f"moda() divergiu de scipy.stats.mode() em '{nome_conjunto}'",
    )

    # Toda moda que devolvemos precisa de fato ter a frequência máxima.
    valores, contagens = np.unique(np.asarray(conjunto), return_counts=True)
    frequencia_maxima = int(contagens.max())
    for valor in nossa_moda:
        indice = int(np.searchsorted(valores, valor))
        assert int(contagens[indice]) == frequencia_maxima


def test_moda_detecta_multimodalidade():
    # 1 e 3 aparecem duas vezes cada: conjunto bimodal.
    assert ms.moda([1.0, 1.0, 2.0, 3.0, 3.0]) == [1.0, 3.0]


def test_moda_de_conjunto_amodal_e_vazia():
    assert ms.moda([1.0, 2.0, 3.0, 4.0]) == []


# ---------------------------------------------------------------------------
# Dispersão
# ---------------------------------------------------------------------------

@todos_os_conjuntos
def test_amplitude_confere_com_numpy(conjunto, nome_conjunto):
    assert_proximo(
        ms.amplitude(conjunto),
        float(np.ptp(conjunto)),
        contexto=f"amplitude() divergiu de np.ptp() em '{nome_conjunto}'",
    )


@todos_os_conjuntos
def test_minimo_e_maximo_conferem_com_numpy(conjunto, nome_conjunto):
    assert_proximo(ms.minimo(conjunto), float(np.min(conjunto)), contexto=nome_conjunto)
    assert_proximo(ms.maximo(conjunto), float(np.max(conjunto)), contexto=nome_conjunto)


@todos_os_conjuntos
def test_variancia_amostral_confere_com_numpy_ddof1(conjunto, nome_conjunto):
    assert_proximo(
        ms.variancia(conjunto, amostral=True),
        float(np.var(conjunto, ddof=1)),
        contexto=f"variancia(amostral=True) divergiu de np.var(ddof=1) em '{nome_conjunto}'",
    )


@todos_os_conjuntos
def test_variancia_populacional_confere_com_numpy_ddof0(conjunto, nome_conjunto):
    assert_proximo(
        ms.variancia(conjunto, amostral=False),
        float(np.var(conjunto, ddof=0)),
        contexto=f"variancia(amostral=False) divergiu de np.var(ddof=0) em '{nome_conjunto}'",
    )


@todos_os_conjuntos
def test_desvio_padrao_confere_com_numpy(conjunto, nome_conjunto):
    assert_proximo(
        ms.desvio_padrao(conjunto, amostral=True),
        float(np.std(conjunto, ddof=1)),
        contexto=f"desvio_padrao(amostral=True) em '{nome_conjunto}'",
    )
    assert_proximo(
        ms.desvio_padrao(conjunto, amostral=False),
        float(np.std(conjunto, ddof=0)),
        contexto=f"desvio_padrao(amostral=False) em '{nome_conjunto}'",
    )


def test_correcao_de_bessel_torna_a_variancia_amostral_maior(dados_normais):
    """s² > σ² sempre, porque o divisor (n−1) é menor que n."""
    assert ms.variancia(dados_normais, amostral=True) > ms.variancia(
        dados_normais, amostral=False
    )


@todos_os_conjuntos
def test_coeficiente_variacao_confere_com_a_definicao(conjunto, nome_conjunto):
    esperado = float(np.std(conjunto, ddof=1)) / abs(float(np.mean(conjunto))) * 100.0
    assert_proximo(
        ms.coeficiente_variacao(conjunto, amostral=True),
        esperado,
        contexto=f"coeficiente_variacao() em '{nome_conjunto}'",
    )


def test_coeficiente_variacao_populacional_confere_com_scipy(dados_normais):
    """scipy.stats.variation usa ddof=0 e não multiplica por 100."""
    assert_proximo(
        ms.coeficiente_variacao(dados_normais, amostral=False, em_percentual=False),
        float(stats.variation(dados_normais)),
        contexto="coeficiente_variacao() divergiu de scipy.stats.variation()",
    )


# ---------------------------------------------------------------------------
# Separatrizes
# ---------------------------------------------------------------------------

@todos_os_conjuntos
@pytest.mark.parametrize("p", [0, 1, 5, 10, 25, 33.333, 50, 66.667, 75, 90, 95, 99, 100])
def test_percentil_confere_com_numpy(conjunto, nome_conjunto, p):
    assert_proximo(
        ms.percentil(conjunto, p),
        float(np.percentile(conjunto, p, method="linear")),
        contexto=f"percentil(p={p}) divergiu de np.percentile() em '{nome_conjunto}'",
    )


@todos_os_conjuntos
def test_quartis_conferem_com_numpy(conjunto, nome_conjunto):
    q1, q2, q3 = ms.quartis(conjunto)
    esperados = np.percentile(conjunto, [25, 50, 75], method="linear")

    for obtido, esperado, nome in zip((q1, q2, q3), esperados, ("Q1", "Q2", "Q3")):
        assert_proximo(
            obtido,
            float(esperado),
            contexto=f"{nome} divergiu de np.percentile() em '{nome_conjunto}'",
        )


@todos_os_conjuntos
def test_q2_e_identico_a_mediana(conjunto, nome_conjunto):
    _, q2, _ = ms.quartis(conjunto)
    assert_proximo(q2, ms.mediana(conjunto), contexto=f"Q2 ≠ mediana em '{nome_conjunto}'")


@todos_os_conjuntos
def test_amplitude_interquartil_confere_com_scipy(conjunto, nome_conjunto):
    assert_proximo(
        ms.amplitude_interquartil(conjunto),
        float(stats.iqr(conjunto, interpolation="linear")),
        contexto=f"amplitude_interquartil() divergiu de scipy.stats.iqr() em '{nome_conjunto}'",
    )


@todos_os_conjuntos
def test_quartis_sao_ordenados(conjunto, nome_conjunto):
    q1, q2, q3 = ms.quartis(conjunto)
    assert q1 <= q2 <= q3, f"quartis fora de ordem em '{nome_conjunto}'"


# ---------------------------------------------------------------------------
# Forma e outliers
# ---------------------------------------------------------------------------

@todos_os_conjuntos
def test_assimetria_pearson_confere_com_a_formula(conjunto, nome_conjunto):
    esperado = (
        3.0
        * (float(np.mean(conjunto)) - float(np.median(conjunto)))
        / float(np.std(conjunto, ddof=1))
    )
    assert_proximo(
        ms.assimetria_pearson(conjunto),
        esperado,
        contexto=f"assimetria_pearson() em '{nome_conjunto}'",
    )


def test_assimetria_positiva_em_distribuicao_com_cauda_a_direita(dados_assimetricos):
    assert ms.assimetria_pearson(dados_assimetricos) > 0


def test_assimetria_quase_nula_em_distribuicao_simetrica(dados_normais):
    assert abs(ms.assimetria_pearson(dados_normais)) < 0.1


@todos_os_conjuntos
def test_limites_de_outliers_conferem_com_numpy(conjunto, nome_conjunto):
    q1, q3 = np.percentile(conjunto, [25, 75], method="linear")
    iqr = q3 - q1

    inferior, superior = ms.limites_outliers_iqr(conjunto, fator=1.5)
    assert_proximo(inferior, float(q1 - 1.5 * iqr), contexto=nome_conjunto)
    assert_proximo(superior, float(q3 + 1.5 * iqr), contexto=nome_conjunto)


@todos_os_conjuntos
def test_outliers_iqr_seleciona_exatamente_os_valores_fora_das_cercas(
    conjunto, nome_conjunto
):
    resultado = ms.outliers_iqr(conjunto, fator=1.5)

    vetor = np.asarray(conjunto, dtype=float)
    mascara = (vetor < resultado["limite_inferior"]) | (vetor > resultado["limite_superior"])

    assert resultado["quantidade"] == int(mascara.sum()), (
        f"contagem de outliers divergiu em '{nome_conjunto}'"
    )
    assert resultado["indices"] == np.flatnonzero(mascara).tolist()
    assert resultado["proporcao"] == pytest.approx(mascara.mean())


def test_outlier_evidente_e_detectado():
    dados = [10.0] * 50 + [1000.0]
    resultado = ms.outliers_iqr(dados)
    assert resultado["valores"] == [1000.0]


# ---------------------------------------------------------------------------
# Medidas bivariadas
# ---------------------------------------------------------------------------

def test_covariancia_amostral_confere_com_numpy(par_correlacionado):
    x, y = par_correlacionado
    assert_proximo(
        ms.covariancia(x, y, amostral=True),
        float(np.cov(x, y, ddof=1)[0, 1]),
        contexto="covariancia(amostral=True) divergiu de np.cov(ddof=1)",
    )


def test_covariancia_populacional_confere_com_numpy(par_correlacionado):
    x, y = par_correlacionado
    assert_proximo(
        ms.covariancia(x, y, amostral=False),
        float(np.cov(x, y, ddof=0)[0, 1]),
        contexto="covariancia(amostral=False) divergiu de np.cov(ddof=0)",
    )


def test_covariancia_de_x_com_ele_mesmo_e_a_variancia(dados_normais):
    assert_proximo(
        ms.covariancia(dados_normais, dados_normais),
        ms.variancia(dados_normais, amostral=True),
        contexto="cov(x, x) deveria ser igual a var(x)",
    )


def test_covariancia_e_simetrica(par_correlacionado):
    x, y = par_correlacionado
    assert_proximo(ms.covariancia(x, y), ms.covariancia(y, x))


def test_correlacao_confere_com_numpy(par_correlacionado):
    x, y = par_correlacionado
    assert_proximo(
        ms.correlacao_pearson(x, y),
        float(np.corrcoef(x, y)[0, 1]),
        contexto="correlacao_pearson() divergiu de np.corrcoef()",
    )


def test_correlacao_confere_com_scipy(par_correlacionado):
    x, y = par_correlacionado
    esperado, _ = stats.pearsonr(x, y)
    assert_proximo(
        ms.correlacao_pearson(x, y),
        float(esperado),
        contexto="correlacao_pearson() divergiu de scipy.stats.pearsonr()",
    )


def test_correlacao_em_variaveis_reais_do_dataset(bike):
    """Validação na escala e no formato em que a aplicação realmente usa."""
    pares = [("temp", "cnt"), ("hum", "cnt"), ("atemp", "temp"), ("casual", "registered")]

    for coluna_x, coluna_y in pares:
        x = bike[coluna_x].astype(float).tolist()
        y = bike[coluna_y].astype(float).tolist()
        assert_proximo(
            ms.correlacao_pearson(x, y),
            float(np.corrcoef(x, y)[0, 1]),
            contexto=f"correlacao_pearson() divergiu em ({coluna_x}, {coluna_y})",
        )


def test_correlacao_perfeita_positiva():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [3.0 * v + 7.0 for v in x]
    assert ms.correlacao_pearson(x, y) == pytest.approx(1.0)


def test_correlacao_perfeita_negativa():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [-2.0 * v + 1.0 for v in x]
    assert ms.correlacao_pearson(x, y) == pytest.approx(-1.0)


@todos_os_conjuntos
def test_correlacao_permanece_no_intervalo_valido(conjunto, nome_conjunto):
    """r ∈ [−1, +1] por Cauchy-Schwarz."""
    metade = len(conjunto) // 2
    x, y = conjunto[:metade], conjunto[metade : 2 * metade]
    r = ms.correlacao_pearson(x, y)
    assert -1.0 <= r <= 1.0, f"r fora de [-1, 1] em '{nome_conjunto}': {r}"


# ---------------------------------------------------------------------------
# Resumo agregado
# ---------------------------------------------------------------------------

def test_resumo_descritivo_bate_com_as_funcoes_individuais(cnt_real):
    resumo = ms.resumo_descritivo(cnt_real)

    assert resumo["n"] == len(cnt_real)
    assert_proximo(resumo["media"], float(np.mean(cnt_real)))
    assert_proximo(resumo["mediana"], float(np.median(cnt_real)))
    assert_proximo(resumo["desvio_padrao_amostral"], float(np.std(cnt_real, ddof=1)))
    assert_proximo(resumo["variancia_populacional"], float(np.var(cnt_real, ddof=0)))
    assert_proximo(resumo["q1"], float(np.percentile(cnt_real, 25)))
    assert_proximo(resumo["q3"], float(np.percentile(cnt_real, 75)))
    assert_proximo(resumo["iqr"], float(stats.iqr(cnt_real)))
    assert_proximo(resumo["amplitude"], float(np.ptp(cnt_real)))


# ---------------------------------------------------------------------------
# Validação de entradas inválidas
# ---------------------------------------------------------------------------

def test_conjunto_vazio_levanta_erro():
    with pytest.raises(ValueError):
        ms.media([])


def test_variancia_amostral_exige_duas_observacoes():
    with pytest.raises(ValueError):
        ms.variancia([5.0], amostral=True)


def test_variancia_populacional_aceita_uma_observacao():
    assert ms.variancia([5.0], amostral=False) == 0.0


def test_valor_nao_numerico_levanta_typeerror():
    with pytest.raises(TypeError):
        ms.media([1.0, "dois", 3.0])


def test_nan_levanta_valueerror():
    with pytest.raises(ValueError):
        ms.media([1.0, float("nan"), 3.0])


def test_percentil_fora_do_intervalo_levanta_erro(dados_simples):
    with pytest.raises(ValueError):
        ms.percentil(dados_simples, 101)
    with pytest.raises(ValueError):
        ms.percentil(dados_simples, -1)


def test_tamanhos_diferentes_levantam_erro():
    with pytest.raises(ValueError):
        ms.correlacao_pearson([1.0, 2.0, 3.0], [1.0, 2.0])


def test_correlacao_com_variavel_constante_levanta_erro():
    with pytest.raises(ValueError):
        ms.correlacao_pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])


def test_cv_com_media_zero_levanta_erro():
    with pytest.raises(ValueError):
        ms.coeficiente_variacao([-1.0, 0.0, 1.0])
