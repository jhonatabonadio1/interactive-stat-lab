"""Validação do Módulo 2 — tabelas de frequência contra NumPy e Pandas."""

import math

import numpy as np
import pandas as pd
import pytest

from core import frequencias as fr
from conftest import assert_proximo


CONJUNTOS = ["dados_normais", "dados_assimetricos", "cnt_real", "temp_real"]
todos_os_conjuntos = pytest.mark.parametrize("nome_conjunto", CONJUNTOS)


@pytest.fixture
def conjunto(request, nome_conjunto):
    return request.getfixturevalue(nome_conjunto)


# ---------------------------------------------------------------------------
# Regra de Sturges
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n, esperado",
    [
        (1, 1),
        (2, 2),        # 1 + log₂(2)     = 2      → 2
        (10, 5),       # 1 + log₂(10)    = 4.322  → 5
        (100, 8),      # 1 + log₂(100)   = 7.644  → 8
        (1000, 11),    # 1 + log₂(1000)  = 10.966 → 11
        (17379, 16),   # 1 + log₂(17379) = 15.085 → 16  (o dataset completo)
    ],
)
def test_sturges_confere_com_o_calculo_manual(n, esperado):
    assert fr.regra_de_sturges(n) == esperado


@pytest.mark.parametrize("n", [2, 7, 50, 500, 5000, 17379])
def test_sturges_equivale_a_teto_de_um_mais_log2(n):
    assert fr.regra_de_sturges(n) == math.ceil(1 + math.log2(n))


@pytest.mark.parametrize("expoente", [1, 2, 4, 8, 10, 14])
def test_sturges_e_exato_em_potencias_de_dois(expoente):
    """Onde a constante arredondada 3,322 erraria para cima por uma classe."""
    n = 2**expoente
    assert fr.regra_de_sturges(n) == expoente + 1


def test_sturges_rejeita_n_invalido():
    with pytest.raises(ValueError):
        fr.regra_de_sturges(0)


# ---------------------------------------------------------------------------
# Tabela de frequências para variável contínua
# ---------------------------------------------------------------------------

@todos_os_conjuntos
def test_contagens_por_classe_conferem_com_numpy_histogram(conjunto, nome_conjunto):
    """np.histogram usa a mesma convenção: [a,b) em todas as classes menos a última."""
    classes, meta = fr.tabela_frequencias_continua(conjunto)

    contagens_referencia, _ = np.histogram(
        conjunto, bins=meta["n_classes"], range=(meta["minimo"], meta["maximo"])
    )

    nossas_contagens = [classe["fi"] for classe in classes]
    assert nossas_contagens == contagens_referencia.tolist(), (
        f"contagens divergiram de np.histogram em '{nome_conjunto}'"
    )


@todos_os_conjuntos
def test_limites_das_classes_conferem_com_numpy(conjunto, nome_conjunto):
    classes, meta = fr.tabela_frequencias_continua(conjunto)
    _, bordas = np.histogram(
        conjunto, bins=meta["n_classes"], range=(meta["minimo"], meta["maximo"])
    )

    for i, classe in enumerate(classes):
        assert_proximo(
            classe["limite_inferior"],
            float(bordas[i]),
            rel_tol=1e-9,
            abs_tol=1e-9,
            contexto=f"limite inferior da classe {i+1} em '{nome_conjunto}'",
        )
        assert_proximo(
            classe["limite_superior"],
            float(bordas[i + 1]),
            rel_tol=1e-9,
            abs_tol=1e-9,
            contexto=f"limite superior da classe {i+1} em '{nome_conjunto}'",
        )


@todos_os_conjuntos
def test_frequencias_somam_o_total(conjunto, nome_conjunto):
    classes, meta = fr.tabela_frequencias_continua(conjunto)

    assert sum(c["fi"] for c in classes) == meta["n"], (
        f"nenhuma observação pode se perder em '{nome_conjunto}'"
    )
    assert_proximo(sum(c["fri"] for c in classes), 1.0, abs_tol=1e-9)


@todos_os_conjuntos
def test_frequencia_acumulada_e_monotona_e_fecha_em_n(conjunto, nome_conjunto):
    classes, meta = fr.tabela_frequencias_continua(conjunto)

    anterior = 0
    for classe in classes:
        assert classe["Fi"] >= anterior, f"Fi não é monótona em '{nome_conjunto}'"
        anterior = classe["Fi"]

    assert classes[-1]["Fi"] == meta["n"]
    assert_proximo(classes[-1]["Fri"], 1.0, abs_tol=1e-9)


@todos_os_conjuntos
def test_classes_sao_contiguas_e_de_mesma_amplitude(conjunto, nome_conjunto):
    classes, meta = fr.tabela_frequencias_continua(conjunto)

    for anterior, seguinte in zip(classes, classes[1:]):
        assert_proximo(
            anterior["limite_superior"],
            seguinte["limite_inferior"],
            rel_tol=1e-9,
            abs_tol=1e-9,
            contexto=f"classes com buraco entre elas em '{nome_conjunto}'",
        )

    for classe in classes:
        largura = classe["limite_superior"] - classe["limite_inferior"]
        assert_proximo(largura, meta["amplitude_classe"], rel_tol=1e-9, abs_tol=1e-9)


@todos_os_conjuntos
def test_ponto_medio_e_o_centro_da_classe(conjunto, nome_conjunto):
    classes, _ = fr.tabela_frequencias_continua(conjunto)
    for classe in classes:
        esperado = (classe["limite_inferior"] + classe["limite_superior"]) / 2.0
        assert_proximo(classe["ponto_medio"], esperado, abs_tol=1e-9)


def test_numero_de_classes_pode_ser_escolhido(dados_normais):
    classes, meta = fr.tabela_frequencias_continua(dados_normais, n_classes=20)
    assert len(classes) == 20
    assert meta["n_classes"] == 20


def test_valor_maximo_cai_na_ultima_classe(dados_assimetricos):
    """O máximo é o único ponto fechado à direita; não pode transbordar."""
    classes, meta = fr.tabela_frequencias_continua(dados_assimetricos)
    assert classes[-1]["limite_superior"] == pytest.approx(meta["maximo"])
    assert classes[-1]["fi"] >= 1


def test_variavel_constante_gera_classe_unica():
    classes, meta = fr.tabela_frequencias_continua([5.0] * 100)
    assert len(classes) == 1
    assert classes[0]["fi"] == 100
    assert meta["amplitude_total"] == 0.0


def test_n_classes_invalido_levanta_erro(dados_normais):
    with pytest.raises(ValueError):
        fr.tabela_frequencias_continua(dados_normais, n_classes=0)


# ---------------------------------------------------------------------------
# Tabela de frequências para variável categórica
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("coluna", ["season", "weathersit", "workingday", "holiday", "weekday"])
def test_contagem_categorica_confere_com_pandas(bike, coluna):
    valores = bike[coluna].tolist()
    linhas, meta = fr.tabela_frequencias_categorica(valores)

    referencia = pd.Series(valores).astype(str).value_counts()

    assert meta["n"] == len(valores)
    assert meta["n_categorias"] == len(referencia)

    for linha in linhas:
        assert linha["fi"] == int(referencia[linha["categoria"]]), (
            f"contagem divergiu de pandas.value_counts() na coluna '{coluna}', "
            f"categoria '{linha['categoria']}'"
        )


def test_frequencias_relativas_categoricas_somam_um(bike):
    linhas, _ = fr.tabela_frequencias_categorica(bike["season"].tolist())
    assert_proximo(sum(linha["fri"] for linha in linhas), 1.0, abs_tol=1e-9)
    assert_proximo(linhas[-1]["Fri"], 1.0, abs_tol=1e-9)


def test_ordenacao_por_frequencia_e_decrescente(bike):
    linhas, _ = fr.tabela_frequencias_categorica(
        bike["weathersit"].tolist(), ordenar_por_frequencia=True
    )
    contagens = [linha["fi"] for linha in linhas]
    assert contagens == sorted(contagens, reverse=True)


def test_ordenacao_alfabetica_quando_pedida(bike):
    linhas, _ = fr.tabela_frequencias_categorica(
        bike["season"].tolist(), ordenar_por_frequencia=False
    )
    categorias = [linha["categoria"] for linha in linhas]
    assert categorias == sorted(categorias)


def test_categoria_unica():
    linhas, meta = fr.tabela_frequencias_categorica(["sim"] * 42)
    assert meta["n_categorias"] == 1
    assert linhas[0]["fi"] == 42
    assert linhas[0]["fri"] == pytest.approx(1.0)


def test_dados_categoricos_vazios_levantam_erro():
    with pytest.raises(ValueError):
        fr.tabela_frequencias_categorica([])
