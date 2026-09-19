"""Conjuntos de dados compartilhados pelos testes de validação.

Aqui — e SOMENTE aqui, dentro de `tests/` — NumPy, SciPy e Pandas são
permitidos. Eles fazem o papel de referência independente: cada função
escrita à mão em `core/` é comparada com a implementação consagrada da
biblioteca correspondente.

TOLERÂNCIA NUMÉRICA
-------------------
Comparações de ponto flutuante usam `math.isclose` com

    rel_tol = 1e-9    e    abs_tol = 1e-12

A tolerância relativa é o critério principal: nossos somatórios são
ingênuos (acumulação sequencial em laço), enquanto o NumPy usa somatório
por pares, que agrupa os termos em árvore e acumula menos erro de
arredondamento. Para as ~17 mil observações do dataset, a diferença
esperada entre os dois métodos é da ordem de 1e-13 relativo — bem dentro
do limite. A tolerância absoluta existe só para o caso em que o valor
verdadeiro é zero, onde a tolerância relativa se anula.
"""

import math
import os

import numpy as np
import pandas as pd
import pytest

REL_TOL = 1e-9
ABS_TOL = 1e-12

RAIZ_PROJETO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMINHO_HOUR = os.path.join(RAIZ_PROJETO, "data", "hour.csv")


def assert_proximo(obtido, esperado, rel_tol=REL_TOL, abs_tol=ABS_TOL, contexto=""):
    """Compara nosso resultado com a referência da biblioteca.

    Falha com uma mensagem que mostra os dois valores e o erro absoluto e
    relativo, para que um teste vermelho diga imediatamente se o problema
    é um erro de fórmula (diferença grande) ou só ruído de ponto
    flutuante (diferença perto da tolerância).
    """
    if math.isclose(obtido, esperado, rel_tol=rel_tol, abs_tol=abs_tol):
        return

    erro_absoluto = abs(obtido - esperado)
    denominador = max(abs(obtido), abs(esperado))
    erro_relativo = erro_absoluto / denominador if denominador else float("inf")

    raise AssertionError(
        f"{contexto}\n"
        f"  nosso core : {obtido!r}\n"
        f"  referência : {esperado!r}\n"
        f"  erro abs   : {erro_absoluto:.3e} (tolerância {abs_tol:.1e})\n"
        f"  erro rel   : {erro_relativo:.3e} (tolerância {rel_tol:.1e})"
    )


# ---------------------------------------------------------------------------
# Conjuntos sintéticos — cada um cobre um cenário diferente
# ---------------------------------------------------------------------------

@pytest.fixture
def dados_simples():
    """Conjunto pequeno, conferível na mão. n par, com um valor repetido."""
    return [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]


@pytest.fixture
def dados_impares():
    """n ímpar, para exercitar o outro ramo da mediana."""
    return [10.0, 2.0, 38.0, 23.0, 38.0, 23.0, 21.0]


@pytest.fixture
def dados_normais():
    """Amostra simétrica grande, gerada com semente fixa (reprodutível)."""
    gerador = np.random.default_rng(42)
    return gerador.normal(loc=50.0, scale=12.0, size=5000).tolist()


@pytest.fixture
def dados_assimetricos():
    """Amostra com cauda longa à direita — testa quartis e outliers."""
    gerador = np.random.default_rng(7)
    return gerador.exponential(scale=3.0, size=4000).tolist()


@pytest.fixture
def dados_com_negativos():
    """Mistura de sinais, para garantir que nada assume positividade."""
    gerador = np.random.default_rng(99)
    return (gerador.normal(0.0, 5.0, size=1500)).tolist()


@pytest.fixture
def par_correlacionado():
    """Par (x, y) com associação linear forte mas não perfeita."""
    gerador = np.random.default_rng(2024)
    x = gerador.uniform(0.0, 100.0, size=3000)
    y = 3.5 * x - 20.0 + gerador.normal(0.0, 15.0, size=3000)
    return x.tolist(), y.tolist()


# ---------------------------------------------------------------------------
# Dados reais do projeto
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def bike():
    """O dataset real, para validar o core na escala em que ele é usado."""
    if not os.path.exists(CAMINHO_HOUR):
        pytest.skip(f"dataset não encontrado em {CAMINHO_HOUR}")
    return pd.read_csv(CAMINHO_HOUR)


@pytest.fixture
def cnt_real(bike):
    """Total de aluguéis por hora: contagem inteira, assimétrica à direita."""
    return bike["cnt"].astype(float).tolist()


@pytest.fixture
def temp_real(bike):
    """Temperatura normalizada: contínua, aproximadamente simétrica."""
    return bike["temp"].astype(float).tolist()
