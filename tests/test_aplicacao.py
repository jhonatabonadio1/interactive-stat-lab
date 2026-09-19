"""Testes de fumaça da interface Streamlit.

Diferente dos demais arquivos de teste, aqui não validamos matemática —
isso já foi feito módulo a módulo. O objetivo é garantir que cada página
da aplicação **renderiza sem levantar exceção** para as combinações de
controle que um usuário pode escolher, incluindo os casos de borda que
costumam quebrar (variável constante depois de um filtro agressivo,
distribuição incompatível com o suporte da variável, X = Y na regressão).

`AppTest` executa o script do Streamlit num ambiente simulado, sem
navegador, e expõe os widgets para que possamos alterá-los e reexecutar.
"""

import os

import pytest
from streamlit.testing.v1 import AppTest

RAIZ_PROJETO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMINHO_APP = os.path.join(RAIZ_PROJETO, "app", "principal.py")

PAGINAS = [
    "Visão geral do dataset",
    "1 · Estatística descritiva",
    "2 · Probabilidade e simulação",
    "3 · Distribuições teóricas",
    "4 · Correlação e regressão",
]


def _sem_excecao(teste, contexto):
    if teste.exception:
        mensagens = "\n".join(
            f"  {e.message}\n{''.join((e.stack_trace or [])[-8:])}"
            for e in teste.exception
        )
        raise AssertionError(f"exceção ao renderizar {contexto}:\n{mensagens}")


@pytest.fixture
def aplicacao():
    teste = AppTest.from_file(CAMINHO_APP, default_timeout=300)
    teste.run()
    _sem_excecao(teste, "o carregamento inicial")
    return teste


def _abrir(teste, pagina):
    teste.sidebar.radio[0].set_value(pagina).run()
    _sem_excecao(teste, f"a página '{pagina}'")
    return teste


@pytest.mark.parametrize("pagina", PAGINAS)
def test_cada_pagina_renderiza(aplicacao, pagina):
    _abrir(aplicacao, pagina)


def test_pagina_inicial_mostra_o_tamanho_do_dataset(aplicacao):
    valores = [metrica.value for metrica in aplicacao.metric]
    assert "17.379" in valores


# ---------------------------------------------------------------------------
# Módulo 2 — todas as variáveis, numéricas e categóricas
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("indice_variavel", range(8))
def test_descritiva_aceita_todas_as_variaveis_numericas(aplicacao, indice_variavel):
    teste = _abrir(aplicacao, "1 · Estatística descritiva")

    opcoes = teste.selectbox[0].options
    teste.selectbox[0].set_value(opcoes[indice_variavel]).run()
    _sem_excecao(teste, f"a variável numérica '{opcoes[indice_variavel]}'")

    assert teste.metric, "esperávamos as medidas descritivas na tela"


@pytest.mark.parametrize("indice_variavel", range(7))
def test_descritiva_aceita_todas_as_variaveis_categoricas(aplicacao, indice_variavel):
    teste = _abrir(aplicacao, "1 · Estatística descritiva")

    teste.radio[0].set_value("Categórica").run()
    _sem_excecao(teste, "a troca para variáveis categóricas")

    opcoes = teste.selectbox[0].options
    teste.selectbox[0].set_value(opcoes[indice_variavel]).run()
    _sem_excecao(teste, f"a variável categórica '{opcoes[indice_variavel]}'")


@pytest.mark.parametrize("n_classes", [5, 16, 60])
def test_descritiva_aceita_os_extremos_do_controle_de_classes(aplicacao, n_classes):
    teste = _abrir(aplicacao, "1 · Estatística descritiva")
    teste.slider[0].set_value(n_classes).run()
    _sem_excecao(teste, f"o histograma com {n_classes} classes")


# ---------------------------------------------------------------------------
# Módulo 4 — cada distribuição candidata de cada variável
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("indice_variavel", range(8))
def test_todas_as_candidatas_de_cada_variavel_ajustam(aplicacao, indice_variavel):
    teste = _abrir(aplicacao, "3 · Distribuições teóricas")

    opcoes_variavel = teste.selectbox[0].options
    teste.selectbox[0].set_value(opcoes_variavel[indice_variavel]).run()
    _sem_excecao(teste, f"a variável '{opcoes_variavel[indice_variavel]}'")

    for familia in teste.selectbox[1].options:
        teste.selectbox[1].set_value(familia).run()
        _sem_excecao(
            teste,
            f"{familia} sobre '{opcoes_variavel[indice_variavel]}'",
        )


# ---------------------------------------------------------------------------
# Módulo 5 — pares de variáveis, inclusive o par degenerado
# ---------------------------------------------------------------------------

def test_regressao_avisa_quando_x_e_y_sao_iguais(aplicacao):
    teste = _abrir(aplicacao, "4 · Correlação e regressão")

    mesma_variavel = teste.selectbox[0].value
    teste.selectbox[1].set_value(mesma_variavel).run()
    _sem_excecao(teste, "X = Y na regressão")

    assert teste.warning, "esperávamos um aviso em vez de um R² = 1 sem sentido"


@pytest.mark.parametrize("indice_y", range(8))
def test_regressao_funciona_com_temperatura_contra_cada_resposta(aplicacao, indice_y):
    teste = _abrir(aplicacao, "4 · Correlação e regressão")

    opcoes = teste.selectbox[1].options
    alvo = opcoes[indice_y]

    if alvo == teste.selectbox[0].value:
        pytest.skip("X = Y é coberto por outro teste")

    teste.selectbox[1].set_value(alvo).run()
    _sem_excecao(teste, f"a regressão de '{alvo}' sobre temperatura")


def test_predicao_interativa_responde_ao_valor_digitado(aplicacao):
    teste = _abrir(aplicacao, "4 · Correlação e regressão")

    teste.number_input[0].set_value(25.0).run()
    _sem_excecao(teste, "a predição em X = 25")

    rotulos = [metrica.label for metrica in teste.metric]
    assert any("Predição" in rotulo for rotulo in rotulos)


def test_extrapolacao_gera_aviso(aplicacao):
    teste = _abrir(aplicacao, "4 · Correlação e regressão")

    # Muito acima da temperatura máxima observada (39 °C).
    teste.number_input[0].set_value(500.0).run()
    _sem_excecao(teste, "a predição extrapolada")

    assert any("Extrapolação" in aviso.value for aviso in teste.warning)


# ---------------------------------------------------------------------------
# Módulo 3 — simulações nos limites dos controles
# ---------------------------------------------------------------------------

def test_simulacao_da_moeda_roda_no_menor_e_no_maior_n(aplicacao):
    teste = _abrir(aplicacao, "2 · Probabilidade e simulação")

    controle = teste.select_slider[0]
    for n in (controle.options[0], controle.options[-1]):
        teste.select_slider[0].set_value(n).run()
        _sem_excecao(teste, f"a simulação da moeda com n = {n}")


def test_simulacao_do_dado_roda(aplicacao):
    teste = _abrir(aplicacao, "2 · Probabilidade e simulação")

    teste.selectbox[0].set_value("Dado").run()
    _sem_excecao(teste, "a simulação do dado")


# ---------------------------------------------------------------------------
# Filtros globais
# ---------------------------------------------------------------------------

def test_filtro_por_ano_reduz_o_conjunto(aplicacao):
    aplicacao.sidebar.multiselect[0].set_value(["2011"]).run()
    _sem_excecao(aplicacao, "o filtro de ano")

    legenda = " ".join(item.value for item in aplicacao.sidebar.caption)
    assert "8.645" in legenda, f"contagem inesperada na legenda: {legenda}"


def test_filtro_vazio_avisa_em_vez_de_quebrar(aplicacao):
    """Filtrar até não sobrar nada é fácil e não pode explodir."""
    aplicacao.sidebar.multiselect[0].set_value([]).run()
    _sem_excecao(aplicacao, "o filtro que não seleciona nada")

    assert aplicacao.warning, "esperávamos um aviso de conjunto vazio"


def test_recorte_agressivo_ainda_renderiza_a_descritiva(aplicacao):
    """Um recorte pequeno pode tornar variáveis quase constantes."""
    teste = _abrir(aplicacao, "1 · Estatística descritiva")

    teste.sidebar.multiselect[0].set_value(["2011"]).run()
    teste.sidebar.multiselect[1].set_value(["Inverno"]).run()
    teste.sidebar.multiselect[2].set_value(["Chuva ou neve leve"]).run()
    _sem_excecao(teste, "a descritiva sobre um recorte muito estreito")
