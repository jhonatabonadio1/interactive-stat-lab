"""
Laboratório Estatístico Interativo — ponto de entrada da aplicação.

Execução:
    streamlit run app/principal.py

Esta camada NÃO calcula estatística. Ela lê dados, monta controles,
chama `core/` e desenha o resultado. Toda fórmula está em `core/`.

Por que o arquivo não se chama `app.py`
---------------------------------------
O Streamlit registra o script executado em `sys.modules` sob o nome do
arquivo. Um `app/app.py` seria registrado como módulo `app` e passaria a
sombrear o PACOTE `app`, quebrando `from app import dados` com um erro de
importação circular. O nome `principal.py` evita a colisão.

Permite `streamlit run app/principal.py` a partir da raiz do projeto sem
instalar o pacote: a raiz precisa estar em sys.path para `import core`
funcionar.
"""

import os
import sys

import streamlit as st

RAIZ_PROJETO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ_PROJETO not in sys.path:
    sys.path.insert(0, RAIZ_PROJETO)

from app import dados as mod_dados  # noqa: E402
from app import tema  # noqa: E402
from app.paginas import (  # noqa: E402
    descritiva,
    distribuicoes,
    probabilidade,
    regressao,
    visao_geral,
)

EQUIPE = ["Plinio", "Paulo Cesar", "Jhonata"]

PAGINAS = {
    "Visão geral do dataset": visao_geral,
    "1 · Estatística descritiva": descritiva,
    "2 · Probabilidade e simulação": probabilidade,
    "3 · Distribuições teóricas": distribuicoes,
    "4 · Correlação e regressão": regressao,
}


def montar_filtros(quadro):
    """Filtros globais na barra lateral, aplicados a todas as páginas.

    Devolve (quadro_filtrado, descricao_legivel_do_recorte).
    """
    st.sidebar.markdown("### Recorte dos dados")

    anos = st.sidebar.multiselect(
        "Ano",
        options=sorted(quadro["ano"].unique()),
        default=sorted(quadro["ano"].unique()),
    )
    estacoes = st.sidebar.multiselect(
        "Estação do ano",
        options=list(mod_dados.ROTULOS_ESTACAO.values()),
        default=list(mod_dados.ROTULOS_ESTACAO.values()),
    )
    climas = st.sidebar.multiselect(
        "Condição climática",
        options=list(mod_dados.ROTULOS_CLIMA.values()),
        default=list(mod_dados.ROTULOS_CLIMA.values()),
    )
    tipo_dia = st.sidebar.radio(
        "Tipo de dia",
        options=["Todos", "Apenas dias úteis", "Apenas fins de semana e feriados"],
        index=0,
    )

    filtrado = quadro[
        quadro["ano"].isin(anos)
        & quadro["estacao"].isin(estacoes)
        & quadro["clima"].isin(climas)
    ]

    if tipo_dia == "Apenas dias úteis":
        filtrado = filtrado[filtrado["workingday"] == 1]
    elif tipo_dia == "Apenas fins de semana e feriados":
        filtrado = filtrado[filtrado["workingday"] == 0]

    partes = []
    if len(anos) < quadro["ano"].nunique():
        partes.append("anos: " + ", ".join(anos))
    if len(estacoes) < 4:
        partes.append("estações: " + ", ".join(estacoes))
    if len(climas) < 4:
        partes.append("clima: " + ", ".join(climas))
    if tipo_dia != "Todos":
        partes.append(tipo_dia.lower())

    descricao = " · ".join(partes) if partes else "dataset completo, sem filtros"

    st.sidebar.caption(
        f"**{len(filtrado):,} de {len(quadro):,} registros** selecionados".replace(
            ",", "."
        )
    )

    return filtrado, descricao


def main():
    st.set_page_config(
        page_title="Laboratório Estatístico Interativo",
        page_icon="📊",
        layout="wide",
    )
    tema.aplicar_estilo()

    st.sidebar.title("Laboratório Estatístico")
    st.sidebar.caption("Matemática e Estatística para Computação")

    nome_pagina = st.sidebar.radio("Módulo", list(PAGINAS.keys()), label_visibility="collapsed")
    st.sidebar.divider()

    try:
        quadro = mod_dados.carregar_dados()
    except FileNotFoundError as erro:
        st.error(str(erro))
        st.stop()

    filtrado, descricao_filtro = montar_filtros(quadro)

    if filtrado.empty:
        st.warning(
            "Nenhum registro sobrou depois dos filtros. "
            "Amplie a seleção na barra lateral para continuar."
        )
        st.stop()

    st.sidebar.divider()
    st.sidebar.markdown("### Equipe")
    for integrante in EQUIPE:
        st.sidebar.caption(integrante)

    PAGINAS[nome_pagina].renderizar(filtrado, descricao_filtro)


if __name__ == "__main__":
    main()
