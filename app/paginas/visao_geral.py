"""Módulo 0 — apresentação do dataset, sua origem e sua estrutura."""

import pandas as pd
import streamlit as st

from app import dados as mod_dados


def renderizar(quadro, descricao_filtro):
    st.title("Laboratório Estatístico Interativo")
    st.caption(
        "Estatística descritiva, probabilidade, distribuições teóricas e "
        "regressão sobre dados reais — com todo o núcleo matemático "
        "implementado do zero."
    )

    fonte = mod_dados.FONTE
    st.markdown(f"### {fonte['nome']}")
    st.write(fonte["descricao"])
    st.markdown(f"**Fonte:** [{fonte['origem']}]({fonte['url']})")

    st.divider()

    st.markdown("#### O dataset em números")
    coluna1, coluna2, coluna3, coluna4 = st.columns(4)
    coluna1.metric("Registros", mod_dados.formatar_inteiro(len(quadro)))
    coluna2.metric("Variáveis numéricas", len(mod_dados.VARIAVEIS_NUMERICAS))
    coluna3.metric("Variáveis categóricas", len(mod_dados.VARIAVEIS_CATEGORICAS))
    coluna4.metric(
        "Período",
        f"{quadro['dteday'].min():%m/%Y} – {quadro['dteday'].max():%m/%Y}",
    )

    if descricao_filtro != "dataset completo, sem filtros":
        st.info(f"Recorte ativo: {descricao_filtro}")

    st.divider()

    esquerda, direita = st.columns(2)

    with esquerda:
        st.markdown("#### Variáveis numéricas")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Variável": meta["rotulo"],
                        "Unidade": meta["unidade"],
                        "Tipo": meta["tipo"].capitalize(),
                        "Descrição": meta["descricao"],
                    }
                    for meta in mod_dados.VARIAVEIS_NUMERICAS.values()
                ]
            ),
            hide_index=True,
        )

    with direita:
        st.markdown("#### Variáveis categóricas")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Variável": meta["rotulo"],
                        "Descrição": meta["descricao"],
                    }
                    for meta in mod_dados.VARIAVEIS_CATEGORICAS.values()
                ]
            ),
            hide_index=True,
        )

    st.divider()

    st.markdown("#### Por que escolhemos este dataset")
    st.markdown(
        """
Ele atende aos requisitos com folga — **17.379 registros**, oito variáveis
numéricas e sete categóricas — mas o motivo real da escolha é que cada módulo do
trabalho encontra aqui um caso natural, em vez de um exemplo forçado:

- **Descritiva** — o total de aluguéis por hora é fortemente assimétrico à
  direita, então média e mediana discordam de forma visível e a discussão sobre
  qual medida usar deixa de ser abstrata.
- **Teorema Central do Limite** — é justamente por partir de uma população
  assimétrica que a convergência das médias amostrais para a Normal impressiona.
- **Distribuições teóricas** — há uma variável de contagem (aluguéis, candidata a
  Poisson), uma aproximadamente simétrica (temperatura, candidata a Normal), uma
  com cauda longa (vento, candidata a Exponencial) e uma praticamente uniforme
  (hora do dia). Quatro naturezas diferentes no mesmo arquivo.
- **Regressão** — temperatura contra aluguéis produz uma reta com inclinação de
  interpretação direta: quantas bicicletas a mais por grau Celsius.
        """
    )

    with st.expander("Sobre a desnormalização das variáveis meteorológicas"):
        st.markdown(
            """
As colunas `temp`, `atemp`, `hum` e `windspeed` vêm normalizadas em [0, 1]. Para
que as medidas tivessem significado físico, reconstruímos as unidades originais.

O `Readme.txt` distribuído com o dataset em 2013 diz que `temp` foi *"dividida
por 41"*. **Essa descrição está errada.** Aplicando-a, a temperatura média de
janeiro em Washington D.C. daria 9,7 °C, contra os cerca de 1,9 °C que a cidade
de fato registra. A página atual do dataset no UCI documenta uma normalização
min-máx,

$$\\text{temp} = \\frac{t - (-8)}{39 - (-8)}$$

que devolve 3,2 °C para janeiro e acompanha a climatologia da cidade nos doze
meses. Conferimos mês a mês antes de decidir, e é a fórmula min-máx que a
aplicação usa.
            """
        )

    st.divider()

    st.markdown("#### Amostra dos dados")
    colunas_exibidas = [
        "dteday", "hora", "estacao", "clima", "dia_util",
        "temp_c", "atemp_c", "hum_pct", "vento_kmh",
        "casual", "registered", "cnt",
    ]
    st.dataframe(
        quadro[colunas_exibidas].head(100),
        hide_index=True,
    )

    st.caption(f"Citação obrigatória — {fonte['citacao']}")
