"""Módulo 2 — estatística descritiva interativa."""

import pandas as pd
import streamlit as st

from app import dados as mod_dados
from app import graficos
from core import frequencias as fr
from core import minhastats as ms


def _formatar(valor, casas=3):
    if valor is None:
        return "—"
    return f"{valor:,.{casas}f}".replace(",", " ")


def _classificar_dispersao(cv):
    """Faixas usuais de leitura do coeficiente de variação."""
    if cv is None:
        return "indefinida (a média é zero)"
    if cv < 15:
        return "**baixa** — os dados são homogêneos em torno da média"
    if cv < 30:
        return "**moderada**"
    return "**alta** — a média sozinha representa mal este conjunto"


def _classificar_assimetria(assimetria):
    if assimetria is None:
        return None, "indefinida"
    magnitude = abs(assimetria)
    if magnitude < 0.15:
        return "simétrica", "praticamente **simétrica**"
    if magnitude < 0.5:
        intensidade = "moderada"
    else:
        intensidade = "forte"
    direcao = "à direita" if assimetria > 0 else "à esquerda"
    return direcao, f"com assimetria **{intensidade} {direcao}**"


def _interpretar_numerica(resumo, atipicos, meta):
    """Gera a leitura automática das medidas.

    O texto é montado a partir dos números que o nosso core acabou de
    produzir — a ideia é que o usuário não precise saber de antemão o que
    olhar numa tabela de doze medidas.
    """
    unidade = f" {meta['unidade']}" if meta["unidade"] else ""
    partes = []

    media, mediana = resumo["media"], resumo["mediana"]
    direcao, descricao_assimetria = _classificar_assimetria(resumo["assimetria"])

    if direcao == "simétrica":
        partes.append(
            f"A média ({_formatar(media)}{unidade}) e a mediana "
            f"({_formatar(mediana)}{unidade}) estão muito próximas, então a "
            f"distribuição é {descricao_assimetria} e as duas medidas contam a "
            f"mesma história."
        )
    elif direcao == "à direita":
        partes.append(
            f"A média ({_formatar(media)}{unidade}) é **maior** que a mediana "
            f"({_formatar(mediana)}{unidade}). Isso indica uma distribuição "
            f"{descricao_assimetria}: existem valores altos que puxam a média "
            f"para cima, enquanto a mediana — que só depende da posição central "
            f"— resiste a eles. Para descrever o caso típico, a **mediana é a "
            f"medida mais honesta** aqui."
        )
    elif direcao == "à esquerda":
        partes.append(
            f"A média ({_formatar(media)}{unidade}) é **menor** que a mediana "
            f"({_formatar(mediana)}{unidade}), o que caracteriza uma "
            f"distribuição {descricao_assimetria}: há valores baixos puxando a "
            f"média para baixo."
        )

    cv = resumo["coeficiente_variacao"]
    if cv is not None:
        partes.append(
            f"O coeficiente de variação é **{_formatar(cv, 1)}%**, ou seja, o "
            f"desvio padrão equivale a {_formatar(cv, 1)}% da média. A "
            f"dispersão relativa é {_classificar_dispersao(cv)}."
        )

    proporcao = atipicos["proporcao"]
    if atipicos["quantidade"] == 0:
        partes.append(
            "Pela regra do IQR, **nenhuma observação é atípica**: todos os "
            "valores caem dentro das cercas de Tukey."
        )
    else:
        partes.append(
            f"A regra do IQR aponta "
            f"**{mod_dados.formatar_inteiro(atipicos['quantidade'])} observações "
            f"atípicas ({proporcao:.2%})**, todas acima de "
            f"{_formatar(atipicos['limite_superior'])}{unidade} ou abaixo de "
            f"{_formatar(atipicos['limite_inferior'])}{unidade}."
        )
        if proporcao > 0.05:
            partes.append(
                "Atenção à interpretação: com mais de 5% dos pontos marcados "
                "como atípicos, provavelmente **não se trata de erro de medição, "
                "e sim de uma cauda longa genuína** da distribuição. A regra do "
                "IQR pressupõe simetria aproximada, e aplicá-la a dados "
                "assimétricos naturalmente sinaliza muitos pontos à direita."
            )

    modas = resumo["moda"]
    if not modas:
        partes.append(
            "A variável é **amodal** no sentido estrito — nenhum valor se "
            "repete mais que os outros, o que é comum em variáveis contínuas."
        )
    elif len(modas) == 1:
        partes.append(f"O valor mais frequente (moda) é **{_formatar(modas[0])}**{unidade}.")
    elif len(modas) <= 5:
        lista = ", ".join(_formatar(m) for m in modas)
        partes.append(f"A distribuição é **multimodal**, com {len(modas)} modas: {lista}.")
    else:
        partes.append(
            f"A distribuição tem **{len(modas)} valores empatados** na "
            f"frequência máxima, o que torna a moda pouco informativa aqui."
        )

    return partes


def _renderizar_numerica(quadro, chave):
    meta = mod_dados.VARIAVEIS_NUMERICAS[chave]
    valores = mod_dados.serie_numerica(quadro, chave)
    rotulo = mod_dados.rotulo_completo(chave)

    st.caption(meta["descricao"])

    resumo = ms.resumo_descritivo(valores)
    atipicos = ms.outliers_iqr(valores)

    st.markdown("#### Medidas de tendência central e dispersão")

    linha1 = st.columns(4)
    linha1[0].metric("n", mod_dados.formatar_inteiro(resumo["n"]))
    linha1[1].metric("Média", _formatar(resumo["media"], 2))
    linha1[2].metric("Mediana", _formatar(resumo["mediana"], 2))
    linha1[3].metric(
        "Moda",
        "amodal" if not resumo["moda"] else _formatar(resumo["moda"][0], 2),
        help="Quando há empate, mostramos a menor das modas.",
    )

    linha2 = st.columns(4)
    linha2[0].metric("Desvio padrão (amostral)", _formatar(resumo["desvio_padrao_amostral"], 2))
    linha2[1].metric("Variância (amostral)", _formatar(resumo["variancia_amostral"], 2))
    linha2[2].metric("Amplitude", _formatar(resumo["amplitude"], 2))
    linha2[3].metric(
        "Coef. de variação",
        "—" if resumo["coeficiente_variacao"] is None
        else f"{resumo['coeficiente_variacao']:.1f}%",
    )

    linha3 = st.columns(4)
    linha3[0].metric("Mínimo", _formatar(resumo["minimo"], 2))
    linha3[1].metric("Q1", _formatar(resumo["q1"], 2))
    linha3[2].metric("Q3", _formatar(resumo["q3"], 2))
    linha3[3].metric("Máximo", _formatar(resumo["maximo"], 2))

    with st.expander("Ver também as medidas populacionais e a tabela completa"):
        st.dataframe(
            pd.DataFrame(
                [
                    {"Medida": "Variância populacional (σ², divisor n)",
                     "Valor": resumo["variancia_populacional"]},
                    {"Medida": "Desvio padrão populacional (σ)",
                     "Valor": resumo["desvio_padrao_populacional"]},
                    {"Medida": "Variância amostral (s², divisor n−1)",
                     "Valor": resumo["variancia_amostral"]},
                    {"Medida": "Desvio padrão amostral (s)",
                     "Valor": resumo["desvio_padrao_amostral"]},
                    {"Medida": "Amplitude interquartil (IQR = Q3 − Q1)",
                     "Valor": resumo["iqr"]},
                    {"Medida": "Assimetria de Pearson  3·(x̄ − Md)/s",
                     "Valor": resumo["assimetria"]},
                ]
            ),
            hide_index=True,
        )
        st.caption(
            "A diferença entre as versões amostral e populacional é o divisor: "
            "n−1 contra n (correção de Bessel). Com n grande elas praticamente "
            "coincidem, e é por isso que os dois valores acima são tão parecidos."
        )

    st.divider()
    st.markdown("#### Distribuição")

    n_sugerido = fr.regra_de_sturges(len(valores))
    n_classes = st.slider(
        "Número de classes do histograma",
        min_value=5,
        max_value=60,
        value=min(n_sugerido, 60),
        help=(
            f"A regra de Sturges sugere k = ⌈1 + log₂({len(valores)})⌉ = "
            f"{n_sugerido} classes. Mexa no controle para ver como a forma "
            f"aparente da distribuição depende dessa escolha."
        ),
    )

    classes, meta_classes = fr.tabela_frequencias_continua(valores, n_classes=n_classes)

    esquerda, direita = st.columns([3, 2])
    with esquerda:
        st.pyplot(graficos.histograma(classes, rotulo))
    with direita:
        st.markdown("**Tabela de frequências por classe**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Classe": f"[{c['limite_inferior']:.2f} ; {c['limite_superior']:.2f})",
                        "Ponto médio": round(c["ponto_medio"], 2),
                        "fi": c["fi"],
                        "fri": f"{c['fri']:.2%}",
                        "Fi": c["Fi"],
                        "Fri": f"{c['Fri']:.2%}",
                    }
                    for c in classes
                ]
            ),
            hide_index=True,
            height=320,
        )
        st.caption(
            f"Amplitude de cada classe: h = {meta_classes['amplitude_classe']:.4f}. "
            "A última classe é fechada nos dois extremos, para acomodar o máximo."
        )

    st.divider()
    st.markdown("#### Valores atípicos pela regra do IQR")

    st.latex(
        r"\text{cerca inferior} = Q_1 - 1{,}5 \cdot IQR"
        r"\qquad"
        r"\text{cerca superior} = Q_3 + 1{,}5 \cdot IQR"
    )

    # Bigodes: a observação mais extrema que ainda está dentro das cercas.
    dentro = [
        v for v in valores
        if atipicos["limite_inferior"] <= v <= atipicos["limite_superior"]
    ]
    extremos = (ms.minimo(dentro), ms.maximo(dentro)) if dentro else (
        resumo["minimo"], resumo["maximo"]
    )

    colunas = st.columns(4)
    colunas[0].metric("Cerca inferior", _formatar(atipicos["limite_inferior"], 2))
    colunas[1].metric("Cerca superior", _formatar(atipicos["limite_superior"], 2))
    colunas[2].metric("Atípicos", mod_dados.formatar_inteiro(atipicos["quantidade"]))
    colunas[3].metric("Proporção", f"{atipicos['proporcao']:.2%}")

    st.pyplot(
        graficos.boxplot(
            resumo,
            (atipicos["limite_inferior"], atipicos["limite_superior"]),
            extremos,
            atipicos["valores"],
            rotulo,
        ),
    )

    st.divider()
    st.markdown("#### Interpretação automática")
    for paragrafo in _interpretar_numerica(resumo, atipicos, meta):
        st.markdown(f"- {paragrafo}")


def _renderizar_categorica(quadro, chave):
    meta = mod_dados.VARIAVEIS_CATEGORICAS[chave]
    coluna = mod_dados.coluna_categorica(chave)
    valores = quadro[coluna].tolist()

    st.caption(meta["descricao"])

    linhas, meta_freq = fr.tabela_frequencias_categorica(valores)

    colunas = st.columns(3)
    colunas[0].metric("n", mod_dados.formatar_inteiro(meta_freq["n"]))
    colunas[1].metric("Categorias", meta_freq["n_categorias"])
    colunas[2].metric(
        "Categoria modal",
        linhas[0]["categoria"],
        delta=f"{linhas[0]['fri']:.1%} do total",
        delta_color="off",
    )

    st.divider()

    esquerda, direita = st.columns([3, 2])
    with esquerda:
        st.pyplot(
            graficos.barras_categoricas(linhas, meta["rotulo"]),
        )
    with direita:
        st.markdown("**Tabela de frequências**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Categoria": linha["categoria"],
                        "fi": linha["fi"],
                        "fri": f"{linha['fri']:.2%}",
                        "Fi": linha["Fi"],
                        "Fri": f"{linha['Fri']:.2%}",
                    }
                    for linha in linhas
                ]
            ),
            hide_index=True,
        )

    if meta_freq["n_categorias"] <= 6:
        st.markdown("#### Composição relativa")
        centro = st.columns([1, 2, 1])[1]
        with centro:
            st.pyplot(
                graficos.pizza_categorica(linhas, meta["rotulo"]),
            )
        st.caption(
            "O gráfico de setores só é usado aqui porque há poucas categorias. "
            "Com muitas fatias ele deixa de ser legível — comparar ângulos é "
            "bem mais difícil do que comparar comprimentos de barra."
        )

    st.divider()
    st.markdown("#### Interpretação automática")

    dominante = linhas[0]
    menos_frequente = linhas[-1]
    razao = dominante["fi"] / menos_frequente["fi"] if menos_frequente["fi"] else float("inf")

    st.markdown(
        f"- A categoria mais frequente é **{dominante['categoria']}**, com "
        f"{mod_dados.formatar_inteiro(dominante['fi'])} ocorrências "
        f"(**{dominante['fri']:.2%}** do total)."
    )
    st.markdown(
        f"- A menos frequente é **{menos_frequente['categoria']}**, com "
        f"{mod_dados.formatar_inteiro(menos_frequente['fi'])} ocorrências "
        f"({menos_frequente['fri']:.2%}). A categoria dominante aparece "
        f"**{razao:.1f}× mais** que ela."
    )

    if meta_freq["n_categorias"] > 1:
        frequencia_uniforme = 1.0 / meta_freq["n_categorias"]
        desvio_maximo = max(abs(linha["fri"] - frequencia_uniforme) for linha in linhas)
        if desvio_maximo < 0.02:
            st.markdown(
                "- As categorias estão **praticamente equilibradas**: nenhuma se "
                f"afasta mais de 2 pontos percentuais da divisão uniforme "
                f"({frequencia_uniforme:.1%} cada)."
            )
        else:
            st.markdown(
                "- A distribuição é **desequilibrada**: a maior diferença em "
                f"relação a uma divisão uniforme ({frequencia_uniforme:.1%} por "
                f"categoria) é de {desvio_maximo:.1%}."
            )

    st.markdown(
        "- Para variáveis qualitativas **não faz sentido calcular média, "
        "mediana ou desvio padrão** — as categorias não têm distância "
        "numérica entre si. As medidas cabíveis são frequência, proporção e moda."
    )


def renderizar(quadro, descricao_filtro):
    st.title("Estatística descritiva")
    st.caption(
        f"Recorte: {descricao_filtro} · "
        f"{mod_dados.formatar_inteiro(len(quadro))} registros"
    )

    tipo = st.radio(
        "Tipo de variável",
        ["Numérica", "Categórica"],
        horizontal=True,
    )

    if tipo == "Numérica":
        chave = st.selectbox(
            "Variável",
            options=list(mod_dados.VARIAVEIS_NUMERICAS.keys()),
            format_func=lambda c: mod_dados.VARIAVEIS_NUMERICAS[c]["rotulo"],
        )
        st.divider()
        _renderizar_numerica(quadro, chave)
    else:
        chave = st.selectbox(
            "Variável",
            options=list(mod_dados.VARIAVEIS_CATEGORICAS.keys()),
            format_func=lambda c: mod_dados.VARIAVEIS_CATEGORICAS[c]["rotulo"],
        )
        st.divider()
        _renderizar_categorica(quadro, chave)
