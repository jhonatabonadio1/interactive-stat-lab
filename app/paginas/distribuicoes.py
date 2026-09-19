"""Módulo 4 — ajuste de distribuições teóricas a variáveis reais."""

import math

import pandas as pd
import streamlit as st

from app import dados as mod_dados
from app import graficos
from core import distribuicoes as dst
from core import frequencias as fr
from core import minhastats as ms


def _candidatas(valores, meta):
    """Quais famílias fazem sentido para esta variável.

    Filtramos pelo suporte da distribuição, não por gosto: oferecer uma
    Exponencial para uma variável que assume valores negativos produziria
    um ajuste sem sentido — a densidade é zero em todo o lado esquerdo.
    """
    minimo = ms.minimo(valores)
    familias = ["Normal", "Uniforme"]

    if minimo >= 0:
        familias.append("Exponencial")

    if meta["tipo"] == "discreta" and minimo >= 0:
        todos_inteiros = all(float(v).is_integer() for v in valores)
        if todos_inteiros:
            familias.append("Poisson")

    return familias


def _probabilidades_esperadas_continua(classes, familia, parametros):
    """P(classe) = F(limite superior) − F(limite inferior), pela nossa CDF."""
    acumuladas = {
        "Normal": lambda x: dst.cdf_normal(x, parametros["mu"], parametros["sigma"]),
        "Uniforme": lambda x: dst.cdf_uniforme(x, parametros["a"], parametros["b"]),
        "Exponencial": lambda x: dst.cdf_exponencial(x, parametros["lambda"]),
    }
    acumulada = acumuladas[familia]

    return [
        acumulada(c["limite_superior"]) - acumulada(c["limite_inferior"])
        for c in classes
    ]


def _probabilidades_esperadas_poisson(classes, parametros):
    """P(classe) = Σ P(X = k) para os inteiros k dentro da classe."""
    lam = parametros["lambda"]
    esperadas = []

    for classe in classes:
        inicio = math.ceil(classe["limite_inferior"])
        fim = math.ceil(classe["limite_superior"])
        # A última classe é fechada à direita e precisa incluir o próprio limite.
        if classe is classes[-1]:
            fim = math.floor(classe["limite_superior"]) + 1

        total = 0.0
        for k in range(max(inicio, 0), fim):
            total += dst.pmf_poisson(k, lam)
        esperadas.append(total)

    return esperadas


def _distancia_variacao_total(observadas, esperadas):
    """Distância de variação total entre a distribuição empírica e o modelo.

        d_VT = ½ · Σ | f_obs(classe) − P_modelo(classe) |

    Vale 0 num ajuste perfeito e 1 no pior caso possível, e tem leitura
    direta: é a maior diferença de probabilidade que o modelo comete em
    qualquer evento. Usamos essa medida, e não o qui-quadrado, porque ela
    não depende do tamanho da amostra — com n = 17.379 o qui-quadrado
    rejeitaria qualquer modelo, inclusive um bom.
    """
    total = 0.0
    for observada, esperada in zip(observadas, esperadas):
        total += abs(observada - esperada)
    return total / 2.0


def _avaliar(distancia):
    if distancia < 0.05:
        return "ajuste **muito bom**"
    if distancia < 0.10:
        return "ajuste **razoável**"
    if distancia < 0.20:
        return "ajuste **fraco**"
    return "ajuste **ruim**"


def renderizar(quadro, descricao_filtro):
    st.title("Distribuições teóricas")
    st.caption(
        f"Recorte: {descricao_filtro} · "
        f"{mod_dados.formatar_inteiro(len(quadro))} registros"
    )

    st.markdown(
        "Estimamos os parâmetros de uma distribuição teórica **a partir dos "
        "próprios dados** e sobrepomos a curva resultante ao histograma. "
        "Nenhum parâmetro é ajustado para agradar ao gráfico: todos saem de "
        "fórmulas fechadas aplicadas à amostra."
    )

    controles = st.columns([2, 2, 1])

    chave = controles[0].selectbox(
        "Variável",
        options=list(mod_dados.VARIAVEIS_NUMERICAS.keys()),
        format_func=lambda c: mod_dados.VARIAVEIS_NUMERICAS[c]["rotulo"],
    )

    meta = mod_dados.VARIAVEIS_NUMERICAS[chave]
    valores = mod_dados.serie_numerica(quadro, chave)
    rotulo = mod_dados.rotulo_completo(chave)

    familias = _candidatas(valores, meta)
    familia = controles[1].selectbox("Distribuição candidata", familias)
    n_classes = controles[2].slider("Classes", 10, 60, 30)

    st.divider()

    classes, _ = fr.tabela_frequencias_continua(valores, n_classes=n_classes)
    observadas = [c["fri"] for c in classes]

    if familia == "Poisson":
        parametros = dst.estimar_poisson(valores)
        esperadas = _probabilidades_esperadas_poisson(classes, parametros)
        descricao_parametros = dst.DISTRIBUICOES_DISCRETAS["Poisson"]["parametros"](
            parametros
        )
        formula = dst.DISTRIBUICOES_DISCRETAS["Poisson"]["formula"]
    else:
        definicao = dst.DISTRIBUICOES_CONTINUAS[familia]
        parametros = definicao["estimar"](valores)
        esperadas = _probabilidades_esperadas_continua(classes, familia, parametros)
        descricao_parametros = definicao["parametros"](parametros)
        formula = definicao["formula"]

    distancia = _distancia_variacao_total(observadas, esperadas)

    colunas = st.columns(3)
    colunas[0].metric("Distribuição", familia)
    colunas[1].metric("Distância de variação total", f"{distancia:.4f}")
    colunas[2].metric("Classes comparadas", len(classes))

    st.markdown(f"**Função:** `{formula}`")
    st.markdown(f"**Parâmetros estimados dos dados:** {descricao_parametros}")

    st.divider()

    if familia == "Poisson":
        st.pyplot(
            graficos.comparacao_classes(
                classes, esperadas, rotulo, "Poisson estimada"
            ),
        )
        st.caption(
            "Para um modelo discreto com centenas de valores possíveis, "
            "comparar barra a barra seria ilegível. Agrupamos nas mesmas "
            "classes do histograma e somamos P(X = k) dentro de cada uma."
        )
    else:
        menor = ms.minimo(valores)
        maior = ms.maximo(valores)
        passos = 400
        largura = (maior - menor) / passos
        curva_x = [menor + i * largura for i in range(passos + 1)]

        definicao = dst.DISTRIBUICOES_CONTINUAS[familia]
        curva_y = [definicao["densidade"](x, parametros) for x in curva_x]

        st.pyplot(
            graficos.histograma_com_curva(
                classes, curva_x, curva_y, rotulo, f"{familia} estimada"
            ),
        )

    st.divider()
    st.markdown("#### Qualidade do ajuste")

    st.markdown(
        f"A distância de variação total é **{distancia:.4f}** — um "
        f"{_avaliar(distancia)}. Ela mede a maior discrepância de probabilidade "
        f"entre o modelo e os dados observados, e vale 0 num ajuste perfeito."
    )

    with st.expander("Comparar o ajuste de todas as candidatas"):
        comparacao = []
        for candidata in familias:
            try:
                if candidata == "Poisson":
                    par = dst.estimar_poisson(valores)
                    esp = _probabilidades_esperadas_poisson(classes, par)
                    texto = dst.DISTRIBUICOES_DISCRETAS["Poisson"]["parametros"](par)
                else:
                    definicao_candidata = dst.DISTRIBUICOES_CONTINUAS[candidata]
                    par = definicao_candidata["estimar"](valores)
                    esp = _probabilidades_esperadas_continua(classes, candidata, par)
                    texto = definicao_candidata["parametros"](par)
            except ValueError as erro:
                comparacao.append(
                    {"Distribuição": candidata, "Distância": None,
                     "Parâmetros": f"não ajustável: {erro}"}
                )
                continue

            comparacao.append(
                {
                    "Distribuição": candidata,
                    "Distância": round(_distancia_variacao_total(observadas, esp), 4),
                    "Parâmetros": texto,
                }
            )

        tabela = pd.DataFrame(comparacao).sort_values(
            "Distância", na_position="last"
        )
        st.dataframe(tabela, hide_index=True)
        st.caption("Menor distância = melhor ajuste.")

    st.divider()
    st.markdown("#### Leitura")

    if familia == "Poisson":
        indice = parametros["indice_dispersao"]
        st.markdown(
            f"""
- Numa Poisson genuína, **média e variância são iguais**, de modo que o índice
  de dispersão Var/λ vale 1. Aqui ele vale **{indice:.1f}**.
- Um índice tão acima de 1 é **sobredispersão**: os dados variam muito mais do
  que uma Poisson permitiria. O motivo é que λ **não é constante** — a demanda
  por bicicletas às 8h de uma terça-feira ensolarada não tem nada a ver com a
  demanda às 3h de uma madrugada chuvosa, e a Poisson simples supõe uma taxa única.
- Ou seja: a variável é uma contagem, mas **isso não basta** para que seja
  Poisson. É um resultado negativo informativo, e está registrado como tal no
  relatório.
            """
        )
    elif familia == "Normal":
        assimetria = ms.assimetria_pearson(valores)
        if abs(assimetria) < 0.15:
            leitura = (
                f"A assimetria de Pearson é **{assimetria:.4f}**, praticamente "
                f"nula, o que é coerente com o bom ajuste da Normal."
            )
        else:
            leitura = (
                f"A assimetria de Pearson é **{assimetria:.4f}**. Como a Normal "
                f"é simétrica por construção, ela não tem como reproduzir essa "
                f"característica — e é aí que o ajuste perde qualidade."
            )
        st.markdown(f"- {leitura}")
        st.markdown(
            "- A Normal foi ajustada por μ̂ = x̄ e σ̂ = s, que são os estimadores "
            "de máxima verossimilhança para essa família."
        )
    elif familia == "Exponencial":
        st.markdown(
            f"""
- A Exponencial foi ajustada por λ̂ = 1/x̄ = **{parametros['lambda']:.6f}**.
- Ela tem **moda em zero** e decai monotonicamente. Se o histograma da variável
  sobe antes de descer, a Exponencial não consegue acompanhar — o pico interno
  é incompatível com a forma da família, por melhor que seja o λ escolhido.
            """
        )
    else:
        st.markdown(
            f"""
- A Uniforme foi ajustada pelos extremos observados: â = **{parametros['a']:.4f}**
  e b̂ = **{parametros['b']:.4f}**.
- Ela atribui a **mesma** densidade a todo o intervalo. É um bom modelo só
  quando o histograma é visivelmente plano — caso da hora do dia, que o dataset
  cobre de forma quase completa, e mau modelo para qualquer variável com pico.
            """
        )
