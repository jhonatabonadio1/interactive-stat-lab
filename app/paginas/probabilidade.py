"""Módulo 3 — probabilidade e simulação de Monte Carlo."""

import math

import streamlit as st

from app import dados as mod_dados
from app import graficos
from core import distribuicoes as dst
from core import frequencias as fr
from core import minhastats as ms
from core import simulacao as sim


def _aba_lei_grandes_numeros():
    st.markdown("### Lei dos Grandes Números")
    st.markdown(
        "A frequência relativa de um evento converge para a sua probabilidade "
        "quando o número de repetições cresce. Aumente o número de lançamentos "
        "e observe a curva se assentar sobre a linha teórica."
    )
    st.latex(r"f_n(A) = \frac{\text{sucessos em } n \text{ ensaios}}{n} \xrightarrow[n \to \infty]{} P(A)")

    controles = st.columns([1, 1, 1, 1])

    experimento = controles[0].selectbox("Experimento", ["Moeda", "Dado"])
    n_lancamentos = controles[1].select_slider(
        "Número de lançamentos",
        options=[100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000],
        value=10_000,
    )
    semente = controles[3].number_input(
        "Semente aleatória",
        min_value=0,
        max_value=999_999,
        value=42,
        help=(
            "Fixar a semente torna a simulação reprodutível: a mesma semente "
            "produz exatamente os mesmos lançamentos. Mude-a para ver outra "
            "realização do mesmo experimento."
        ),
    )

    if experimento == "Moeda":
        p_cara = controles[2].slider(
            "P(cara)", min_value=0.05, max_value=0.95, value=0.50, step=0.05
        )
        with st.spinner("Simulando…"):
            resultado = sim.simular_lancamentos_moeda(
                n_lancamentos, p_cara=p_cara, semente=int(semente)
            )
    else:
        n_faces = controles[2].selectbox("Número de faces", [6, 10, 12, 20], index=0)
        face_alvo = 1
        with st.spinner("Simulando…"):
            resultado = sim.simular_lancamentos_dado(
                n_lancamentos,
                face_alvo=face_alvo,
                n_faces=n_faces,
                semente=int(semente),
            )

    st.divider()

    probabilidade = resultado["probabilidade_teorica"]
    erro_padrao = math.sqrt(probabilidade * (1 - probabilidade) / resultado["n"])

    colunas = st.columns(4)
    colunas[0].metric("Probabilidade teórica", f"{probabilidade:.6f}")
    colunas[1].metric("Frequência observada", f"{resultado['frequencia_final']:.6f}")
    colunas[2].metric("Erro absoluto", f"{resultado['erro_absoluto']:.6f}")
    colunas[3].metric(
        "Erro em desvios padrão",
        f"{resultado['erro_absoluto'] / erro_padrao:.2f}σ",
        help=(
            "O desvio típico da frequência relativa é √(p(1−p)/n). Valores "
            "abaixo de 2σ são o esperado; acima de 3σ seriam surpreendentes."
        ),
    )

    ns, frequencias = sim.reduzir_trajetoria(resultado["trajetoria"])
    st.pyplot(
        graficos.convergencia(ns, frequencias, probabilidade, resultado["evento"]),
    )

    if experimento == "Dado":
        st.markdown("#### Distribuição empírica das faces")
        st.pyplot(
            graficos.barras_faces(resultado["contagem_faces"], resultado["n"]),
        )

    st.divider()
    st.markdown("#### O que observar")
    st.markdown(
        f"""
- Com **{mod_dados.formatar_inteiro(resultado['n'])} lançamentos**, a frequência
  relativa ficou a **{resultado['erro_absoluto']:.6f}** da probabilidade teórica —
  cerca de **{resultado['erro_absoluto'] / erro_padrao:.2f} erros padrão**.
- Repare que a curva **não** se aproxima de forma suave e monótona: ela oscila
  bastante no começo e vai se acalmando. A Lei dos Grandes Números garante a
  convergência, **não** que cada passo melhore o anterior.
- A precisão cresce com **1/√n**, e não com 1/n. Para dobrar a precisão é
  preciso **quadruplicar** o número de lançamentos — motivo pelo qual o eixo
  horizontal está em escala logarítmica.
        """
    )


def _aba_teorema_central_limite(quadro):
    st.markdown("### Teorema Central do Limite")
    st.markdown(
        "Sorteamos repetidas amostras de uma variável **real** do dataset e "
        "observamos a distribuição das médias amostrais. O teorema afirma que "
        "essa distribuição tende à Normal conforme o tamanho da amostra cresce — "
        "**qualquer que seja o formato da população de origem**."
    )
    st.latex(r"\bar{X} \;\xrightarrow{\ \ n \to \infty\ }\; N\!\left(\mu,\ \frac{\sigma^2}{n}\right)")

    controles = st.columns([2, 1, 1, 1])

    chave = controles[0].selectbox(
        "Variável da população",
        options=list(mod_dados.VARIAVEIS_NUMERICAS.keys()),
        format_func=lambda c: mod_dados.VARIAVEIS_NUMERICAS[c]["rotulo"],
    )
    tamanho_amostra = controles[1].select_slider(
        "Tamanho de cada amostra (n)",
        options=[2, 5, 10, 30, 50, 100, 200, 500],
        value=30,
    )
    n_repeticoes = controles[2].select_slider(
        "Número de amostras sorteadas",
        options=[100, 500, 1_000, 2_000, 5_000],
        value=2_000,
    )
    semente = controles[3].number_input(
        "Semente aleatória", min_value=0, max_value=999_999, value=7
    )

    populacao = mod_dados.serie_numerica(quadro, chave)
    rotulo = mod_dados.rotulo_completo(chave)

    with st.spinner(f"Sorteando {mod_dados.formatar_inteiro(n_repeticoes)} amostras…"):
        resultado = sim.simular_medias_amostrais(
            populacao,
            tamanho_amostra=tamanho_amostra,
            n_repeticoes=n_repeticoes,
            semente=int(semente),
        )

    st.divider()

    colunas = st.columns(4)
    colunas[0].metric("μ da população", f"{resultado['mu_populacional']:.4f}")
    colunas[1].metric(
        "Média das médias",
        f"{resultado['media_das_medias']:.4f}",
        help="Deve ficar próxima de μ: X̄ é estimador não enviesado.",
    )
    colunas[2].metric(
        "Erro padrão teórico  σ/√n",
        f"{resultado['erro_padrao_teorico']:.4f}",
    )
    colunas[3].metric(
        "Erro padrão observado",
        f"{resultado['erro_padrao_observado']:.4f}",
        delta=f"razão = {resultado['razao_erro_padrao']:.4f}",
        delta_color="off",
        help="Quanto mais perto de 1 a razão, melhor o TCL descreve a simulação.",
    )

    assimetria_populacao = ms.assimetria_pearson(populacao)
    assimetria_medias = ms.assimetria_pearson(resultado["medias"])

    esquerda, direita = st.columns(2)

    with esquerda:
        st.markdown("**População de origem**")
        classes_populacao, _ = fr.tabela_frequencias_continua(populacao, n_classes=40)
        st.pyplot(
            graficos.histograma(
                classes_populacao,
                rotulo,
                titulo=f"Assimetria de Pearson = {assimetria_populacao:.4f}",
            ),
        )

    with direita:
        st.markdown(f"**Médias de {mod_dados.formatar_inteiro(n_repeticoes)} amostras de n = {tamanho_amostra}**")

        classes_medias, _ = fr.tabela_frequencias_continua(
            resultado["medias"], n_classes=40
        )

        # Normal teórica prevista pelo TCL: N(μ, σ²/n).
        mu = resultado["mu_populacional"]
        sigma = resultado["erro_padrao_teorico"]
        menor = ms.minimo(resultado["medias"])
        maior = ms.maximo(resultado["medias"])

        passos = 300
        largura = (maior - menor) / passos
        curva_x = [menor + i * largura for i in range(passos + 1)]
        curva_y = [dst.pdf_normal(x, mu, sigma) for x in curva_x]

        st.pyplot(
            graficos.histograma_com_curva(
                classes_medias,
                curva_x,
                curva_y,
                f"Média amostral de {rotulo}",
                f"N(μ, σ²/n) prevista pelo TCL",
                titulo=f"Assimetria de Pearson = {assimetria_medias:.4f}",
            ),
        )

    st.divider()
    st.markdown("#### O que observar")

    reducao = (
        abs(assimetria_populacao) / abs(assimetria_medias)
        if assimetria_medias != 0
        else float("inf")
    )

    st.markdown(
        f"""
- A população de **{rotulo}** tem assimetria de Pearson **{assimetria_populacao:.4f}**;
  a distribuição das médias tem **{assimetria_medias:.4f}** — uma redução de cerca de
  **{reducao:.1f}×**. A forma irregular da população praticamente desaparece nas médias.
- O erro padrão observado (**{resultado['erro_padrao_observado']:.4f}**) bate com o
  previsto por σ/√n (**{resultado['erro_padrao_teorico']:.4f}**): razão de
  **{resultado['razao_erro_padrao']:.4f}**. A curva laranja **não foi ajustada** ao
  histograma — ela é a Normal que a teoria prevê a partir de μ e σ da população.
- Experimente **n = 2**: com amostras minúsculas o TCL ainda não teve efeito e o
  histograma das médias herda a assimetria da população. Vá subindo para 30, 100,
  500 e veja o sino se formar.
        """
    )

    st.info(
        "**Por que isso importa na prática:** é o TCL que justifica usar a "
        "Normal em intervalos de confiança e testes de hipótese mesmo quando os "
        "dados originais não são normais — desde que se trabalhe com médias e a "
        "amostra seja grande o bastante."
    )


def renderizar(quadro, descricao_filtro):
    st.title("Probabilidade e simulação")
    st.caption(
        f"Recorte: {descricao_filtro} · "
        f"{mod_dados.formatar_inteiro(len(quadro))} registros"
    )

    aba_lgn, aba_tcl = st.tabs(
        ["Lei dos Grandes Números", "Teorema Central do Limite"]
    )

    with aba_lgn:
        _aba_lei_grandes_numeros()

    with aba_tcl:
        _aba_teorema_central_limite(quadro)
