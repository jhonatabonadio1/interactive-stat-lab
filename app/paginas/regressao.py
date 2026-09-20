"""Módulo 5 — correlação e regressão linear simples."""

import streamlit as st

from app import dados as mod_dados
from app import graficos
from core import minhastats as ms
from core.regressao import ajustar_regressao_linear


def _classificar_correlacao(r):
    magnitude = abs(r)
    if magnitude < 0.10:
        forca = "desprezível"
    elif magnitude < 0.30:
        forca = "fraca"
    elif magnitude < 0.50:
        forca = "moderada"
    elif magnitude < 0.70:
        forca = "forte"
    else:
        forca = "muito forte"

    direcao = "positiva" if r > 0 else "negativa"
    return forca, direcao


def renderizar(quadro, descricao_filtro):
    st.title("Correlação e regressão linear")
    st.caption(
        f"Recorte: {descricao_filtro} · "
        f"{mod_dados.formatar_inteiro(len(quadro))} registros"
    )

    chaves = list(mod_dados.VARIAVEIS_NUMERICAS.keys())

    controles = st.columns(2)
    chave_x = controles[0].selectbox(
        "Variável explicativa (X)",
        options=chaves,
        index=chaves.index("temp_c"),
        format_func=lambda c: mod_dados.VARIAVEIS_NUMERICAS[c]["rotulo"],
    )
    chave_y = controles[1].selectbox(
        "Variável resposta (Y)",
        options=chaves,
        index=chaves.index("cnt"),
        format_func=lambda c: mod_dados.VARIAVEIS_NUMERICAS[c]["rotulo"],
    )

    if chave_x == chave_y:
        st.warning(
            "Escolha duas variáveis diferentes. Regredir uma variável sobre ela "
            "mesma sempre dá R² = 1 e não informa nada."
        )
        return

    x = mod_dados.serie_numerica(quadro, chave_x)
    y = mod_dados.serie_numerica(quadro, chave_y)

    rotulo_x = mod_dados.rotulo_completo(chave_x)
    rotulo_y = mod_dados.rotulo_completo(chave_y)
    unidade_x = mod_dados.VARIAVEIS_NUMERICAS[chave_x]["unidade"]
    unidade_y = mod_dados.VARIAVEIS_NUMERICAS[chave_y]["unidade"]

    try:
        modelo = ajustar_regressao_linear(x, y)
    except ValueError as erro:
        st.error(f"Não foi possível ajustar a reta: {erro}")
        return

    covariancia = ms.covariancia(x, y)

    st.divider()
    st.markdown("#### Medidas de associação")

    colunas = st.columns(4)
    colunas[0].metric("Correlação de Pearson (r)", f"{modelo.r:.4f}")
    colunas[1].metric("Coeficiente de determinação (R²)", f"{modelo.r2:.4f}")
    colunas[2].metric("Covariância amostral", f"{covariancia:,.2f}".replace(",", "."))
    colunas[3].metric("Erro padrão da estimativa", f"{modelo.erro_padrao_estimativa:,.2f}".replace(",", "."))

    st.divider()
    st.markdown("#### Reta de mínimos quadrados")

    st.latex(
        r"b_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}"
        r"\qquad b_0 = \bar{y} - b_1 \bar{x}"
    )

    st.markdown(f"### `{modelo.equacao(casas=4, nome_x='X')}`")
    st.caption(
        f"onde X = {rotulo_x} e ŷ = {rotulo_y}. "
        f"Estimada com todas as {modelo.n:,} observações do recorte.".replace(",", ".")
    )

    st.pyplot(
        graficos.dispersao_com_reta(x, y, modelo, rotulo_x, rotulo_y),
    )

    st.divider()
    st.markdown("#### Predição interativa")

    esquerda, direita = st.columns([1, 2])

    with esquerda:
        valor_padrao = float((modelo.x_minimo + modelo.x_maximo) / 2)
        valor_x = st.number_input(
            f"Informe um valor de X — {rotulo_x}",
            value=round(valor_padrao, 2),
            step=round((modelo.x_maximo - modelo.x_minimo) / 20, 2) or 0.1,
            format="%.2f",
        )

        previsto = modelo.prever(valor_x)
        st.metric(
            f"Predição ŷ — {rotulo_y}",
            f"{previsto:,.2f}".replace(",", "."),
        )

        if modelo.extrapola(valor_x):
            st.warning(
                f"**Extrapolação.** O valor {valor_x:g} está fora do intervalo "
                f"observado de X, que vai de {modelo.x_minimo:.2f} a "
                f"{modelo.x_maximo:.2f}. A reta foi estimada só dentro desse "
                f"intervalo e nada garante que a relação continue linear fora dele."
            )

        if previsto < 0 and ms.minimo(y) >= 0:
            st.warning(
                "A predição ficou **negativa** para uma variável que nunca é "
                "negativa nos dados. Isso é uma limitação do modelo linear, que "
                "não conhece o limite inferior zero."
            )

    with direita:
        st.pyplot(
            graficos.dispersao_com_reta(
                x, y, modelo, rotulo_x, rotulo_y,
                ponto_previsto=(valor_x, previsto),
            ),
        )

    st.divider()
    st.markdown("#### Interpretação dos coeficientes")

    forca, direcao = _classificar_correlacao(modelo.r)
    sufixo_x = f" {unidade_x}" if unidade_x else ""
    sufixo_y = f" {unidade_y}" if unidade_y else ""

    nome_x = mod_dados.VARIAVEIS_NUMERICAS[chave_x]["rotulo"].lower()
    nome_y = mod_dados.VARIAVEIS_NUMERICAS[chave_y]["rotulo"].lower()

    # "a mais"/"a menos", e não "aumento"/"redução": a segunda forma sugere que
    # X provoca a mudança em Y, que é exatamente o que uma regressão NÃO mostra.
    sentido = "a mais" if modelo.b1 >= 0 else "a menos"
    efeito_texto = f"**{abs(modelo.b1):.4f}{sufixo_y} {sentido}**"

    if modelo.x_minimo <= 0 <= modelo.x_maximo:
        leitura_intercepto = (
            "Como X = 0 está dentro do intervalo observado, esse número tem "
            "leitura direta."
        )
    else:
        leitura_intercepto = (
            f"Cuidado: X = 0 está **fora** do intervalo observado "
            f"({modelo.x_minimo:.2f} a {modelo.x_maximo:.2f}), então o "
            f"intercepto é apenas onde a reta cruzaria o eixo, sem significado "
            f"empírico."
        )

    st.markdown(
        f"""
- **Inclinação (b₁ = {modelo.b1:.4f}).** Cada **1{sufixo_x}** a mais em
  {nome_x} está **associada, em média**, a {efeito_texto} em {nome_y}.
  O termo *associada* é deliberado: a reta descreve como as duas variáveis
  variam juntas nos dados observados, e não afirma que uma produz a outra.
- **Intercepto (b₀ = {modelo.b0:.4f}).** É o valor previsto quando X = 0.
  {leitura_intercepto}
- **Correlação (r = {modelo.r:.4f}).** Associação linear **{forca} {direcao}**.
- **R² = {modelo.r2:.4f}.** A reta explica **{modelo.r2:.2%}** da variação de
  {nome_y}; os **{1 - modelo.r2:.2%}** restantes ficam por conta de tudo o que
  este modelo ignora.
- **Erro padrão da estimativa = {modelo.erro_padrao_estimativa:.2f}{sufixo_y}.**
  É a dispersão típica dos pontos em torno da reta — a margem de erro que se
  deve esperar de uma predição.
        """
    )

    st.error(
        "**Correlação não implica causalidade.** Um r alto entre duas variáveis "
        "não autoriza afirmar que uma causa a outra. As três explicações "
        "alternativas continuam de pé: a relação pode ser **inversa** (Y causa X), "
        "pode haver uma **terceira variável** influenciando as duas, ou pode ser "
        "**coincidência**. No caso de temperatura e aluguéis, por exemplo, a "
        "estação do ano afeta simultaneamente a temperatura, a duração do dia, "
        "o calendário escolar e os hábitos de lazer — e o modelo não tem como "
        "separar esses efeitos."
    )

    st.divider()
    with st.expander("Diagnóstico dos resíduos"):
        st.markdown(
            "Se a relação for de fato linear e a dispersão constante, os "
            "resíduos devem se espalhar sem padrão em torno de zero. Um funil "
            "que abre indica **heterocedasticidade**; uma curva indica que a "
            "relação **não é linear**."
        )
        residuos_modelo = modelo.residuos(x, y)
        ajustados = modelo.prever_varios(x)
        st.pyplot(
            graficos.residuos(ajustados, residuos_modelo),
        )

        colunas_residuos = st.columns(3)
        colunas_residuos[0].metric("Soma dos resíduos", f"{sum(residuos_modelo):.2e}")
        colunas_residuos[1].metric("SQ total", f"{modelo.sq_total:,.0f}".replace(",", "."))
        colunas_residuos[2].metric("SQ residual", f"{modelo.sq_residual:,.0f}".replace(",", "."))
        st.caption(
            "A soma dos resíduos é nula por construção — é a primeira equação "
            "normal do método dos mínimos quadrados, e serve como conferência "
            "numérica do nosso ajuste."
        )
