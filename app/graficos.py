"""
Desenho dos gráficos.

Ponto importante para a avaliação do trabalho: nenhuma função aqui
CALCULA estatística. O histograma é desenhado a partir da tabela de
classes produzida por `core.frequencias`, e o boxplot a partir dos
quartis produzidos por `core.minhastats` — em vez de `plt.hist` e
`plt.boxplot`, que fariam os próprios cálculos por dentro e esconderiam
a matemática que o trabalho pede.
"""

from matplotlib.patches import Rectangle

from app.tema import (
    COR_GRADE,
    COR_REFERENCIA,
    COR_SERIE_1,
    COR_SERIE_2,
    COR_SUPERFICIE,
    COR_TEXTO_SECUNDARIO,
    criar_figura,
    finalizar,
)


# ---------------------------------------------------------------------------
# Variáveis contínuas
# ---------------------------------------------------------------------------

def histograma(classes, rotulo_x, densidade=False, titulo=None):
    """Histograma desenhado a partir da nossa tabela de classes.

    Com `densidade=True` a altura de cada barra é fri/h (frequência
    relativa dividida pela largura da classe), de modo que a área total
    das barras vale 1 — é assim que o histograma fica na mesma escala de
    uma função densidade e a sobreposição do Módulo 4 faz sentido.
    """
    figura, eixo = criar_figura()

    centros = [c["ponto_medio"] for c in classes]
    larguras = [c["limite_superior"] - c["limite_inferior"] for c in classes]

    if densidade:
        alturas = [
            c["fri"] / largura if largura > 0 else 0.0
            for c, largura in zip(classes, larguras)
        ]
        rotulo_y = "Densidade de frequência"
    else:
        alturas = [c["fi"] for c in classes]
        rotulo_y = "Frequência absoluta"

    eixo.bar(
        centros,
        alturas,
        width=[largura * 0.98 for largura in larguras],
        color=COR_SERIE_1,
        edgecolor=COR_SUPERFICIE,
        linewidth=0.8,
    )

    eixo.set_xlabel(rotulo_x)
    eixo.set_ylabel(rotulo_y)
    if titulo:
        eixo.set_title(titulo)
    eixo.grid(axis="x", visible=False)

    return finalizar(figura)


def histograma_com_curva(
    classes, curva_x, curva_y, rotulo_x, nome_curva, titulo=None
):
    """Histograma em densidade com uma curva teórica sobreposta."""
    figura, eixo = criar_figura()

    centros = [c["ponto_medio"] for c in classes]
    larguras = [c["limite_superior"] - c["limite_inferior"] for c in classes]
    alturas = [
        c["fri"] / largura if largura > 0 else 0.0
        for c, largura in zip(classes, larguras)
    ]

    eixo.bar(
        centros,
        alturas,
        width=[largura * 0.98 for largura in larguras],
        color=COR_SERIE_1,
        edgecolor=COR_SUPERFICIE,
        linewidth=0.8,
        label="Dados observados",
    )
    eixo.plot(
        curva_x,
        curva_y,
        color=COR_SERIE_2,
        linewidth=2.0,
        label=nome_curva,
    )

    eixo.set_xlabel(rotulo_x)
    eixo.set_ylabel("Densidade de frequência")
    if titulo:
        eixo.set_title(titulo)
    eixo.legend(loc="upper right")
    eixo.grid(axis="x", visible=False)

    return finalizar(figura)


def boxplot(resumo, limites, extremos_nao_atipicos, valores_atipicos, rotulo):
    """Boxplot construído a partir dos quartis que nós calculamos.

    A caixa vai de Q1 a Q3, a linha interna é a mediana, os bigodes
    alcançam a observação mais extrema que ainda está dentro das cercas
    de Tukey, e os pontos além das cercas são desenhados um a um.
    """
    figura, eixo = criar_figura(altura=2.6)

    q1, q2, q3 = resumo["q1"], resumo["q2"], resumo["q3"]
    inferior, superior = extremos_nao_atipicos

    altura_caixa = 0.4
    centro_y = 0.0

    # Bigodes.
    eixo.plot([inferior, q1], [centro_y, centro_y], color=COR_SERIE_1, linewidth=1.5)
    eixo.plot([q3, superior], [centro_y, centro_y], color=COR_SERIE_1, linewidth=1.5)
    for extremo in (inferior, superior):
        eixo.plot(
            [extremo, extremo],
            [centro_y - altura_caixa / 3, centro_y + altura_caixa / 3],
            color=COR_SERIE_1,
            linewidth=1.5,
        )

    # Caixa interquartil.
    eixo.add_patch(
        Rectangle(
            (q1, centro_y - altura_caixa / 2),
            q3 - q1,
            altura_caixa,
            facecolor=COR_SERIE_1,
            alpha=0.25,
            edgecolor=COR_SERIE_1,
            linewidth=1.5,
        )
    )

    # Mediana.
    eixo.plot(
        [q2, q2],
        [centro_y - altura_caixa / 2, centro_y + altura_caixa / 2],
        color=COR_SERIE_1,
        linewidth=2.5,
    )

    # Média, para contraste visual com a mediana.
    eixo.plot(
        [resumo["media"]],
        [centro_y],
        marker="D",
        markersize=7,
        color=COR_SERIE_2,
        linestyle="none",
        label="Média",
        zorder=5,
    )

    if valores_atipicos:
        eixo.plot(
            valores_atipicos,
            [centro_y] * len(valores_atipicos),
            marker="o",
            markersize=4,
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=COR_REFERENCIA,
            markeredgewidth=0.9,
            alpha=0.5,
            label=f"Atípicos ({len(valores_atipicos)})",
        )

    # Cercas de Tukey, recessivas.
    for limite in limites:
        eixo.axvline(limite, color=COR_GRADE, linewidth=1.0, linestyle="--", zorder=0)

    eixo.annotate(
        "Q1", xy=(q1, centro_y + altura_caixa / 2), xytext=(0, 6),
        textcoords="offset points", ha="center", fontsize=8,
        color=COR_TEXTO_SECUNDARIO,
    )
    eixo.annotate(
        "mediana", xy=(q2, centro_y + altura_caixa / 2), xytext=(0, 6),
        textcoords="offset points", ha="center", fontsize=8,
        color=COR_TEXTO_SECUNDARIO,
    )
    eixo.annotate(
        "Q3", xy=(q3, centro_y + altura_caixa / 2), xytext=(0, 6),
        textcoords="offset points", ha="center", fontsize=8,
        color=COR_TEXTO_SECUNDARIO,
    )

    eixo.set_xlabel(rotulo)
    eixo.set_yticks([])
    eixo.set_ylim(-0.6, 0.8)
    eixo.spines["left"].set_visible(False)
    eixo.grid(axis="y", visible=False)
    eixo.legend(loc="lower right", ncol=2)

    return finalizar(figura)


# ---------------------------------------------------------------------------
# Variáveis categóricas
# ---------------------------------------------------------------------------

def barras_categoricas(linhas, rotulo, titulo=None):
    """Barras horizontais, ordenadas por frequência, com rótulo direto."""
    figura, eixo = criar_figura(altura=max(2.4, 0.45 * len(linhas) + 1.2))

    categorias = [linha["categoria"] for linha in linhas]
    frequencias = [linha["fi"] for linha in linhas]
    posicoes = list(range(len(linhas) - 1, -1, -1))

    eixo.barh(
        posicoes,
        frequencias,
        height=0.68,
        color=COR_SERIE_1,
        edgecolor=COR_SUPERFICIE,
        linewidth=0.8,
    )

    limite = max(frequencias) if frequencias else 1
    for posicao, linha in zip(posicoes, linhas):
        eixo.annotate(
            f"{linha['fi']:,}".replace(",", ".") + f"  ({linha['fri']:.1%})",
            xy=(linha["fi"], posicao),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=COR_TEXTO_SECUNDARIO,
        )

    eixo.set_yticks(posicoes)
    eixo.set_yticklabels(categorias)
    eixo.set_xlabel("Frequência absoluta")
    eixo.set_xlim(0, limite * 1.28)
    if titulo:
        eixo.set_title(titulo)
    eixo.grid(axis="y", visible=False)

    return finalizar(figura)


def pizza_categorica(linhas, titulo=None):
    """Gráfico de setores — só faz sentido com poucas categorias."""
    figura, eixo = criar_figura(largura=5.0, altura=4.0)

    frequencias = [linha["fi"] for linha in linhas]
    rotulos = [linha["categoria"] for linha in linhas]

    # Um tom por categoria, do mesmo azul: a ordem já é a da frequência,
    # então a escala sequencial carrega a informação sem inventar matizes.
    tons = ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
    cores = [tons[i % len(tons)] for i in range(len(linhas))]

    eixo.pie(
        frequencias,
        labels=rotulos,
        autopct="%1.1f%%",
        colors=cores,
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": COR_SUPERFICIE, "linewidth": 1.5},
        textprops={"fontsize": 8.5},
    )
    eixo.set_aspect("equal")
    if titulo:
        eixo.set_title(titulo)
    eixo.grid(visible=False)
    eixo.set_xticks([])
    eixo.set_yticks([])
    for lado in eixo.spines.values():
        lado.set_visible(False)

    return finalizar(figura)


# ---------------------------------------------------------------------------
# Simulação
# ---------------------------------------------------------------------------

def convergencia(ns, frequencias, probabilidade_teorica, rotulo_evento):
    """Trajetória da frequência relativa contra a probabilidade teórica.

    Eixo x em escala logarítmica: a convergência acontece em ordens de
    grandeza, e num eixo linear os primeiros milhares de lançamentos —
    justamente onde está toda a ação — virariam uma faixa ilegível
    colada no zero.
    """
    figura, eixo = criar_figura()

    eixo.plot(ns, frequencias, color=COR_SERIE_1, linewidth=1.4, label="Frequência relativa observada")
    eixo.axhline(
        probabilidade_teorica,
        color=COR_SERIE_2,
        linewidth=2.0,
        linestyle="--",
        label=f"Probabilidade teórica = {probabilidade_teorica:.4f}",
    )

    eixo.set_xscale("log")
    eixo.set_xlabel("Número de repetições (escala logarítmica)")
    eixo.set_ylabel(f"Frequência relativa de “{rotulo_evento}”")
    eixo.legend(loc="upper right")

    return finalizar(figura)


def barras_faces(contagem_faces, n_lancamentos):
    """Distribuição empírica das faces do dado contra a frequência esperada."""
    figura, eixo = criar_figura(altura=3.2)

    faces = list(contagem_faces.keys())
    contagens = list(contagem_faces.values())
    esperado = n_lancamentos / len(faces)

    eixo.bar(
        faces,
        contagens,
        width=0.68,
        color=COR_SERIE_1,
        edgecolor=COR_SUPERFICIE,
        linewidth=0.8,
        label="Observado",
    )
    eixo.axhline(
        esperado,
        color=COR_SERIE_2,
        linewidth=2.0,
        linestyle="--",
        label=f"Esperado = {esperado:,.0f}".replace(",", "."),
    )

    eixo.set_xlabel("Face do dado")
    eixo.set_ylabel("Frequência absoluta")
    eixo.set_xticks(faces)
    eixo.legend(loc="lower right")
    eixo.grid(axis="x", visible=False)

    return finalizar(figura)


def comparacao_classes(classes, probabilidades_teoricas, rotulo_x, nome_modelo):
    """Frequência relativa observada contra a probabilidade de um modelo discreto.

    Para um modelo discreto com muitos valores possíveis, comparar barra
    a barra seria ilegível. Agrupamos nas mesmas classes do histograma e
    somamos a probabilidade do modelo dentro de cada uma — o que é a
    comparação correta e é também o que um teste de aderência faria.
    """
    figura, eixo = criar_figura()

    centros = [c["ponto_medio"] for c in classes]
    larguras = [c["limite_superior"] - c["limite_inferior"] for c in classes]
    observadas = [c["fri"] for c in classes]

    deslocamento = [largura * 0.22 for largura in larguras]

    eixo.bar(
        [centro - desloc for centro, desloc in zip(centros, deslocamento)],
        observadas,
        width=[largura * 0.42 for largura in larguras],
        color=COR_SERIE_1,
        edgecolor=COR_SUPERFICIE,
        linewidth=0.8,
        label="Observado",
    )
    eixo.bar(
        [centro + desloc for centro, desloc in zip(centros, deslocamento)],
        probabilidades_teoricas,
        width=[largura * 0.42 for largura in larguras],
        color=COR_SERIE_2,
        edgecolor=COR_SUPERFICIE,
        linewidth=0.8,
        label=nome_modelo,
    )

    eixo.set_xlabel(rotulo_x)
    eixo.set_ylabel("Probabilidade / frequência relativa")
    eixo.legend(loc="upper right")
    eixo.grid(axis="x", visible=False)

    return finalizar(figura)


# ---------------------------------------------------------------------------
# Regressão
# ---------------------------------------------------------------------------

def dispersao_com_reta(
    x, y, modelo, rotulo_x, rotulo_y, ponto_previsto=None, max_pontos=4000
):
    """Diagrama de dispersão com a reta de mínimos quadrados sobreposta.

    Com 17 mil pontos o gráfico vira uma mancha sólida, então amostramos
    sistematicamente (um a cada k) para plotagem. Isso afeta APENAS o
    desenho: a reta e o R² exibidos foram estimados com todos os pontos.
    """
    figura, eixo = criar_figura(altura=4.6)

    passo = max(1, len(x) // max_pontos)
    x_plot = x[::passo]
    y_plot = y[::passo]

    eixo.plot(
        x_plot,
        y_plot,
        marker="o",
        markersize=3,
        linestyle="none",
        color=COR_SERIE_1,
        alpha=0.18,
        markeredgewidth=0,
        label=f"Observações (n = {len(x):,})".replace(",", "."),
    )

    extremos = [modelo.x_minimo, modelo.x_maximo]
    eixo.plot(
        extremos,
        modelo.prever_varios(extremos),
        color=COR_SERIE_2,
        linewidth=2.2,
        label=f"Reta ajustada · R² = {modelo.r2:.4f}",
    )

    if ponto_previsto is not None:
        px, py = ponto_previsto
        eixo.plot(
            [px],
            [py],
            marker="o",
            markersize=10,
            color=COR_SERIE_2,
            markeredgecolor=COR_SUPERFICIE,
            markeredgewidth=2,
            linestyle="none",
            zorder=6,
            label=f"Predição: x = {px:g} → ŷ = {py:,.1f}".replace(",", "."),
        )

    eixo.set_xlabel(rotulo_x)
    eixo.set_ylabel(rotulo_y)
    eixo.legend(loc="upper left")

    if passo > 1:
        eixo.annotate(
            f"exibindo 1 a cada {passo} pontos; o ajuste usa todos",
            xy=(1.0, -0.16),
            xycoords="axes fraction",
            ha="right",
            fontsize=7.5,
            color=COR_TEXTO_SECUNDARIO,
        )

    return finalizar(figura)


def residuos(valores_ajustados, residuos_modelo, max_pontos=4000):
    """Resíduos contra valores ajustados — o diagnóstico padrão do ajuste."""
    figura, eixo = criar_figura(altura=3.4)

    passo = max(1, len(valores_ajustados) // max_pontos)

    eixo.plot(
        valores_ajustados[::passo],
        residuos_modelo[::passo],
        marker="o",
        markersize=3,
        linestyle="none",
        color=COR_SERIE_1,
        alpha=0.18,
        markeredgewidth=0,
    )
    eixo.axhline(0.0, color=COR_SERIE_2, linewidth=1.8)

    eixo.set_xlabel("Valor ajustado (ŷ)")
    eixo.set_ylabel("Resíduo (y − ŷ)")

    return finalizar(figura)
