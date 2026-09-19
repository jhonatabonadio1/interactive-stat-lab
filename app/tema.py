"""
Identidade visual dos gráficos.

Paleta e regras de marcação centralizadas aqui para que todos os módulos
produzam gráficos consistentes. As três cores de série foram verificadas
quanto à separação para daltonismo (deuteranopia/protanopia/tritanopia):
o pior par tem ΔE 9,2 em OKLab×100, acima do alvo de 8, e ΔE 24,0 para
visão normal, acima do piso de 15.

O verde (#1baf7a) fica abaixo de 3:1 de contraste sobre fundo branco, e
por isso nunca é usado sozinho para transmitir informação: toda tela que
o emprega traz também a tabela de dados correspondente.
"""

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

COR_SERIE_1 = "#2a78d6"  # azul    — dados observados
COR_SERIE_2 = "#eb6834"  # laranja — curva teórica / reta ajustada
COR_SERIE_3 = "#1baf7a"  # verde   — terceira série, sempre com tabela junto

COR_SUPERFICIE = "#ffffff"
COR_TEXTO = "#0b0b0b"
COR_TEXTO_SECUNDARIO = "#52514e"
COR_GRADE = "#e6e5e1"
COR_REFERENCIA = "#8a8984"  # linhas de referência: recessivas, nunca de série

TAMANHO_PADRAO = (8.0, 4.2)


def aplicar_estilo():
    """Configura o matplotlib. Chamada uma vez, no início da aplicação."""
    plt.rcParams.update(
        {
            "figure.facecolor": COR_SUPERFICIE,
            "axes.facecolor": COR_SUPERFICIE,
            "axes.edgecolor": COR_GRADE,
            "axes.labelcolor": COR_TEXTO_SECUNDARIO,
            "axes.titlecolor": COR_TEXTO,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": COR_GRADE,
            "grid.linewidth": 0.8,
            "xtick.color": COR_TEXTO_SECUNDARIO,
            "ytick.color": COR_TEXTO_SECUNDARIO,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 9.5,
            "figure.dpi": 130,
        }
    )


def criar_figura(largura=None, altura=None):
    """Cria figura e eixo já com as bordas superiores/direitas removidas."""
    largura = largura or TAMANHO_PADRAO[0]
    altura = altura or TAMANHO_PADRAO[1]

    figura, eixo = plt.subplots(figsize=(largura, altura))
    eixo.spines["top"].set_visible(False)
    eixo.spines["right"].set_visible(False)
    return figura, eixo


def finalizar(figura):
    """Ajusta as margens. Chamar antes de entregar a figura ao Streamlit."""
    figura.tight_layout()
    return figura
