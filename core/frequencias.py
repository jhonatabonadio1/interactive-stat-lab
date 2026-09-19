"""
Tabelas de frequência — suporte matemático do Módulo 2.

Continua valendo a regra de ouro: nada de `np.histogram`,
`pd.value_counts` ou `pd.cut`. O agrupamento em classes, a contagem e as
frequências acumuladas são todos construídos com laços explícitos.
"""

import math

from core.minhastats import _validar, maximo, minimo

__all__ = [
    "regra_de_sturges",
    "tabela_frequencias_continua",
    "tabela_frequencias_categorica",
]


def regra_de_sturges(n):
    """Número de classes sugerido pela regra de Sturges.

        k = ⌈ 1 + 3,322 · log₁₀(n) ⌉  =  ⌈ 1 + log₂(n) ⌉

    A ideia da regra é supor que as frequências das classes seguem os
    coeficientes binomiais de uma distribuição simétrica — o que a faz
    subestimar o número de classes em dados muito assimétricos, como a
    variável `cnt` deste dataset.

    Sobre a constante 3,322
    -----------------------
    Ela é o arredondamento de 1/log₁₀(2) = 3,321928..., e por ser
    ligeiramente MAIOR que o valor exato produz uma classe a mais quando
    n é potência exata de 2. Com n = 2, por exemplo, a versão arredondada
    dá 1 + 1,00002 → ⌈2,00002⌉ = 3, enquanto a fórmula exata dá
    1 + 1 = 2. Usamos `math.log2`, que é o valor exato, para que a regra
    fique estável e reproduza a definição original de Sturges.
    """
    if n < 1:
        raise ValueError(f"n precisa ser positivo, recebeu {n}")
    if n == 1:
        return 1
    return int(math.ceil(1.0 + math.log2(n)))


def tabela_frequencias_continua(dados, n_classes=None):
    """Agrupa uma variável contínua em classes de igual amplitude.

    Com k classes e amplitude total A = x_máx − x_mín, a amplitude de
    cada classe é

        h = A / k

    e a i-ésima classe cobre o intervalo

        [ x_mín + (i−1)·h ,  x_mín + i·h )

    fechado à esquerda e aberto à direita, exceto a última classe, que é
    fechada dos dois lados para acomodar o valor máximo.

    Para cada classe devolvemos:
        fi   frequência absoluta      — quantas observações caíram nela
        fri  frequência relativa      — fi / n
        Fi   frequência acumulada     — Σ fi até a classe atual
        Fri  freq. relativa acumulada — Fi / n
        ponto_medio                   — (li + ls) / 2

    Retorna (lista_de_classes, metadados).
    """
    valores = _validar(dados, minimo_exigido=1)
    n = len(valores)

    if n_classes is None:
        n_classes = regra_de_sturges(n)
    if n_classes < 1:
        raise ValueError(f"n_classes precisa ser ≥ 1, recebeu {n_classes}")

    x_min = minimo(valores)
    x_max = maximo(valores)
    amplitude_total = x_max - x_min

    # Variável constante: uma única classe degenerada, para não dividir por zero.
    if amplitude_total == 0:
        classe = {
            "indice": 1,
            "limite_inferior": x_min,
            "limite_superior": x_min,
            "ponto_medio": x_min,
            "fi": n,
            "fri": 1.0,
            "Fi": n,
            "Fri": 1.0,
        }
        return [classe], {
            "n": n,
            "n_classes": 1,
            "amplitude_classe": 0.0,
            "amplitude_total": 0.0,
            "minimo": x_min,
            "maximo": x_max,
        }

    amplitude_classe = amplitude_total / n_classes

    contagens = [0] * n_classes
    for x in valores:
        # Índice da classe por divisão inteira do desvio em relação ao mínimo.
        indice = int((x - x_min) / amplitude_classe)
        # O valor máximo cairia em `n_classes`, fora da lista: vai para a última.
        if indice >= n_classes:
            indice = n_classes - 1
        contagens[indice] += 1

    classes = []
    acumulada = 0
    for i in range(n_classes):
        limite_inferior = x_min + i * amplitude_classe
        limite_superior = x_min + (i + 1) * amplitude_classe
        acumulada += contagens[i]

        classes.append(
            {
                "indice": i + 1,
                "limite_inferior": limite_inferior,
                "limite_superior": limite_superior,
                "ponto_medio": (limite_inferior + limite_superior) / 2.0,
                "fi": contagens[i],
                "fri": contagens[i] / n,
                "Fi": acumulada,
                "Fri": acumulada / n,
            }
        )

    metadados = {
        "n": n,
        "n_classes": n_classes,
        "amplitude_classe": amplitude_classe,
        "amplitude_total": amplitude_total,
        "minimo": x_min,
        "maximo": x_max,
    }
    return classes, metadados


def tabela_frequencias_categorica(dados, ordenar_por_frequencia=True):
    """Conta as ocorrências de cada categoria de uma variável qualitativa.

        fi  = número de ocorrências da categoria
        fri = fi / n
        Fi, Fri seguem a ordem em que as categorias são apresentadas

    Em variável nominal a frequência acumulada não tem significado
    estatístico (não existe ordem natural entre as categorias), mas é
    devolvida mesmo assim porque é útil para leitura de Pareto quando
    ordenamos por frequência decrescente.
    """
    categorias = list(dados)
    n = len(categorias)
    if n == 0:
        raise ValueError("dados categóricos vazios")

    contagens = {}
    for categoria in categorias:
        chave = str(categoria)
        contagens[chave] = contagens.get(chave, 0) + 1

    itens = list(contagens.items())
    if ordenar_por_frequencia:
        # Frequência decrescente; empate desfeito pelo nome, para a saída ser determinística.
        itens.sort(key=lambda par: (-par[1], par[0]))
    else:
        itens.sort(key=lambda par: par[0])

    linhas = []
    acumulada = 0
    for categoria, fi in itens:
        acumulada += fi
        linhas.append(
            {
                "categoria": categoria,
                "fi": fi,
                "fri": fi / n,
                "Fi": acumulada,
                "Fri": acumulada / n,
            }
        )

    metadados = {"n": n, "n_categorias": len(linhas)}
    return linhas, metadados
