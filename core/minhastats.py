"""
Núcleo estatístico próprio — Módulo 1 do Laboratório Estatístico Interativo.

REGRA DE OURO DO PROJETO
------------------------
Nenhuma medida estatística deste arquivo pode usar funções prontas de
NumPy, SciPy ou do módulo `statistics`. Tudo é calculado com somatórios
e laços explícitos. As únicas dependências permitidas aqui são `math`
(funções matemáticas elementares como raiz quadrada e logaritmo) e a
biblioteca padrão do Python.

NumPy/SciPy aparecem SOMENTE em `tests/`, como referência de validação.

Notação usada nas docstrings
----------------------------
    n       tamanho do conjunto de dados
    x_i     i-ésima observação
    x̄       média aritmética de x
    Σ       somatório de i = 1 até n
"""

import math

__all__ = [
    "soma",
    "minimo",
    "maximo",
    "media",
    "mediana",
    "moda",
    "amplitude",
    "variancia",
    "desvio_padrao",
    "percentil",
    "quartis",
    "amplitude_interquartil",
    "coeficiente_variacao",
    "assimetria_pearson",
    "limites_outliers_iqr",
    "outliers_iqr",
    "covariancia",
    "correlacao_pearson",
    "resumo_descritivo",
]


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------

def _validar(dados, minimo_exigido=1, nome="dados"):
    """Converte a entrada em lista de floats e valida o tamanho mínimo.

    Levanta ValueError se o conjunto for pequeno demais para a medida
    pedida, e TypeError se algum elemento não for numérico.
    """
    valores = []
    for item in dados:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{nome} contém valor não numérico: {item!r}")
        if isinstance(item, float) and math.isnan(item):
            raise ValueError(f"{nome} contém NaN; limpe os dados antes de calcular")
        valores.append(float(item))

    if len(valores) < minimo_exigido:
        raise ValueError(
            f"{nome} precisa de pelo menos {minimo_exigido} observação(ões), "
            f"recebeu {len(valores)}"
        )
    return valores


def _validar_pareado(x, y, minimo_exigido=2):
    """Valida dois conjuntos que precisam ter o mesmo tamanho (dados bivariados)."""
    vx = _validar(x, minimo_exigido, nome="x")
    vy = _validar(y, minimo_exigido, nome="y")
    if len(vx) != len(vy):
        raise ValueError(
            f"x e y precisam ter o mesmo tamanho: {len(vx)} != {len(vy)}"
        )
    return vx, vy


def _ordenar(valores):
    """Ordenação crescente por merge sort.

    Poderíamos usar `sorted()`, mas como mediana e quartis dependem
    inteiramente da ordenação, deixamos o algoritmo explícito para que a
    construção dessas medidas fique inteiramente visível no código.
    Complexidade: O(n log n).
    """
    if len(valores) <= 1:
        return list(valores)

    meio = len(valores) // 2
    esquerda = _ordenar(valores[:meio])
    direita = _ordenar(valores[meio:])

    resultado = []
    i = j = 0
    while i < len(esquerda) and j < len(direita):
        if esquerda[i] <= direita[j]:
            resultado.append(esquerda[i])
            i += 1
        else:
            resultado.append(direita[j])
            j += 1
    resultado.extend(esquerda[i:])
    resultado.extend(direita[j:])
    return resultado


# ---------------------------------------------------------------------------
# Agregações elementares
# ---------------------------------------------------------------------------

def soma(dados):
    """Somatório simples.

        S = Σ x_i
    """
    valores = _validar(dados)
    total = 0.0
    for x in valores:
        total += x
    return total


def minimo(dados):
    """Menor valor do conjunto, por varredura linear."""
    valores = _validar(dados)
    menor = valores[0]
    for x in valores[1:]:
        if x < menor:
            menor = x
    return menor


def maximo(dados):
    """Maior valor do conjunto, por varredura linear."""
    valores = _validar(dados)
    maior = valores[0]
    for x in valores[1:]:
        if x > maior:
            maior = x
    return maior


# ---------------------------------------------------------------------------
# Medidas de tendência central
# ---------------------------------------------------------------------------

def media(dados):
    """Média aritmética.

        x̄ = (1/n) · Σ x_i

    É o centro de massa da distribuição: sensível a valores extremos,
    porque cada observação entra na soma com o seu valor cheio.
    """
    valores = _validar(dados)
    total = 0.0
    for x in valores:
        total += x
    return total / len(valores)


def mediana(dados):
    """Mediana — o valor que parte o conjunto ordenado em duas metades.

    Seja x_(1) ≤ x_(2) ≤ ... ≤ x_(n) o conjunto ordenado:

        n ímpar:  Md = x_((n+1)/2)
        n par:    Md = [ x_(n/2) + x_(n/2 + 1) ] / 2

    Diferente da média, é resistente a outliers: mover o maior valor para
    o infinito não altera a mediana.
    """
    valores = _ordenar(_validar(dados))
    n = len(valores)

    if n % 2 == 1:
        return valores[(n + 1) // 2 - 1]

    inferior = valores[n // 2 - 1]
    superior = valores[n // 2]
    return (inferior + superior) / 2.0


def moda(dados):
    """Moda — o(s) valor(es) de maior frequência absoluta.

        Mo = { x : f(x) = max f }

    Retorna sempre uma LISTA ordenada, porque um conjunto pode ser
    bimodal ou multimodal. Se todos os valores aparecem exatamente uma
    vez, o conjunto é amodal e a lista retornada é vazia.
    """
    valores = _validar(dados)

    frequencias = {}
    for x in valores:
        frequencias[x] = frequencias.get(x, 0) + 1

    frequencia_maxima = 0
    for contagem in frequencias.values():
        if contagem > frequencia_maxima:
            frequencia_maxima = contagem

    if frequencia_maxima == 1:
        return []

    modas = [valor for valor, c in frequencias.items() if c == frequencia_maxima]
    return _ordenar(modas)


# ---------------------------------------------------------------------------
# Medidas de dispersão
# ---------------------------------------------------------------------------

def amplitude(dados):
    """Amplitude total.

        A = x_máx − x_mín
    """
    return maximo(dados) - minimo(dados)


def variancia(dados, amostral=True):
    """Variância — dispersão média quadrática em torno da média.

        populacional:  σ² = (1/n)     · Σ (x_i − x̄)²
        amostral:      s² = (1/(n−1)) · Σ (x_i − x̄)²

    O divisor (n−1) na versão amostral é a correção de Bessel: ao usar x̄
    (estimada dos próprios dados) no lugar de μ, os desvios ficam
    sistematicamente pequenos demais, e dividir por n−1 corrige esse viés.

    `amostral=True`  equivale a np.var(..., ddof=1)
    `amostral=False` equivale a np.var(..., ddof=0)
    """
    minimo_exigido = 2 if amostral else 1
    valores = _validar(dados, minimo_exigido)
    n = len(valores)

    x_barra = media(valores)

    soma_quadrados = 0.0
    for x in valores:
        desvio = x - x_barra
        soma_quadrados += desvio * desvio

    divisor = n - 1 if amostral else n
    return soma_quadrados / divisor


def desvio_padrao(dados, amostral=True):
    """Desvio padrão — raiz quadrada da variância.

        σ = √σ²        s = √s²

    Volta à unidade original da variável, o que o torna interpretável
    (a variância está em "unidade ao quadrado").
    """
    return math.sqrt(variancia(dados, amostral=amostral))


def coeficiente_variacao(dados, amostral=True, em_percentual=True):
    """Coeficiente de variação — dispersão relativa à média.

        CV = s / x̄          (ou ×100 para percentual)

    Adimensional, o que permite comparar a dispersão de variáveis em
    unidades diferentes. Indefinido quando x̄ = 0, pois a média é o
    denominador.
    """
    x_barra = media(dados)
    if x_barra == 0:
        raise ValueError("CV indefinido: a média dos dados é zero")

    cv = desvio_padrao(dados, amostral=amostral) / abs(x_barra)
    return cv * 100.0 if em_percentual else cv


# ---------------------------------------------------------------------------
# Medidas separatrizes
# ---------------------------------------------------------------------------

def percentil(dados, p):
    """Percentil de ordem p (0 ≤ p ≤ 100) por interpolação linear.

    Sobre o conjunto ordenado x_(1) ≤ ... ≤ x_(n), define-se a posição

        h = (n − 1) · p/100

    Se h é inteiro, P_p = x_(h+1). Caso contrário, interpola-se
    linearmente entre os dois vizinhos:

        P_p = x_(⌊h⌋+1) + (h − ⌊h⌋) · [ x_(⌈h⌉+1) − x_(⌊h⌋+1) ]

    Este é o mesmo método adotado por padrão em np.percentile
    (`method="linear"`), o que torna a comparação dos testes direta.
    """
    return _percentil_ordenado(_ordenar(_validar(dados)), p)


def _percentil_ordenado(valores, p):
    """Núcleo do cálculo de percentil, sobre dados JÁ ordenados.

    Separado de `percentil` para que `quartis` ordene uma única vez em
    vez de três — com 17 mil linhas do dataset isso é a diferença entre
    uma tela instantânea e uma tela travada.
    """
    if not isinstance(p, (int, float)) or isinstance(p, bool):
        raise TypeError(f"p precisa ser numérico, recebeu {p!r}")
    if p < 0 or p > 100:
        raise ValueError(f"p precisa estar entre 0 e 100, recebeu {p}")

    n = len(valores)
    if n == 1:
        return valores[0]

    h = (n - 1) * (p / 100.0)
    inferior = math.floor(h)
    superior = math.ceil(h)

    if inferior == superior:
        return valores[int(h)]

    fracao = h - inferior
    return valores[inferior] + fracao * (valores[superior] - valores[inferior])


def quartis(dados):
    """Os três quartis, que dividem o conjunto ordenado em quatro partes.

        Q1 = P_25      Q2 = P_50 = mediana      Q3 = P_75

    Retorna a tupla (Q1, Q2, Q3).
    """
    ordenados = _ordenar(_validar(dados))
    return (
        _percentil_ordenado(ordenados, 25),
        _percentil_ordenado(ordenados, 50),
        _percentil_ordenado(ordenados, 75),
    )


def amplitude_interquartil(dados):
    """Amplitude interquartil — dispersão dos 50% centrais.

        IQR = Q3 − Q1

    É a base da regra de detecção de outliers e, por ignorar as caudas,
    não é afetada por valores extremos.
    """
    q1, _, q3 = quartis(dados)
    return q3 - q1


# ---------------------------------------------------------------------------
# Forma e valores atípicos
# ---------------------------------------------------------------------------

def assimetria_pearson(dados):
    """Segundo coeficiente de assimetria de Pearson.

        As = 3 · (x̄ − Md) / s

    Leitura do sinal:
        As > 0 → cauda alongada à direita (média puxada para cima)
        As ≈ 0 → distribuição aproximadamente simétrica
        As < 0 → cauda alongada à esquerda

    Escolhemos esta fórmula, e não o momento padronizado de 3ª ordem,
    porque ela expressa exatamente a comparação média × mediana que a
    interpretação automática do Módulo 2 apresenta ao usuário.
    """
    s = desvio_padrao(dados, amostral=True)
    if s == 0:
        raise ValueError("assimetria indefinida: desvio padrão igual a zero")
    return 3.0 * (media(dados) - mediana(dados)) / s


def limites_outliers_iqr(dados, fator=1.5):
    """Cercas de Tukey para detecção de valores atípicos.

        limite inferior = Q1 − k · IQR
        limite superior = Q3 + k · IQR

    com k = 1,5 para outliers moderados (padrão do boxplot) e k = 3,0
    para outliers extremos. Retorna (limite_inferior, limite_superior).
    """
    if fator <= 0:
        raise ValueError(f"fator precisa ser positivo, recebeu {fator}")

    q1, _, q3 = quartis(dados)
    iqr = q3 - q1
    return (q1 - fator * iqr, q3 + fator * iqr)


def outliers_iqr(dados, fator=1.5):
    """Separa as observações em atípicas e típicas pela regra do IQR.

    Retorna um dicionário com os limites, a lista de outliers (com os
    índices originais preservados) e a contagem/proporção — tudo o que a
    interface precisa exibir no Módulo 2.
    """
    valores = _validar(dados)
    limite_inferior, limite_superior = limites_outliers_iqr(valores, fator=fator)

    indices = []
    atipicos = []
    for i, x in enumerate(valores):
        if x < limite_inferior or x > limite_superior:
            indices.append(i)
            atipicos.append(x)

    return {
        "limite_inferior": limite_inferior,
        "limite_superior": limite_superior,
        "fator": fator,
        "indices": indices,
        "valores": atipicos,
        "quantidade": len(atipicos),
        "proporcao": len(atipicos) / len(valores),
    }


# ---------------------------------------------------------------------------
# Medidas bivariadas
# ---------------------------------------------------------------------------

def covariancia(x, y, amostral=True):
    """Covariância — como duas variáveis variam em conjunto.

        populacional:  σ_xy = (1/n)     · Σ (x_i − x̄)(y_i − ȳ)
        amostral:      s_xy = (1/(n−1)) · Σ (x_i − x̄)(y_i − ȳ)

    O sinal indica a direção da associação linear, mas a magnitude
    depende das unidades de x e y — por isso normalizamos em
    `correlacao_pearson` para poder comparar pares diferentes.
    """
    minimo_exigido = 2 if amostral else 1
    vx, vy = _validar_pareado(x, y, minimo_exigido)
    n = len(vx)

    x_barra = media(vx)
    y_barra = media(vy)

    soma_produtos = 0.0
    for i in range(n):
        soma_produtos += (vx[i] - x_barra) * (vy[i] - y_barra)

    divisor = n - 1 if amostral else n
    return soma_produtos / divisor


def correlacao_pearson(x, y):
    """Coeficiente de correlação linear de Pearson.

        r = s_xy / (s_x · s_y)

    ou, de forma equivalente e sem calcular desvios padrão separados:

              Σ (x_i − x̄)(y_i − ȳ)
        r = ─────────────────────────────────
            √[ Σ (x_i − x̄)² · Σ (y_i − ȳ)² ]

    Implementamos a segunda forma porque ela usa uma única passagem de
    somatórios e evita a divisão intermediária por (n−1), que se cancela.

    r ∈ [−1, +1]: ±1 é alinhamento perfeito sobre uma reta e 0 indica
    ausência de associação LINEAR (pode haver relação não linear forte
    com r ≈ 0).
    """
    vx, vy = _validar_pareado(x, y, minimo_exigido=2)
    n = len(vx)

    x_barra = media(vx)
    y_barra = media(vy)

    soma_produtos = 0.0
    soma_quadrados_x = 0.0
    soma_quadrados_y = 0.0
    for i in range(n):
        dx = vx[i] - x_barra
        dy = vy[i] - y_barra
        soma_produtos += dx * dy
        soma_quadrados_x += dx * dx
        soma_quadrados_y += dy * dy

    denominador = math.sqrt(soma_quadrados_x * soma_quadrados_y)
    if denominador == 0:
        raise ValueError(
            "correlação indefinida: ao menos uma das variáveis é constante"
        )

    return soma_produtos / denominador


# ---------------------------------------------------------------------------
# Conveniência para a interface
# ---------------------------------------------------------------------------

def resumo_descritivo(dados, amostral=True):
    """Calcula de uma vez todas as medidas do Módulo 1.

    Existe para que a camada de interface faça uma única chamada em vez
    de doze, mantendo toda a matemática deste lado da fronteira.
    """
    valores = _validar(dados, minimo_exigido=2)
    q1, q2, q3 = quartis(valores)

    resumo = {
        "n": len(valores),
        "media": media(valores),
        "mediana": mediana(valores),
        "moda": moda(valores),
        "minimo": minimo(valores),
        "maximo": maximo(valores),
        "amplitude": amplitude(valores),
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "iqr": q3 - q1,
        "variancia_amostral": variancia(valores, amostral=True),
        "variancia_populacional": variancia(valores, amostral=False),
        "desvio_padrao_amostral": desvio_padrao(valores, amostral=True),
        "desvio_padrao_populacional": desvio_padrao(valores, amostral=False),
    }

    # CV e assimetria têm pré-condições (média ≠ 0, s ≠ 0) que um dado
    # real pode violar; nesses casos devolvemos None em vez de quebrar a tela.
    try:
        resumo["coeficiente_variacao"] = coeficiente_variacao(
            valores, amostral=amostral
        )
    except ValueError:
        resumo["coeficiente_variacao"] = None

    try:
        resumo["assimetria"] = assimetria_pearson(valores)
    except ValueError:
        resumo["assimetria"] = None

    return resumo
