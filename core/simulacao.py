"""
Probabilidade e simulação de Monte Carlo — Módulo 3.

Demonstra numericamente dois teoremas centrais:

    Lei dos Grandes Números  — a frequência relativa de um evento
                               converge para sua probabilidade.
    Teorema Central do Limite — a distribuição das médias amostrais
                               tende à Normal, seja qual for a forma da
                               população de origem.

Sobre a fonte de aleatoriedade
------------------------------
Usamos `random.Random` da biblioteca padrão. Isso NÃO fere a regra de
ouro do projeto: `random` é um gerador de números pseudoaleatórios
(sorteio), não uma biblioteca de estatística. Nenhuma medida calculada
sobre os resultados sorteados vem de biblioteca externa — média, desvio
padrão e frequências continuam saindo de `core/minhastats.py`.

Toda simulação aceita uma semente para ser reprodutível: rodar duas
vezes com a mesma semente produz exatamente os mesmos números, o que é
indispensável para que os valores citados no relatório possam ser
conferidos.
"""

import math
import random

from core.minhastats import (
    _validar,
    desvio_padrao,
    media,
)

__all__ = [
    "simular_lancamentos_moeda",
    "simular_lancamentos_dado",
    "erro_padrao_teorico",
    "simular_medias_amostrais",
    "reduzir_trajetoria",
]


# ---------------------------------------------------------------------------
# (a) Lei dos Grandes Números
# ---------------------------------------------------------------------------

def _trajetoria_frequencia_relativa(sucessos):
    """Frequência relativa acumulada após cada ensaio.

                     nº de sucessos nos primeiros n ensaios
        f_n(A)  =  ───────────────────────────────────────
                                    n

    A Lei Forte dos Grandes Números garante que f_n(A) → P(A) quase
    certamente quando n → ∞. O gráfico dessa trajetória é a demonstração
    visual do teorema: uma curva que oscila muito no início e vai se
    achatando sobre a reta y = P(A).
    """
    trajetoria = []
    acumulado = 0
    for i, sucesso in enumerate(sucessos, start=1):
        acumulado += sucesso
        trajetoria.append(acumulado / i)
    return trajetoria


def simular_lancamentos_moeda(n_lancamentos, p_cara=0.5, semente=None):
    """Simula n lançamentos de uma moeda e acompanha a convergência.

    Cada lançamento é um ensaio de Bernoulli: vale 1 (cara) com
    probabilidade p e 0 (coroa) com probabilidade 1 − p. A frequência
    relativa acumulada de caras deve convergir para p.

    O desvio esperado em torno de p decresce na ordem de 1/√n: pelo
    próprio TCL,

        f_n ≈ Normal( p , p(1−p)/n )

    ou seja, para quadruplicar a precisão é preciso multiplicar o número
    de lançamentos por 16 — o que a simulação deixa visível.
    """
    if n_lancamentos < 1:
        raise ValueError(f"n_lancamentos precisa ser ≥ 1, recebeu {n_lancamentos}")
    if not 0.0 <= p_cara <= 1.0:
        raise ValueError(f"p_cara precisa estar em [0, 1], recebeu {p_cara}")

    gerador = random.Random(semente)
    resultados = [1 if gerador.random() < p_cara else 0 for _ in range(n_lancamentos)]

    trajetoria = _trajetoria_frequencia_relativa(resultados)
    frequencia_final = trajetoria[-1]

    return {
        "tipo": "moeda",
        "evento": "sair cara",
        "probabilidade_teorica": p_cara,
        "n": n_lancamentos,
        "resultados": resultados,
        "trajetoria": trajetoria,
        "frequencia_final": frequencia_final,
        "erro_absoluto": abs(frequencia_final - p_cara),
        "n_sucessos": sum(resultados),
    }


def simular_lancamentos_dado(n_lancamentos, face_alvo=6, n_faces=6, semente=None):
    """Simula n lançamentos de um dado honesto de `n_faces` faces.

    O evento de interesse é "sair a face alvo", de probabilidade teórica
    1/n_faces. Convém contrastar com a moeda na interface: com p = 1/6 a
    convergência é visivelmente mais lenta em termos relativos, porque o
    evento é mais raro.
    """
    if n_lancamentos < 1:
        raise ValueError(f"n_lancamentos precisa ser ≥ 1, recebeu {n_lancamentos}")
    if n_faces < 2:
        raise ValueError(f"n_faces precisa ser ≥ 2, recebeu {n_faces}")
    if not 1 <= face_alvo <= n_faces:
        raise ValueError(
            f"face_alvo precisa estar entre 1 e {n_faces}, recebeu {face_alvo}"
        )

    gerador = random.Random(semente)
    faces = [gerador.randint(1, n_faces) for _ in range(n_lancamentos)]
    sucessos = [1 if face == face_alvo else 0 for face in faces]

    probabilidade_teorica = 1.0 / n_faces
    trajetoria = _trajetoria_frequencia_relativa(sucessos)
    frequencia_final = trajetoria[-1]

    # Distribuição empírica de todas as faces, para o gráfico de barras.
    contagem_faces = {face: 0 for face in range(1, n_faces + 1)}
    for face in faces:
        contagem_faces[face] += 1

    return {
        "tipo": "dado",
        "evento": f"sair a face {face_alvo}",
        "probabilidade_teorica": probabilidade_teorica,
        "n": n_lancamentos,
        "faces": faces,
        "trajetoria": trajetoria,
        "frequencia_final": frequencia_final,
        "erro_absoluto": abs(frequencia_final - probabilidade_teorica),
        "n_sucessos": sum(sucessos),
        "contagem_faces": contagem_faces,
    }


def reduzir_trajetoria(trajetoria, max_pontos=2000):
    """Reamostra a trajetória para plotagem, preservando a forma da curva.

    Com 100 mil lançamentos não há sentido em mandar 100 mil pontos para
    o gráfico. Tomamos pontos espaçados em ESCALA LOGARÍTMICA, e não
    uniformemente: é no começo da série que a curva oscila, e é
    justamente essa parte que a amostragem uniforme destruiria.

    Retorna (ns, frequencias) com o último ponto sempre incluído.
    """
    n = len(trajetoria)
    if n <= max_pontos:
        return list(range(1, n + 1)), list(trajetoria)

    indices = set()
    for i in range(max_pontos):
        posicao = (10 ** (math.log10(n) * i / (max_pontos - 1))) - 1
        indices.add(min(int(posicao), n - 1))
    indices.add(n - 1)

    ordenados = sorted(indices)
    return [i + 1 for i in ordenados], [trajetoria[i] for i in ordenados]


# ---------------------------------------------------------------------------
# (b) Teorema Central do Limite
# ---------------------------------------------------------------------------

def erro_padrao_teorico(desvio_padrao_populacional, tamanho_amostra):
    """Erro padrão da média — o desvio padrão previsto para X̄.

        σ_X̄ = σ / √n

    É a peça quantitativa do TCL: as médias amostrais se concentram em
    torno de μ, e a velocidade dessa concentração é 1/√n. Quadruplicar o
    tamanho da amostra reduz a dispersão das médias pela metade.
    """
    if desvio_padrao_populacional < 0:
        raise ValueError("desvio padrão não pode ser negativo")
    if tamanho_amostra < 1:
        raise ValueError(
            f"tamanho_amostra precisa ser ≥ 1, recebeu {tamanho_amostra}"
        )
    return desvio_padrao_populacional / math.sqrt(tamanho_amostra)


def simular_medias_amostrais(
    populacao, tamanho_amostra, n_repeticoes, semente=None, com_reposicao=True
):
    """Demonstração empírica do Teorema Central do Limite.

    Procedimento (Monte Carlo):
        1. sorteia uma amostra de tamanho n da população real;
        2. calcula a média dessa amostra (com nosso próprio `media`);
        3. repete r vezes;
        4. estuda a distribuição das r médias obtidas.

    O TCL afirma que, para n suficientemente grande,

        X̄  ~aprox~  Normal( μ , σ²/n )

    QUALQUER que seja o formato da distribuição da população — é isso que
    torna o teorema notável. No dataset usamos `cnt`, que é fortemente
    assimétrica à direita, e ainda assim a distribuição das médias fica
    visivelmente simétrica já com n moderado.

    Comparamos três coisas no resultado:
        média das médias      deve ≈ μ  (X̄ é estimador não enviesado)
        desvio das médias     deve ≈ σ/√n  (erro padrão teórico)
        forma da distribuição deve ≈ Normal
    """
    valores = _validar(populacao, minimo_exigido=2, nome="populacao")

    if tamanho_amostra < 1:
        raise ValueError(
            f"tamanho_amostra precisa ser ≥ 1, recebeu {tamanho_amostra}"
        )
    if n_repeticoes < 2:
        raise ValueError(
            f"n_repeticoes precisa ser ≥ 2 para haver dispersão, "
            f"recebeu {n_repeticoes}"
        )
    if not com_reposicao and tamanho_amostra > len(valores):
        raise ValueError(
            f"sem reposição, tamanho_amostra ({tamanho_amostra}) não pode "
            f"exceder a população ({len(valores)})"
        )

    gerador = random.Random(semente)
    tamanho_populacao = len(valores)

    medias = []
    for _ in range(n_repeticoes):
        if com_reposicao:
            amostra = [
                valores[gerador.randrange(tamanho_populacao)]
                for _ in range(tamanho_amostra)
            ]
        else:
            amostra = gerador.sample(valores, tamanho_amostra)
        medias.append(media(amostra))

    mu_populacional = media(valores)
    sigma_populacional = desvio_padrao(valores, amostral=False)

    esperado = erro_padrao_teorico(sigma_populacional, tamanho_amostra)
    observado = desvio_padrao(medias, amostral=True)

    return {
        "medias": medias,
        "tamanho_amostra": tamanho_amostra,
        "n_repeticoes": n_repeticoes,
        "mu_populacional": mu_populacional,
        "sigma_populacional": sigma_populacional,
        "media_das_medias": media(medias),
        "erro_padrao_teorico": esperado,
        "erro_padrao_observado": observado,
        # Razão observado/teórico: quanto mais perto de 1, melhor o TCL
        # está descrevendo a simulação.
        "razao_erro_padrao": observado / esperado if esperado > 0 else float("nan"),
    }
