"""Validação do Módulo 3 — Lei dos Grandes Números e Teorema Central do Limite.

Testes de simulação são estocásticos por natureza. Para que não fiquem
instáveis ("flaky"), toda simulação aqui usa SEMENTE FIXA, e as margens
de tolerância são derivadas da própria teoria — por exemplo, o desvio
esperado de uma frequência relativa é √(p(1−p)/n), então aceitamos
desvios de até 4 erros padrão, o que falharia por acaso em menos de
1 caso em 15 mil.
"""

import math

import numpy as np
import pytest
from scipy import stats

from core import minhastats as ms
from core import simulacao as sim
from conftest import assert_proximo


# ---------------------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------------------

def test_mesma_semente_produz_o_mesmo_resultado():
    primeiro = sim.simular_lancamentos_moeda(5000, semente=123)
    segundo = sim.simular_lancamentos_moeda(5000, semente=123)
    assert primeiro["resultados"] == segundo["resultados"]


def test_sementes_diferentes_produzem_resultados_diferentes():
    primeiro = sim.simular_lancamentos_moeda(5000, semente=1)
    segundo = sim.simular_lancamentos_moeda(5000, semente=2)
    assert primeiro["resultados"] != segundo["resultados"]


def test_tcl_e_reprodutivel(cnt_real):
    kwargs = dict(tamanho_amostra=30, n_repeticoes=500, semente=7)
    primeiro = sim.simular_medias_amostrais(cnt_real, **kwargs)
    segundo = sim.simular_medias_amostrais(cnt_real, **kwargs)
    assert primeiro["medias"] == segundo["medias"]


# ---------------------------------------------------------------------------
# Lei dos Grandes Números — moeda
# ---------------------------------------------------------------------------

def test_frequencia_relativa_acumulada_e_calculada_corretamente():
    resultado = sim.simular_lancamentos_moeda(1000, semente=42)

    referencia = np.cumsum(resultado["resultados"]) / np.arange(1, 1001)
    for i, (obtido, esperado) in enumerate(zip(resultado["trajetoria"], referencia)):
        assert_proximo(
            obtido, float(esperado), contexto=f"trajetória divergiu no passo {i+1}"
        )


@pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.75, 0.9])
def test_frequencia_converge_para_a_probabilidade_teorica(p):
    """Margem de 4 erros padrão: √(p(1−p)/n)."""
    n = 200000
    resultado = sim.simular_lancamentos_moeda(n, p_cara=p, semente=2024)

    erro_padrao = math.sqrt(p * (1 - p) / n)
    assert resultado["erro_absoluto"] < 4 * erro_padrao, (
        f"frequência final {resultado['frequencia_final']:.6f} longe demais de p={p}"
    )


def test_erro_diminui_com_mais_lancamentos():
    """A essência da LGN: mais dados, menos erro (na média de várias sementes)."""
    def erro_medio(n):
        erros = [
            sim.simular_lancamentos_moeda(n, semente=s)["erro_absoluto"]
            for s in range(30)
        ]
        return sum(erros) / len(erros)

    assert erro_medio(100000) < erro_medio(100)


def test_erro_decresce_na_ordem_de_um_sobre_raiz_de_n():
    """Multiplicar n por 100 deve reduzir o erro típico ~10 vezes."""
    def erro_medio(n):
        erros = [
            sim.simular_lancamentos_moeda(n, semente=s)["erro_absoluto"]
            for s in range(40)
        ]
        return sum(erros) / len(erros)

    razao = erro_medio(100) / erro_medio(10000)
    assert 5.0 < razao < 20.0, f"razão de erros fora do previsto: {razao:.2f}"


def test_contagem_de_sucessos_bate_com_os_resultados():
    resultado = sim.simular_lancamentos_moeda(3000, p_cara=0.4, semente=5)
    assert resultado["n_sucessos"] == sum(resultado["resultados"])
    assert resultado["frequencia_final"] == pytest.approx(
        resultado["n_sucessos"] / resultado["n"]
    )


def test_moeda_sempre_cara_e_sempre_coroa():
    assert sim.simular_lancamentos_moeda(100, p_cara=1.0, semente=1)["n_sucessos"] == 100
    assert sim.simular_lancamentos_moeda(100, p_cara=0.0, semente=1)["n_sucessos"] == 0


# ---------------------------------------------------------------------------
# Lei dos Grandes Números — dado
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_faces", [6, 10, 20])
def test_dado_converge_para_um_sobre_n_faces(n_faces):
    n = 300000
    resultado = sim.simular_lancamentos_dado(n, n_faces=n_faces, face_alvo=1, semente=99)

    p = 1.0 / n_faces
    erro_padrao = math.sqrt(p * (1 - p) / n)
    assert resultado["erro_absoluto"] < 4 * erro_padrao
    assert resultado["probabilidade_teorica"] == pytest.approx(p)


def test_todas_as_faces_aparecem_com_frequencia_parecida():
    """Qui-quadrado de aderência: um dado honesto não deve ser rejeitado."""
    resultado = sim.simular_lancamentos_dado(60000, semente=31)
    observadas = list(resultado["contagem_faces"].values())

    assert sum(observadas) == 60000
    _, p_valor = stats.chisquare(observadas)
    assert p_valor > 0.001, f"dado parece viciado (p = {p_valor:.5f})"


def test_parametros_invalidos_do_dado():
    with pytest.raises(ValueError):
        sim.simular_lancamentos_dado(100, face_alvo=7, n_faces=6)
    with pytest.raises(ValueError):
        sim.simular_lancamentos_dado(100, n_faces=1)
    with pytest.raises(ValueError):
        sim.simular_lancamentos_dado(0)


def test_p_cara_invalido():
    with pytest.raises(ValueError):
        sim.simular_lancamentos_moeda(100, p_cara=1.5)


# ---------------------------------------------------------------------------
# Redução da trajetória para plotagem
# ---------------------------------------------------------------------------

def test_trajetoria_curta_nao_e_reduzida():
    trajetoria = [0.5] * 100
    ns, valores = sim.reduzir_trajetoria(trajetoria, max_pontos=2000)
    assert ns == list(range(1, 101))
    assert valores == trajetoria


def test_trajetoria_longa_e_reduzida_preservando_o_fim():
    trajetoria = [i / 100000 for i in range(100000)]
    ns, valores = sim.reduzir_trajetoria(trajetoria, max_pontos=500)

    assert len(ns) <= 500
    assert len(ns) == len(valores)
    assert ns[-1] == 100000
    assert valores[-1] == trajetoria[-1]
    assert ns == sorted(ns)
    assert len(set(ns)) == len(ns)


def test_reducao_preserva_o_inicio_da_curva():
    """A amostragem logarítmica existe para não perder as oscilações iniciais."""
    trajetoria = [1.0 / (i + 1) for i in range(50000)]
    ns, _ = sim.reduzir_trajetoria(trajetoria, max_pontos=300)

    pontos_iniciais = [n for n in ns if n <= 100]
    assert len(pontos_iniciais) >= 30, (
        "amostragem logarítmica deveria concentrar pontos no início"
    )


# ---------------------------------------------------------------------------
# Erro padrão da média
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sigma, n", [(10.0, 1), (10.0, 4), (181.4, 30), (2.5, 1000)])
def test_erro_padrao_teorico_confere_com_a_formula(sigma, n):
    assert sim.erro_padrao_teorico(sigma, n) == pytest.approx(sigma / math.sqrt(n))


def test_quadruplicar_a_amostra_reduz_o_erro_padrao_pela_metade():
    assert sim.erro_padrao_teorico(20.0, 100) == pytest.approx(
        2 * sim.erro_padrao_teorico(20.0, 400)
    )


def test_erro_padrao_com_parametros_invalidos():
    with pytest.raises(ValueError):
        sim.erro_padrao_teorico(10.0, 0)
    with pytest.raises(ValueError):
        sim.erro_padrao_teorico(-1.0, 10)


# ---------------------------------------------------------------------------
# Teorema Central do Limite
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tamanho_amostra", [5, 30, 100, 500])
def test_media_das_medias_estima_a_media_populacional(cnt_real, tamanho_amostra):
    """X̄ é estimador não enviesado de μ, qualquer que seja n."""
    resultado = sim.simular_medias_amostrais(
        cnt_real, tamanho_amostra=tamanho_amostra, n_repeticoes=4000, semente=101
    )

    erro_padrao_da_media_das_medias = resultado["erro_padrao_teorico"] / math.sqrt(4000)
    desvio = abs(resultado["media_das_medias"] - resultado["mu_populacional"])

    assert desvio < 4 * erro_padrao_da_media_das_medias, (
        f"média das médias ({resultado['media_das_medias']:.3f}) longe de μ "
        f"({resultado['mu_populacional']:.3f})"
    )


@pytest.mark.parametrize("tamanho_amostra", [10, 30, 100, 400])
def test_dispersao_das_medias_segue_sigma_sobre_raiz_de_n(cnt_real, tamanho_amostra):
    resultado = sim.simular_medias_amostrais(
        cnt_real, tamanho_amostra=tamanho_amostra, n_repeticoes=5000, semente=202
    )
    assert resultado["razao_erro_padrao"] == pytest.approx(1.0, abs=0.06), (
        f"erro padrão observado {resultado['erro_padrao_observado']:.4f} "
        f"vs teórico {resultado['erro_padrao_teorico']:.4f}"
    )


def test_distribuicao_das_medias_fica_normal_apesar_da_populacao_assimetrica(cnt_real):
    """O coração do TCL: `cnt` é muito assimétrica, mas suas médias não são."""
    assimetria_populacional = ms.assimetria_pearson(cnt_real)
    assert assimetria_populacional > 0.3, "esperávamos população assimétrica à direita"

    resultado = sim.simular_medias_amostrais(
        cnt_real, tamanho_amostra=200, n_repeticoes=5000, semente=303
    )
    assimetria_das_medias = ms.assimetria_pearson(resultado["medias"])

    assert abs(assimetria_das_medias) < abs(assimetria_populacional) / 3, (
        f"as médias ainda parecem assimétricas: {assimetria_das_medias:.4f}"
    )


def test_normalidade_das_medias_melhora_com_n_maior(cnt_real):
    """Shapiro-Wilk sobre as médias: a estatística W deve subir com n."""
    def estatistica_w(tamanho_amostra):
        resultado = sim.simular_medias_amostrais(
            cnt_real,
            tamanho_amostra=tamanho_amostra,
            n_repeticoes=1500,
            semente=404,
        )
        return float(stats.shapiro(resultado["medias"]).statistic)

    assert estatistica_w(200) > estatistica_w(2)


def test_dispersao_das_medias_encolhe_quando_a_amostra_cresce(cnt_real):
    pequena = sim.simular_medias_amostrais(
        cnt_real, tamanho_amostra=10, n_repeticoes=3000, semente=505
    )
    grande = sim.simular_medias_amostrais(
        cnt_real, tamanho_amostra=1000, n_repeticoes=3000, semente=505
    )
    assert grande["erro_padrao_observado"] < pequena["erro_padrao_observado"]


def test_parametros_populacionais_conferem_com_numpy(cnt_real):
    resultado = sim.simular_medias_amostrais(
        cnt_real, tamanho_amostra=25, n_repeticoes=100, semente=1
    )
    assert_proximo(resultado["mu_populacional"], float(np.mean(cnt_real)))
    assert_proximo(resultado["sigma_populacional"], float(np.std(cnt_real, ddof=0)))


def test_amostragem_sem_reposicao_respeita_o_tamanho_da_populacao():
    populacao = [float(i) for i in range(50)]
    with pytest.raises(ValueError):
        sim.simular_medias_amostrais(
            populacao, tamanho_amostra=100, n_repeticoes=10, com_reposicao=False
        )


def test_tcl_com_parametros_invalidos(cnt_real):
    with pytest.raises(ValueError):
        sim.simular_medias_amostrais(cnt_real, tamanho_amostra=0, n_repeticoes=100)
    with pytest.raises(ValueError):
        sim.simular_medias_amostrais(cnt_real, tamanho_amostra=10, n_repeticoes=1)
