"""
Regressão linear simples por mínimos quadrados — Módulo 5.

Implementação à mão do estimador de mínimos quadrados ordinários (MQO),
sem `np.polyfit`, `scipy.stats.linregress` ou `sklearn`.
"""

import math

from core.minhastats import _validar_pareado, correlacao_pearson, media

__all__ = ["RegressaoLinear", "ajustar_regressao_linear"]


class RegressaoLinear:
    """Resultado de um ajuste de reta por mínimos quadrados.

    Guarda os coeficientes e as somas de quadrados, e sabe prever novos
    valores. É uma classe (e não um dicionário) porque a predição
    interativa do Módulo 5 precisa reaplicar o modelo já ajustado a
    valores de X digitados pelo usuário.
    """

    def __init__(
        self,
        b0,
        b1,
        r,
        r2,
        sq_total,
        sq_residual,
        sq_regressao,
        erro_padrao_estimativa,
        erro_padrao_b1,
        n,
        x_minimo,
        x_maximo,
    ):
        self.b0 = b0
        self.b1 = b1
        self.r = r
        self.r2 = r2
        self.sq_total = sq_total
        self.sq_residual = sq_residual
        self.sq_regressao = sq_regressao
        self.erro_padrao_estimativa = erro_padrao_estimativa
        self.erro_padrao_b1 = erro_padrao_b1
        self.n = n
        self.x_minimo = x_minimo
        self.x_maximo = x_maximo

    def prever(self, x):
        """Valor ajustado para um X qualquer.

            ŷ = b₀ + b₁·x
        """
        return self.b0 + self.b1 * x

    def prever_varios(self, xs):
        """Aplica `prever` a uma sequência de valores."""
        return [self.prever(x) for x in xs]

    def extrapola(self, x):
        """Indica se x está fora do intervalo observado de X.

        Serve para a interface avisar o usuário: a reta só foi estimada
        dentro de [x_mín, x_máx] e não há evidência de que a relação
        linear continue valendo fora desse intervalo.
        """
        return x < self.x_minimo or x > self.x_maximo

    def equacao(self, casas=4, nome_x="x", nome_y="ŷ"):
        """A equação da reta formatada para exibição."""
        sinal = "+" if self.b1 >= 0 else "−"
        return (
            f"{nome_y} = {self.b0:.{casas}f} {sinal} "
            f"{abs(self.b1):.{casas}f}·{nome_x}"
        )

    def residuos(self, x, y):
        """Resíduos observados: e_i = y_i − ŷ_i.

        São a matéria-prima do diagnóstico do ajuste: se a relação for
        mesmo linear e homocedástica, eles devem se espalhar sem padrão
        em torno de zero.
        """
        vx, vy = _validar_pareado(x, y)
        return [vy[i] - self.prever(vx[i]) for i in range(len(vx))]

    def __repr__(self):
        return (
            f"RegressaoLinear(b0={self.b0:.6f}, b1={self.b1:.6f}, "
            f"r2={self.r2:.6f}, n={self.n})"
        )


def ajustar_regressao_linear(x, y):
    """Ajusta ŷ = b₀ + b₁·x minimizando a soma dos quadrados dos resíduos.

    O método dos mínimos quadrados escolhe b₀ e b₁ que minimizam

        SQ_res(b₀, b₁) = Σ (y_i − b₀ − b₁·x_i)²

    Derivando em relação a b₀ e b₁ e igualando a zero, chega-se às
    equações normais, cuja solução é:

              Σ (x_i − x̄)(y_i − ȳ)        S_xy
        b₁ = ─────────────────────── =  ───────
                 Σ (x_i − x̄)²             S_xx

        b₀ = ȳ − b₁·x̄

    A segunda equação mostra que a reta ajustada sempre passa pelo ponto
    médio (x̄, ȳ), qualquer que sejam os dados.

    Decomposição da variabilidade de Y:

        SQ_tot = Σ (y_i − ȳ)²        variação total
        SQ_reg = Σ (ŷ_i − ȳ)²        explicada pela reta
        SQ_res = Σ (y_i − ŷ_i)²      não explicada (resíduo)

        SQ_tot = SQ_reg + SQ_res

    Coeficiente de determinação — a fração da variação de Y que a reta
    consegue explicar:

        R² = SQ_reg / SQ_tot = 1 − SQ_res / SQ_tot

    Na regressão linear SIMPLES vale ainda R² = r², o que os testes
    verificam explicitamente.

    Erro padrão da estimativa (dispersão típica dos pontos em torno da
    reta, na unidade de Y); o divisor (n−2) reflete os dois parâmetros
    estimados:

        s_e = √[ SQ_res / (n − 2) ]

    Erro padrão do coeficiente angular:

        s_b1 = s_e / √S_xx
    """
    vx, vy = _validar_pareado(x, y, minimo_exigido=3)
    n = len(vx)

    x_barra = media(vx)
    y_barra = media(vy)

    # S_xy = Σ(x−x̄)(y−ȳ)   e   S_xx = Σ(x−x̄)²
    s_xy = 0.0
    s_xx = 0.0
    for i in range(n):
        desvio_x = vx[i] - x_barra
        s_xy += desvio_x * (vy[i] - y_barra)
        s_xx += desvio_x * desvio_x

    if s_xx == 0:
        raise ValueError(
            "regressão impossível: a variável X é constante, "
            "não há inclinação definida"
        )

    b1 = s_xy / s_xx
    b0 = y_barra - b1 * x_barra

    sq_total = 0.0
    sq_residual = 0.0
    sq_regressao = 0.0
    for i in range(n):
        y_ajustado = b0 + b1 * vx[i]
        desvio_total = vy[i] - y_barra
        residuo = vy[i] - y_ajustado
        desvio_explicado = y_ajustado - y_barra

        sq_total += desvio_total * desvio_total
        sq_residual += residuo * residuo
        sq_regressao += desvio_explicado * desvio_explicado

    if sq_total == 0:
        raise ValueError(
            "regressão impossível: a variável Y é constante, R² indefinido"
        )

    r2 = 1.0 - sq_residual / sq_total

    # Y não é constante (checado acima) e X tampouco, então r está definido.
    r = correlacao_pearson(vx, vy)

    erro_padrao_estimativa = math.sqrt(sq_residual / (n - 2))
    erro_padrao_b1 = erro_padrao_estimativa / math.sqrt(s_xx)

    x_min = vx[0]
    x_max = vx[0]
    for valor in vx:
        if valor < x_min:
            x_min = valor
        if valor > x_max:
            x_max = valor

    return RegressaoLinear(
        b0=b0,
        b1=b1,
        r=r,
        r2=r2,
        sq_total=sq_total,
        sq_residual=sq_residual,
        sq_regressao=sq_regressao,
        erro_padrao_estimativa=erro_padrao_estimativa,
        erro_padrao_b1=erro_padrao_b1,
        n=n,
        x_minimo=x_min,
        x_maximo=x_max,
    )
