"""
Carregamento e catálogo de variáveis do dataset.

Esta camada pertence à INTERFACE, não ao núcleo: aqui podemos usar
Pandas à vontade para ler o CSV, filtrar linhas e rotular categorias. O
que não pode acontecer aqui é cálculo de medida estatística — isso é
responsabilidade exclusiva de `core/`.
"""

import os

import pandas as pd
import streamlit as st

CAMINHO_DADOS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hour.csv"
)

FONTE = {
    "nome": "Bike Sharing Dataset",
    "origem": "UCI Machine Learning Repository",
    "url": "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset",
    "citacao": (
        "Fanaee-T, Hadi & Gama, Joao (2013). Event labeling combining ensemble "
        "detectors and background knowledge. Progress in Artificial Intelligence, "
        "Springer Berlin Heidelberg. doi:10.1007/s13748-013-0040-3"
    ),
    "descricao": (
        "Registro horário do sistema público de aluguel de bicicletas Capital "
        "Bikeshare, em Washington D.C., cobrindo os anos de 2011 e 2012, "
        "enriquecido com dados meteorológicos da freemeteo.com."
    ),
}


# ---------------------------------------------------------------------------
# Desnormalização
# ---------------------------------------------------------------------------
# As variáveis meteorológicas vêm normalizadas no intervalo [0, 1]. Para que
# as medidas e os coeficientes de regressão tenham significado físico,
# reconstruímos as unidades originais.
#
# ATENÇÃO — divergência de documentação:
# O arquivo Readme.txt distribuído com o dataset (2013) afirma que `temp` foi
# "dividida por 41" e `atemp` "dividida por 50". Essa descrição está ERRADA.
# A página atual do dataset no UCI documenta uma normalização min-máx,
#     temp  = (t − (−8)) / (39 − (−8))
#     atemp = (t − (−16)) / (50 − (−16))
# e é essa que reproduz a realidade: sob a fórmula do Readme, a média de
# janeiro em Washington D.C. seria de 9,7 °C, contra 1,9 °C observados
# historicamente na cidade; sob a min-máx, dá 3,2 °C. Conferimos mês a mês e
# a min-máx acompanha a climatologia da cidade em todos os doze. Usamos a
# min-máx e registramos a divergência no RELATORIO.md.

TEMP_MINIMA = -8.0
TEMP_MAXIMA = 39.0
SENSACAO_MINIMA = -16.0
SENSACAO_MAXIMA = 50.0
VENTO_MAXIMO = 67.0

ROTULOS_ESTACAO = {1: "Primavera", 2: "Verão", 3: "Outono", 4: "Inverno"}
ROTULOS_CLIMA = {
    1: "Céu limpo",
    2: "Névoa/nublado",
    3: "Chuva ou neve leve",
    4: "Chuva forte/tempestade",
}
ROTULOS_DIA_SEMANA = {
    0: "Domingo",
    1: "Segunda",
    2: "Terça",
    3: "Quarta",
    4: "Quinta",
    5: "Sexta",
    6: "Sábado",
}
ROTULOS_SIM_NAO = {0: "Não", 1: "Sim"}
ROTULOS_ANO = {0: "2011", 1: "2012"}
ROTULOS_MES = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril",
    5: "Maio", 6: "Junho", 7: "Julho", 8: "Agosto",
    9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro",
}


@st.cache_data
def carregar_dados():
    """Lê o CSV e acrescenta as colunas desnormalizadas e rotuladas.

    O cache do Streamlit evita reler e reprocessar o arquivo a cada
    interação do usuário — sem ele, cada clique repetiria a leitura das
    17.379 linhas.
    """
    if not os.path.exists(CAMINHO_DADOS):
        raise FileNotFoundError(
            f"Dataset não encontrado em {CAMINHO_DADOS}.\n"
            "Baixe-o de https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset "
            "e coloque hour.csv dentro da pasta data/."
        )

    dados = pd.read_csv(CAMINHO_DADOS)
    dados["dteday"] = pd.to_datetime(dados["dteday"])

    dados["temp_c"] = dados["temp"] * (TEMP_MAXIMA - TEMP_MINIMA) + TEMP_MINIMA
    dados["atemp_c"] = (
        dados["atemp"] * (SENSACAO_MAXIMA - SENSACAO_MINIMA) + SENSACAO_MINIMA
    )
    dados["hum_pct"] = dados["hum"] * 100.0
    dados["vento_kmh"] = dados["windspeed"] * VENTO_MAXIMO

    dados["estacao"] = dados["season"].map(ROTULOS_ESTACAO)
    dados["clima"] = dados["weathersit"].map(ROTULOS_CLIMA)
    dados["dia_semana"] = dados["weekday"].map(ROTULOS_DIA_SEMANA)
    dados["dia_util"] = dados["workingday"].map(ROTULOS_SIM_NAO)
    dados["feriado"] = dados["holiday"].map(ROTULOS_SIM_NAO)
    dados["ano"] = dados["yr"].map(ROTULOS_ANO)
    dados["mes"] = dados["mnth"].map(ROTULOS_MES)
    dados["hora"] = dados["hr"]

    return dados


# ---------------------------------------------------------------------------
# Catálogo de variáveis
# ---------------------------------------------------------------------------

VARIAVEIS_NUMERICAS = {
    "cnt": {
        "rotulo": "Total de aluguéis por hora",
        "unidade": "bicicletas",
        "tipo": "discreta",
        "descricao": (
            "Número de bicicletas alugadas na hora. É uma CONTAGEM, o que a "
            "torna candidata natural a um ajuste de Poisson."
        ),
    },
    "casual": {
        "rotulo": "Aluguéis por usuários casuais",
        "unidade": "bicicletas",
        "tipo": "discreta",
        "descricao": "Aluguéis feitos por usuários não cadastrados no sistema.",
    },
    "registered": {
        "rotulo": "Aluguéis por usuários cadastrados",
        "unidade": "bicicletas",
        "tipo": "discreta",
        "descricao": (
            "Aluguéis feitos por assinantes. Concentram a maior parte da "
            "demanda e seguem o padrão de deslocamento para o trabalho."
        ),
    },
    "temp_c": {
        "rotulo": "Temperatura",
        "unidade": "°C",
        "tipo": "continua",
        "descricao": (
            "Temperatura do ar, reconstruída a partir da coluna normalizada "
            "`temp` pela fórmula min-máx com t ∈ [−8 °C, 39 °C]."
        ),
    },
    "atemp_c": {
        "rotulo": "Sensação térmica",
        "unidade": "°C",
        "tipo": "continua",
        "descricao": (
            "Temperatura aparente, que combina temperatura, umidade e vento. "
            "Reconstruída de `atemp` com t ∈ [−16 °C, 50 °C]."
        ),
    },
    "hum_pct": {
        "rotulo": "Umidade relativa",
        "unidade": "%",
        "tipo": "continua",
        "descricao": "Umidade relativa do ar, de 0 a 100%.",
    },
    "vento_kmh": {
        "rotulo": "Velocidade do vento",
        "unidade": "km/h",
        "tipo": "continua",
        "descricao": (
            "Velocidade do vento. Fortemente assimétrica à direita e com "
            "excesso de zeros — boa candidata a um ajuste Exponencial."
        ),
    },
    "hora": {
        "rotulo": "Hora do dia",
        "unidade": "h",
        "tipo": "discreta",
        "descricao": (
            "Hora do registro, de 0 a 23. Por construção do dataset é quase "
            "perfeitamente uniforme."
        ),
    },
}

VARIAVEIS_CATEGORICAS = {
    "estacao": {
        "rotulo": "Estação do ano",
        "descricao": "Primavera, verão, outono ou inverno.",
    },
    "clima": {
        "rotulo": "Condição climática",
        "descricao": (
            "Situação do tempo na hora do registro, do céu limpo à "
            "tempestade."
        ),
    },
    "dia_semana": {
        "rotulo": "Dia da semana",
        "descricao": "De domingo a sábado.",
    },
    "dia_util": {
        "rotulo": "É dia útil?",
        "descricao": "Verdadeiro quando não é fim de semana nem feriado.",
    },
    "feriado": {
        "rotulo": "É feriado?",
        "descricao": "Feriados oficiais do Distrito de Columbia.",
    },
    "estacao_ano": {
        "rotulo": "Ano",
        "descricao": "2011 ou 2012 — o sistema cresceu muito entre eles.",
    },
    "mes": {
        "rotulo": "Mês",
        "descricao": "Mês do registro.",
    },
}

# `estacao_ano` é só um apelido de exibição para a coluna `ano`, que já é
# usada como rótulo; o mapa evita colidir com o nome `estacao`.
COLUNA_REAL_CATEGORICA = {"estacao_ano": "ano"}


def formatar_inteiro(valor):
    """Inteiro com ponto como separador de milhar: 17379 → "17.379".

    Existe para evitar o erro de aplicar `.replace(",", ".")` sobre uma
    frase inteira: isso converte também as vírgulas do texto, e
    "dataset completo, sem filtros" vira "dataset completo. sem filtros".
    A troca precisa acontecer só sobre o número.
    """
    return f"{valor:,}".replace(",", ".")


def coluna_categorica(chave):
    """Resolve a chave do catálogo na coluna real do DataFrame."""
    return COLUNA_REAL_CATEGORICA.get(chave, chave)


def rotulo_completo(chave):
    """Rótulo com unidade, para eixos e títulos de gráfico."""
    if chave in VARIAVEIS_NUMERICAS:
        meta = VARIAVEIS_NUMERICAS[chave]
        unidade = meta["unidade"]
        return f"{meta['rotulo']} ({unidade})" if unidade else meta["rotulo"]
    if chave in VARIAVEIS_CATEGORICAS:
        return VARIAVEIS_CATEGORICAS[chave]["rotulo"]
    return chave


def serie_numerica(dados, chave):
    """Extrai uma variável numérica como lista de floats, pronta para o core.

    A fronteira entre interface e núcleo é exatamente aqui: daqui para
    dentro de `core/` só trafegam listas de números do Python, nunca
    Series do Pandas ou arrays do NumPy.
    """
    return dados[chave].astype(float).tolist()
