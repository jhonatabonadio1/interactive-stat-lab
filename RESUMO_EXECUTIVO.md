# Laboratório Estatístico Interativo — Resumo executivo

**Matemática e Estatística para Computação · CEUB**
Plínio Roberto Pereira (72650385) · Paulo César Farias Silva (72650229) · Jhonata Bonadio (72650384)
Repositório: `github.com/jhonatabonadio1/interactive-stat-lab`

---

## Dataset

**Bike Sharing Dataset** (UCI Machine Learning Repository, dataset 275) — registro
horário do sistema público de bicicletas de Washington D.C. em 2011 e 2012, com
dados meteorológicos. **17.379 registros, 8 variáveis numéricas e 7 categóricas.**
Escolhido porque cada módulo do trabalho encontra nele um caso natural: uma
contagem (aluguéis por hora), uma variável aproximadamente simétrica
(temperatura), uma com cauda longa (vento) e uma quase uniforme (hora do dia).

Já na preparação encontramos um erro na documentação oficial: o `Readme.txt` do
dataset informa a fórmula errada de desnormalização da temperatura, o que
superestima todos os meses em cerca de 6 °C. Conferimos contra a climatologia
real de Washington e adotamos a fórmula correta.

## O que foi implementado

**Aplicação Streamlit em Python**, dividida em duas camadas que não se misturam:
`core/` contém matemática pura, sem nenhuma dependência de interface; `app/`
apenas lê dados, monta controles e desenha resultados.

| Módulo | Entrega |
|---|---|
| **1 · Descritiva** | Tabela de frequências (classes por Sturges, ajustável), todas as medidas de tendência central e dispersão, histograma, boxplot, detecção de outliers pelo IQR e interpretação textual gerada automaticamente |
| **2 · Probabilidade** | Monte Carlo da Lei dos Grandes Números (moeda e dado) e do Teorema Central do Limite sobre variável real, com repetições, tamanho de amostra e semente sob controle do usuário |
| **3 · Distribuições** | Normal, Exponencial, Uniforme e Poisson sobrepostas ao histograma, com parâmetros estimados dos dados e medida objetiva de qualidade do ajuste |
| **4 · Regressão** | Dispersão, correlação, reta de mínimos quadrados, R², predição interativa com aviso de extrapolação, diagnóstico de resíduos e alerta sobre causalidade |

**A regra de ouro:** toda medida exibida ao usuário é calculada por código nosso,
a partir da fórmula explícita, usando apenas o módulo `math` da biblioteca padrão.
Os gráficos seguem o mesmo princípio — o histograma vem da nossa tabela de
classes e o boxplot dos nossos quartis, sem `plt.hist` ou `plt.boxplot`.

**Validação: 612 testes automatizados, todos passando**, comparando cada função
com NumPy e SciPy sobre sete conjuntos de dados diferentes, com tolerância
relativa de 10⁻⁹. Os testes verificam também propriedades que precisam valer por
construção (SQ_tot = SQ_reg + SQ_res, R² = r², soma dos resíduos nula, falta de
memória da Exponencial) e encontraram dois bugs reais durante o desenvolvimento.

## As três descobertas

**1. Ser uma contagem não faz de uma variável uma Poisson.** O total de aluguéis
por hora é um inteiro não negativo — o caso de livro-texto da Poisson. Foi o pior
ajuste de todos os candidatos testados (distância 0,8422, contra 0,1061 da
Exponencial). O índice de dispersão Var/λ vale **173,66** quando deveria valer 1.
A causa é conceitual: a Poisson supõe taxa constante, mas a demanda vai de 6
bicicletas às 4h a 461 às 17h. Não existe um λ único, existe um λ(t).

**2. O coeficiente de Pearson subestimou em 3× o efeito da hora do dia.** Pela
correlação, a hora parece menos importante que a temperatura (r = 0,3941 contra
0,4048). Mas o diagrama de dispersão revela dois picos — 8h e 17–18h, o padrão
casa–trabalho — que uma reta não consegue descrever. A razão de correlação, que
não pressupõe forma alguma, mostra que a hora explica **50,2%** da variação,
contra os 15,5% que o modelo linear capta. Correlação perto de zero não significa
ausência de relação, significa ausência de relação *linear*.

**3. O TCL não só funciona sobre dados torcidos, como acerta o número.** A
população tem assimetria de 0,7850 e CV de 95,7%. Ainda assim, a assimetria das
médias amostrais cai para **0,0255** com n = 200 — uma redução de 31×. Mais do
que isso: o erro padrão observado bate com o σ/√n previsto com erro inferior a
**0,5%** em n = 30, e a curva Normal sobreposta não foi ajustada aos dados — seus
parâmetros vêm da população, calculados antes da simulação começar.

## Como executar

```bash
git clone https://github.com/jhonatabonadio1/interactive-stat-lab.git
cd interactive-stat-lab
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/principal.py      # a aplicação
pytest                              # os 612 testes
```
