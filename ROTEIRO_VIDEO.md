# Roteiro do vídeo de demonstração

**Duração alvo:** 4 min 30 s (dentro da faixa de 3 a 5 min)
**Formato:** gravação de tela da aplicação rodando, com narração alternada
**Equipe:** Plínio · Paulo César · Jhonata — cada um narra pelo menos dois trechos

---

## Antes de gravar

- [ ] `streamlit run app/principal.py` já aberto, na página **Visão geral**
- [ ] Barra lateral com **todos os filtros abertos** (dataset completo)
- [ ] Rodar `pytest` uma vez antes e deixar o terminal com os **612 passed**
      visível, em outra aba, para o trecho do minuto 3:25
- [ ] Zoom do navegador em 100%; janela maximizada
- [ ] Cronometrar um ensaio: o trecho que mais costuma estourar é o 4

---

## Trecho 1 — O dataset · 0:00 a 0:30 · **Jhonata**

**Tela:** página *Visão geral do dataset*, rolando devagar pelas métricas.

> "Nosso dataset é o Bike Sharing, do repositório UCI: dezessete mil trezentos e
> setenta e nove registros horários do sistema público de bicicletas de
> Washington, em 2011 e 2012. São oito variáveis numéricas e sete categóricas.
>
> Escolhemos ele porque cada módulo do trabalho encontra aqui um caso natural:
> tem uma variável de contagem, uma quase simétrica, uma com cauda longa e uma
> praticamente uniforme.
>
> E ele já rendeu a primeira lição: o Readme oficial do dataset informa a
> fórmula errada para desnormalizar a temperatura. Conferimos mês a mês contra o
> clima real de Washington e usamos a fórmula correta."

*(Abrir rapidamente o expander "Sobre a desnormalização" e fechar.)*

---

## Trecho 2 — Estatística descritiva · 0:30 a 1:15 · **Plínio**

**Tela:** módulo *1 · Estatística descritiva*, variável **Total de aluguéis por hora**.

> "No primeiro módulo o usuário escolhe uma variável e a aplicação calcula tudo.
> Repare: média cento e oitenta e nove, mediana cento e quarenta e dois. Essa
> diferença de quarenta e sete bicicletas não é ruído — é assimetria à direita.
>
> O coeficiente de variação é de noventa e seis por cento: o desvio padrão quase
> iguala a média, o que já avisa que a média sozinha descreve mal esses dados."

*(Rolar até o boxplot.)*

> "No boxplot, o losango laranja é a média e a linha é a mediana. A média está
> visivelmente à direita — é a cauda longa puxando ela.
>
> E vejam a cerca inferior do IQR: menos trezentos e vinte e um. Um número
> impossível para uma contagem de bicicletas. Isso acontece porque a regra do
> IQR pressupõe simetria, e esses dados não são simétricos."

*(Rolar até a Interpretação automática.)*

> "Esse texto aqui embaixo é gerado automaticamente a partir dos números que o
> nosso núcleo acabou de calcular."

---

## Trecho 3 — Probabilidade e simulação · 1:15 a 2:00 · **Paulo César**

**Tela:** módulo *2 · Probabilidade e simulação*, aba **Lei dos Grandes Números**.

> "Aqui simulamos a Lei dos Grandes Números. Vou subir o número de lançamentos."

*(Arrastar o controle de 100 até 100.000 — deixar a curva se assentar na tela.)*

> "A frequência relativa converge para a probabilidade teórica. Repare que ela
> **não** melhora de forma suave: oscila muito no começo e vai se acalmando. A
> lei garante a convergência, não que cada passo seja melhor que o anterior.
>
> O eixo está em escala logarítmica porque a precisão cresce com um sobre raiz de
> n — para dobrar a precisão é preciso quadruplicar os lançamentos."

*(Trocar para a aba **Teorema Central do Limite**.)*

> "Agora o Teorema Central do Limite, com uma variável real do dataset. À
> esquerda, a população: bem torta, assimetria de zero vírgula setenta e nove.
> À direita, as médias de amostras de tamanho trinta."

*(Mudar o tamanho da amostra de 2 para 30 e depois 200, mostrando o sino se formar.)*

> "Com n igual a dois, as médias ainda herdam a assimetria. Com trinta já é um
> sino. E a curva laranja **não foi ajustada** ao histograma: são os parâmetros
> que a teoria prevê, calculados antes da simulação começar."

---

## Trecho 4 — Distribuições teóricas · 2:00 a 2:40 · **Plínio**

**Tela:** módulo *3 · Distribuições teóricas*, variável **Total de aluguéis por hora**,
distribuição **Poisson**.

> "Total de aluguéis é uma contagem de eventos por hora — o caso de livro-texto
> da Poisson. Ajustamos, e foi o pior resultado de todos."

*(Apontar a distância de variação total: 0,8422.)*

> "Olhem o gráfico: as barras laranja, que são a Poisson, empilham quase toda a
> probabilidade em duas classes. Os dados observados se espalham de um a
> novecentos e setenta e sete.
>
> O motivo está no índice de dispersão: cento e setenta e três vírgula seis.
> Numa Poisson genuína ele valeria um. A Poisson supõe uma taxa constante, e
> aqui a demanda vai de seis bicicletas às quatro da manhã a quatrocentas e
> sessenta e uma às cinco da tarde. Não existe um lambda único."

*(Abrir o expander "Comparar o ajuste de todas as candidatas".)*

> "Aqui dá para ver: a Exponencial ajusta melhor que a Poisson, e até a Uniforme
> ajusta melhor. Ser uma contagem não faz de uma variável uma Poisson."

---

## Trecho 5 — Correlação e regressão · 2:40 a 3:25 · **Jhonata**

**Tela:** módulo *4 · Correlação e regressão*, X = **Temperatura**, Y = **Total de aluguéis**.

> "Temperatura contra aluguéis: correlação de zero vírgula quarenta, R² de zero
> vírgula dezesseis. A reta diz que cada grau Celsius a mais corresponde a oito
> bicicletas a mais por hora."

*(Digitar 25 no campo de predição.)*

> "A predição é interativa — a vinte e cinco graus, o modelo prevê duzentos e
> sessenta e sete aluguéis."

*(Digitar 60 e mostrar o aviso de extrapolação.)*

> "E se eu pedir sessenta graus, a aplicação avisa que estou extrapolando."

*(Trocar X para **Hora do dia**.)*

> "Agora o achado que mais nos surpreendeu. A hora do dia tem correlação de zero
> vírgula trinta e nove — parece menos importante que a temperatura. Mas olhem a
> nuvem: dois picos, às oito e às dezoito horas. É o padrão casa–trabalho.
>
> A reta não captura isso de jeito nenhum. Calculamos a razão de correlação, que
> não pressupõe forma nenhuma, e a hora explica **cinquenta por cento** da
> variação, contra os quinze que a reta enxerga. O modelo linear vê menos de um
> terço do efeito real."

*(Rolar até o aviso vermelho.)*

> "E fechamos sempre com o aviso: correlação não implica causalidade."

---

## Trecho 6 — O núcleo estatístico por dentro · 3:25 a 4:05 · **Paulo César**

**Tela:** editor de código aberto em `core/minhastats.py`, na função `variancia`.

> "Nenhuma medida que vocês viram usa NumPy ou SciPy. Está tudo escrito na mão.
> Esta é a variância:"

```python
x_barra = media(valores)

soma_quadrados = 0.0
for x in valores:
    desvio = x - x_barra
    soma_quadrados += desvio * desvio

divisor = n - 1 if amostral else n
return soma_quadrados / divisor
```

> "É o somatório dos desvios ao quadrado, dividido por n menos um na versão
> amostral. Esse n menos um é a correção de Bessel: como usamos a média estimada
> dos próprios dados no lugar da média verdadeira, os desvios ficam pequenos
> demais, e o divisor menor corrige esse viés."

*(Trocar para o terminal com o resultado do pytest.)*

> "E cada função dessas tem teste comparando com a biblioteca de referência. São
> seiscentos e doze testes, com tolerância relativa de dez elevado a menos nove.
>
> Dois deles acharam bugs de verdade: um mostrou que a constante três vírgula
> trezentos e vinte e dois da regra de Sturges erra por uma classe quando n é
> potência de dois, e outro pegou uma colisão de nome de módulo que quebrava a
> aplicação."

---

## Trecho 7 — Fechamento · 4:05 a 4:30 · **Jhonata**

**Tela:** voltar ao módulo *4*, com o gráfico da hora do dia na tela.

> "Nosso melhor achado foi esse: o coeficiente de Pearson subestimou em três
> vezes o efeito da hora do dia, porque só enxerga relação linear e o padrão real
> tem dois picos.
>
> A lição que levamos é que correlação perto de zero não significa ausência de
> relação — significa ausência de relação *linear*. Olhar o gráfico de dispersão
> antes de confiar no coeficiente é o que separa a conclusão certa da errada.
>
> O código, o relatório completo e as instruções de execução estão no
> repositório. Obrigado!"

---

## Divisão da narração

| Integrante | Trechos | Tempo aproximado |
|---|---|---|
| Jhonata | 1, 5, 7 | 1 min 40 s |
| Plínio | 2, 4 | 1 min 25 s |
| Paulo César | 3, 6 | 1 min 25 s |

## Checklist final

- [ ] Todos os três integrantes aparecem narrando
- [ ] Dataset apresentado nos primeiros 30 segundos
- [ ] Pelo menos um gráfico de cada módulo aparece em tela
- [ ] Trecho de código do núcleo mostrado e explicado
- [ ] Melhor achado no fechamento
- [ ] Duração entre 3 e 5 minutos
