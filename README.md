## Analásie de sentimentos - Canal Você Sabia? - Utilizando biblioteca pronta da https://huggingface.co/pysentimiento/bertweet-pt-sentiment

Com o intuito de estudo e possível escalabilidade, propomos um estudo de sentimentos do Canal Você Sábia?
Pois é notável o tamanho do canal e seu crescimento ao longo dos anos.

Avaliamos uma das ferramentas de análise de sentimentos, a pysentimiento, sendo essa uma ferramenta treinada em posts do antigo Twitter, onde essa ferramenta é capaz de avaliar se um comentário é positivo, negativo ou neutro, com base na distribuição das palavras e emoji, atribuindo pesos em cada.

## Resultados da Análise de Sentimento Calibrada

Na situação em questão, precisei realizar um ajuste de pesos, pois, como a ferramenta é versada em comentários do antigo Twitter, poderia não ser capaz de avaliar o sarcasmo ou repetição de caracteres, mas, trouxe resultados bastante interessantes, porém ao deparar com a taxa de acerto, vi que não era o que esperava, entretanto, era de se esperar que o resultado fosse baixo, devido ao Dataset ser pequeno.

Mas nessa situação, foi optado interferir um pouco na métrica e reavaliar, informando para a ferramenta que ela avaliasse como neutro somente se tivesse ou 70% ou 80% ou 90% de certeza, forçando a mesma a optar por positivo ou negativo, onde tivemos resultados ainda melhores, e uma taxa de acerto mais promissora mesmo que baixa.

Após aplicar a função de calibragem de confiança, os resultados do modelo de análise de sentimento `pysentimiento` são os seguintes:

# Acurácia (Qual a porcentagem total de acertos?)
acuracia = accuracy_score(y_verdadeiro, y_predito)
print(f"🎯 Acurácia Geral do pysentimiento: {acuracia * 100:.2f}%")

# F1-Score (A média harmônica, excelente para ver se ele é bom em todas as classes)
# Usamos average='weighted' para balancear caso tenha muitos NEU e poucos POS
f1 = f1_score(y_verdadeiro, y_predito, average='weighted')
print(f"📊 F1-Score: {f1:.4f}")

# Matriz de Confusão (Onde o modelo está errando?)
labels = ["POS", "NEG", "NEU"]
matriz = confusion_matrix(y_verdadeiro, y_predito, labels=labels)

*   **Acurácia Geral:** 56.85%
    *   Este valor representa a proporção total de predições corretas (sentimentos positivos, negativos e neutros) em relação ao total de comentários analisados.

*   **F1-Score (Ponderado):** 0.5525
    *   O F1-Score é uma média harmônica entre a precisão e o recall, sendo uma métrica robusta, especialmente útil quando as classes estão desbalanceadas. Um F1-Score de 0.5525 indica um desempenho moderado do modelo em identificar corretamente os sentimentos em todas as classes, considerando a penalização aplicada aos neutros.

*   **Matriz de Confusão:**

|           | Predito_POS | Predito_NEG | Predito_NEU |
| :-------- | :---------- | :---------- | :---------- |
| **Real_POS**  | 152         | 11          | 44          |
| **Real_NEG**  | 27          | 46          | 96          |
| **Real_NEU**  | 58          | 38          | 163         |

### Interpretação da Matriz de Confusão:

*   **Verdadeiros Positivos (Real_POS | Predito_POS):** 152 comentários que eram realmente `POS` foram classificados corretamente como `POS`.
*   **Falsos Negativos (Real_POS | Predito_NEG / Predito_NEU):** Dos comentários `POS`, 11 foram classificados erroneamente como `NEG` e 44 como `NEU`.
*   **Verdadeiros Negativos (Real_NEG | Predito_NEG):** 46 comentários que eram realmente `NEG` foram classificados corretamente como `NEG`.
*   **Falsos Positivos (Real_NEG | Predito_POS / Predito_NEU):** Dos comentários `NEG`, 27 foram classificados erroneamente como `POS` e 96 como `NEU`.
*   **Verdadeiros Neutros (Real_NEU | Predito_NEU):** 163 comentários que eram realmente `NEU` foram classificados corretamente como `NEU`.
*   **Mal Classificados de Neutros (Real_NEU | Predito_POS / Predito_NEG):** Dos comentários `NEU`, 58 foram classificados erroneamente como `POS` e 38 como `NEG`.

Esta matriz oferece uma visão detalhada de onde o modelo acerta e erra, e como a calibragem afetou a classificação dos sentimentos, especialmente a propensão a classificar neutros como positivos ou negativos.# Projetos-de-estudos

## Outras Avaliações
Utilizando uma penalização de 90% tivemos os resultados 

#🎯 Acurácia Geral do pysentimiento: 51.18%
#📊 F1-Score: 0.4838

#🔍 Matriz de Confusão:
#         Predito_POS | Predito_NEG | Predito_NEU
#Real_POS:    178     | 14      | 15
#Real_NEG:    53      | 72      | 44
#Real_NEU:    120     | 64      | 75

Utilizando uma penalização de 80%

#🎯 Acurácia Geral do pysentimiento: 56.54%
#📊 F1-Score: 0.5526

#🔍 Matriz de Confusão:
#         Predito_POS | Predito_NEG | Predito_NEU
#Real_POS:    162     | 12      | 33
#Real_NEG:    35      | 57      | 77
#Real_NEU:    73      | 46      | 140

Pretendo ainda realizar uma nova avaliação com um Dataset consideravelmente maior, para ter resultados melhores.
