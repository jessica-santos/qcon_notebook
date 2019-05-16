# Deep Learning com Keras e Tensor Flow

> Entre o dias 06 e 08 de maio de 2019 participamos da [QCon SP](https://qconsp.com/). Um dos maiores eventos de tecnologia do Brasil, com maior público senior e mais de 80 palestras técnicas. Nele apresentamos a palestra *Deep Learning com Keras e Tensor Flow*, em breve disponível na [InfoQ](https://www.infoq.com/br/conferences/qconsp2019) e descrita neste post.


Nós trabalhamos na [NeuralMed](neuralmed.ai), uma startup que usa Inteligência Artificial para auxiliar no diagnóstico de imagens médicas. Portanto, usar Deep Learning faz parte do nosso dia-a-dia, e é a base desse conhecimento que gostaríamos de compartilhar. Os tópicos principais que cobriremos são:

- Como construir uma Convolutional Neural Network (CNN) usando Keras + Tensorflow

- Como e porque usar TransferLearning

- Alguns pontos de atenção e desafios encontrados

Por trabalharmos com imagens médicas esse será nosso caso de uso, mais especificamente imagens de raio-x de tórax. Porém, todas as redes construídas podem ser aplicadas em outros tipos de imagens, como classificação de gatos e cachorros, por exemplo.

<img src=https://upload.wikimedia.org/wikipedia/commons/a/a1/Normal_posteroanterior_%28PA%29_chest_radiograph_%28X-ray%29.jpg width=350/>

*Exemplo de raio-x de tórax*


# Por que Tensorflow + Keras?

-   Continuam sendo as bibliotecas mais usadas e mais pesquisadas para Deep Learning:

Um artigo recente publicado no [medium]([https://towardsdatascience.com/which-deep-learning-framework-is-growing-fastest-3f77f14aa318](https://towardsdatascience.com/which-deep-learning-framework-is-growing-fastest-3f77f14aa318)) fez uma comparação entre as principais bibliotecas para Deep Learning e mostrou que Tensorflow é o que mais cresce e o que apresenta a maior demanda.

![enter image description here](https://cdn-images-1.medium.com/max/1600/1*c67KMUJj3waIlxnUJ1enTw.png)
*Resultado do estudo realizado por [Jeff Hale]([https://towardsdatascience.com/which-deep-learning-framework-is-growing-fastest-3f77f14aa318)*

E porque então usar Keras? Keras roda em cima do Tensorflow e oferece uma linguagem muito mais amigável, fácil e rápida para escrever os modelos (já mencionamos que é muito mais fácil?)

O Tensorflow anunciou recentemente sua versão [2.0](https://www.tensorflow.org/alpha), que usa Keras como sua API de alto nível já integrada dentro do `tf.keras`. Portanto, continuarem usando o Keras como framework para Deep Learning. (Ainda mais com esse novo mascote!)

<img src=https://pbs.twimg.com/media/DtgmEDSV4AAIjLm.jpg:small width=300/>

*Keras mascote: Prof Keras*


## Convolutional Neural Network

CNN é uma categoria de redes de deep learning normalmente aplicadas para reconhecimento de imagens. 

### Como elas funcionam?

São baseadas principalmente em Filtros Convolucionais que extraem características de imagens. Esses filtros já são usados há muito tempos na área de Processamento de Imagens, a ideia é que ao aplicar essas matrizes sobre a imagem se consiga obter as características desejadas. O gif abaixo mostra os principais filtros (ou kernels) para extração de bordas.


<img src=https://camo.githubusercontent.com/7513873c1d99957d9604d134564d7fc61d321fa4/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f313030302f312a54416f336173656c4a4e5677724c4c723635344d79672e676966 width=500/>

*Fonte: [Gentle Dive into Math Behind Convolutional Neural Networks](https://towardsdatascience.com/gentle-dive-into-math-behind-convolutional-neural-networks-79a07dd44cf9)*

Nas redes convolucionais esses filtros são aplicados em várias camadas. Além disso, os valores dos filtros são aprendidos, portanto a própria rede aprende quais características são relevantes para o problema em questão.


<img src=https://camo.githubusercontent.com/385e37807ce5f3edce3890c52e2cc5cfed2bd808/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f313630302f312a5f333445747267596b366351786c4a326272353148512e676966 width=500/>

*Fonte: [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/convolutional-networks/)*

Vamos a prática!

## Construindo uma CNN pra predizer lateralidade do raio-X

Nosso desafio é criar uma rede para aprender se um raio X é lateral ou de frente:

![](https://github.com/nandaw/qcon_notebook/raw/b237348ccff36ef09e1c4f3c76c4340fa942574c/lateral.png)

Para facilitar vamos dividir a criação da rede em 5 etapas:
1. Input de dados
2. Definição da arquitetura
3. Compilação do Modelo
4. Treinamento
5. Validação

Antes de ver cada uma delas, vamos fazer todos os imports necessários para o funcionamento do código:

```python
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import time
import os
import datetime

from keras import datasets, Model
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras import Sequential
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import backend
import tensorflow as tf

from keras.preprocessing.image import load_img, img_to_array
```

### 1. Input de Dados

Existem várias formas de inputar os dados para treinamento, vamos usar o `Image Data Generator` para ler as imagens a partir do disco.
Vamos primeiramente instanciar o generator, nesta etapa existem vários parâmetros possíveis tanto para definir formato da imagem que será lida, quanto para aplicar transformações fazendo image augmentation, todos os parâmetros podem ser vistos [aqui](https://keras.io/preprocessing/image/).
Neste exemplo colocamos apenas que a imagem precisa ser normalizada (dividindo por 255) e que 30% dos dados serão separados para validação:
```python
# aqui definimos as transformações que serão aplicadas na imagem e a % de dados 
# que serão usados para validação
data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.30)
```
A próxima etapa é passar quais dados serão lidos. Aqui também há diferentes métodos possíveis, os mais comuns são `flow_from_dataframe` que lê as imagens de acordo com os caminhos específicados em um dataframe, e `flow_from_directory`, que lê os arquivos direto de uma pasta, cada classe deve estar em uma pasta separada. 


<!--stackedit_data:
eyJoaXN0b3J5IjpbNTUzODAwMDg4LC0xMTAyMzM3MzYwXX0=
-->