# Deep Learning com Keras e Tensor Flow - Part 1

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

Um artigo recente publicado no [medium](https://towardsdatascience.com/which-deep-learning-framework-is-growing-fastest-3f77f14aa318) fez uma comparação entre as principais bibliotecas para Deep Learning e mostrou que Tensorflow é o que mais cresce e o que apresenta a maior demanda.

![enter image description here](https://cdn-images-1.medium.com/max/1600/1*c67KMUJj3waIlxnUJ1enTw.png)

*Resultado do estudo realizado por [Jeff Hale](https://towardsdatascience.com/which-deep-learning-framework-is-growing-fastest-3f77f14aa318)*


E porque então usar Keras? Keras roda em cima do Tensorflow e oferece uma linguagem muito mais amigável, fácil e rápida para escrever os modelos (já mencionamos que é muito mais fácil?)

O Tensorflow anunciou recentemente sua [versão 2.0](https://www.tensorflow.org/alpha), que usa Keras como sua API de alto nível já integrada dentro do `tf.keras`. Portanto, continuaremos usando o Keras como framework para Deep Learning. (Ainda mais com esse novo mascote!)

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

Nosso desafio é criar uma rede para aprender se um raio X é lateral ou de frente. Esse é um dos poucos desafios da área médica que nós, leigos, conseguimos validar visualmente sem o auxílio de um médico:

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
O primeiro passo é instanciar o generator, nesta etapa existem vários parâmetros possíveis tanto para definir o formato da imagem que será lida, quanto para aplicar transformações fazendo image augmentor, todos os parâmetros podem ser consultados [aqui](https://keras.io/preprocessing/image/).
Neste exemplo definimos apenas que a imagem precisa ser normalizada (dividindo por 255) e que 30% dos dados serão separados para validação:
```python
# aqui definimos as transformações que serão aplicadas na imagem e a % de dados 
# que serão usados para validação
data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.30)
```
A próxima etapa é passar quais dados serão lidos. Aqui também há diferentes métodos possíveis, os mais comuns são `flow_from_dataframe` que lê as imagens de acordo com os caminhos específicados em um dataframe, e `flow_from_directory`, que lê os arquivos direto de uma pasta, cada classe deve estar em uma pasta separada:

```python
# para criar os generators precisamos definir o path da pasta raiz com as imagens e o tamanho da BATCH SIZE
path = 'images-chest-orientation/train/'
# dentro da pasta train, há uma pasta frente e outra lateral com suas respectivas imagens

BATCH_SIZE = 50

train_generator = data_generator.flow_from_directory(path, shuffle=True, seed=13,
                                    class_mode='categorical', batch_size=BATCH_SIZE, subset="training")

validation_generator = data_generator.flow_from_directory(path, shuffle=True, seed=13,
                                    class_mode='categorical', batch_size=BATCH_SIZE, subset="validation")
```

Também definimos nesta etapa o `BATCH_SIZE`, ou seja, quantas imagens serão lidas por bloco, se os dados serão lidos de forma aleatória (`shuffle=True`) e o `seed`. Outros parâmetros podem ser vistos na documentação do [keras](https://keras.io/preprocessing/image/#flow_from_directory).

### 2. Definição da arquitetura

O método abaixo define a arquitetura da rede:

```python
model = Sequential()

# primeira camada adiciona o shape do input
# também é possível alterar a inicializacao, bias, entre outros -- https://keras.io/layers/convolutional/#conv2d
model.add(Conv2D(filters=64, kernel_size=2, activation='relu', input_shape=(256,256)))
#Tamanho do downsampling
model.add(MaxPooling2D(pool_size=2))
# Fracao das unidades que serao zeradas
model.add(Dropout(0.3))

# Segunda camada
model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))

# Da um reshape no output transformando em array
model.add(Flatten())

# Camada full-connected 
model.add(Dense(256, activation='relu'))

#Camada de saida com o resultado das classes
model.add(Dense(2, activation='sigmoid'))
````

Os modelos do keras podem ser `Functional API` ou `Sequential`. Quando estamos definindo a nossa rede usamos o `Sequential` para definirmos as nossas camadas de forma sequencial. 
```python
model = Sequential()
```
A primeira camada adicionada neste exemplo é uma convolucional com 64 filtros e dimensão de 2x2, com função de ativação `relu`. Essa é a função de ativação tradicionalmente utilizada nas camadas intermediárias, ela ativa os neurônios que tiveram resultados maiores que 0, para outras funções disponíveis pelo Keras veja [aqui](https://keras.io/activations/). Também é na primeira camada que definimos qual o formato de entrada da rede, no caso passaremos imagens de 256x256:
```python
model.add(Conv2D(filters=64, kernel_size=2, activation='relu', input_shape=(256,256)))
```

Em seguida adicionamos uma camada de `MaxPooling`, uma camada que realiza o *downsampling* calculando o valor máximo de cada *pool* como mostrado na figura, para essa camada precisamos apenas definir o tamanho do *pool*.
```python
model.add(MaxPooling2D(pool_size=2))
```

![enter image description here](https://developers.google.com/machine-learning/practica/image-classification/images/maxpool_animation.gif)
*Exemplo de funcionamento do MaxPooling, Fonte: [Google Developers: ML Practicum: Image Classification](https://developers.google.com/machine-learning/practica/image-classification)*

Depois adicionamos uma camada de `Dropout`, uma das técnicas atualmente mais utilizadas para evitar overfitting. Ele aleatoriamente desativa uma porcentagem de neurônios durante cada época de treinamento. Precisamos apenas definir a porcentagem que queremos desativar.
```python
model.add(Dropout(0.3))
```
 > Cuidado para não exagerar na quantidade de dropouts nem na porcentagem ou acabará gerando *underfitting*

Após as camadas convolucionais precisamos redimensionar as features para 1 dimensão. Aqui faremos isso utilizando uma `Flatten`.
```python
model.add(Flatten())
```
Depois disso adicionamos uma camada `Densa` com 256 neurônios e por fim a câmada de saída com 2 neurônios (um para cada classe), e a função de ativação sigmóide, que retorna a probabilidade da instância ser daquela classe.
```python
model.add(Dense(256, activation='relu'))   
model.add(Dense(2, activation='sigmoid'))
```

Essa foi a arquitetura que nós definimos para o modelo em questão, as camadas usadas e seus respectivos tamanhos são totalmente parametrizáveis, recomendamos testar o modelo com diferentes arquiteturas para analisar os resultados.


### 3. Compilação

Precisamos definir como a rede irá aprender, isto é, qual a função de loss e o otimizador.

```python
# Compila o modelo definindo: otimizador, metrica e loss function
model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
```

Para este exemplo utilizamos o otimizador Adam, com os valores padrões (`lr=0.001, beta_1=0.9, beta_2=0.999`), você pode aprender mais sobre ele [aqui](https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c). 
Para o cálculo do loss usamos a função padrão para classificação binária: `binary_crossentropy`.

### 4. Treinamento
Como lemos os dados usando um generator, o fit do keras também será usando um  `fit_generator`.

Também usaremos alguns  `callbacks`:

-   ModelCheckPoint para salvar o modelo que tiver o melhor loss durante o treinamento e,
-   EarlyStop para interromper o treinamento caso a rede pare de aprender.

```python 
checkpoint = ModelCheckpoint('chest_orientation_model.hdf5', 
                             monitor='val_loss', 
                             verbose=1, mode='min', 
                             save_best_only=True)

early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=0.001,
                                   patience=5,
                                   mode='min',
                                   verbose=1)
```

Definidos os callbacks, vamos ao treinamento em si:

```python
model.fit_generator(generator=train_generator,
                    steps_per_epoch = train_generator.samples//BATCH_SIZE,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples//BATCH_SIZE,
                    epochs= 50,
                    callbacks=[checkpoint, early_stop]
                    )
```
Como faremos a leitura de dados com um `generator`, utilizaremos o `fit_generator` para o treinamento. A definição dos parâmetros é bem simples, como é possível ver no código: passamos os generators, a quantidade de passos que será preciso no treinamento e na validação , isto é, para ler todos os todos (total de dados dividido por tamanho do batch), a quantidade máxima de épocas e os callbacks já criados.

O resultado do treinamento foi o seguinte:

![saida](https://github.com/jessica-santos/qcon_notebook/blob/master/output.png?raw=true)
 Como podemos ver, apesar de termos colocado como máximo 50 épocas, o modelo parou na 10a época devido ao `early_stop`. Conseguimos ver que ele parou mesmo de aprender na época 5, onde atingiu 0.9853 de acurácia na validação.

### 5. Avaliação:

Sempre importante separar uma quantidade de dados para testar o modelo no final. Aqui faremos apenas um teste visual para efeito de demonstração

```python
# Carregando imagens de teste
import glob

test_set = glob.glob('images-chest-orientation/test/**/*.jpg')

# temos que fazer o load do model que teve o melhor loss
model = load_model('chest_orientation_model.hdf5')

image_test = np.array([img_to_array(load_img(image_name, target_size=(256, 256), color_mode='rgb'))/255 for image_name in test_set])

y_pred = model.predict(image_test)
```

Precisamos carregar o modelo salvo. Lembre-se que o modelo que treinou até a última época e estiver em memória não é necessariamente o melhor, o melhor foi salvo pelo `ModelCheckpoint`. 
```python 
model = load_model('chest_orientation_model.hdf5')
```
Depois carregamos as imagens de teste em memória usando os métodos do próprio keras:
```python
image_test = np.array([img_to_array(load_img(image_name, target_size=(256, 256), color_mode='rgb'))/255 for image_name in test_set])
```
E, por fim, fazemos a predição para estas imagens:
```python
y_pred = model.predict(image_test)
```

Com as predições e os valores reais podemos calcular as métricas necessárias para validar nosso modelo. Geralmente utilizamos uma matriz de confusão ou calculamos métricas como precisão e sensibilidade, que são mais ou menos importantes de acordo com o problema. Porém, a fim de ilustrar melhor os resultados, iremos apenas visualizar os resultados para cada imagem.

```python
y_true = [0,0,0,0,0,1,1,1,1,1]
labels = ['Frente', 'Lateral']
figure = plt.figure(figsize=(20, 8))
for i in range(10):
    ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    # Display each image
    im = plt.imread(test_set[i])
    ax.imshow(im)
    predict_index = np.argmax(y_pred[i])
    true_index = y_true[i]
    # Set the title for each image
    ax.set_title("{} ({})".format(labels[predict_index], 
                                  labels[true_index]),
                                  color=("green" if predict_index == true_index else "red"))
 ```
 ![enter image description here](https://github.com/jessica-santos/qcon_notebook/blob/master/result.png?raw=true)
Como podemos perceber, o modelo acerta bem quase todas as imagens separadas para teste, errando apenas a terceira imagem, provavelmente por estar de cabeça para baixo. Se quisemos corrigir esse tipo de erro poderíamos usar o image augmentor e gerar mais algumas imagens em diferentes posições.

<!--stackedit_data:
eyJoaXN0b3J5IjpbNTM2NTAyODMxLC0xOTUwMTYzMzQzLC0xMD
Y4NTQyODgxLC0xNTM1MjM0ODEzLDE2Mzk5ODA4NCwxMDI2NzE4
ODYyLDE4NzIzMTY0MDBdfQ==
-->