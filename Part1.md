# Deep Learning com Keras e Tensor Flow

> Entre o dias 06 e 08 de maio de 2019 participamos da [QCon SP](https://qconsp.com/). Um dos maiores eventos de tecnologia do Brasil, com maior público senior e mais de 80 palestras técnicas. Nele apresentamos a palestra *Deep Learning com Keras e Tensor Flow*, em breve disponível na [InfoQ](https://www.infoq.com/br/conferences/qconsp2019) e descrita neste post.


Nós trabalhamos na [NeuralMed](neuralmed.ai), uma startup que usa Inteligência Artificial para auxiliar no diagnóstico de imagens médicas. Portanto, usar Deep Learning faz parte do nosso dia-a-dia, e é a base desse conhecimento que gostaríamos de compartilhar. Os tópicos principais que cobriremos são:

- Como construir uma Convolutional Neural Network (CNN) usando Keras + Tensorflow

- Como e porque usar TransferLearning

- Alguns pontos de atenção e desafios encontrados

Por trabalharmos com imagens médicas, esse será nosso caso de uso, mais especificamente imagens de raio-x de tórax. Porém, todas as redes construídas podem ser aplicadas em outros tipos de imagens, como classificação de gatos e cachorros, por exemplo.

![Exemplo de Raio-X de tórax](https://upload.wikimedia.org/wikipedia/commons/a/a1/Normal_posteroanterior_%28PA%29_chest_radiograph_%28X-ray%29.jpg)
*Exemplo de raio-x de tórax*

# Por que Tensorflow + Keras?

-   Continuam sendo as bibliotecas mais usadas e mais pesquisadas para Deep Learning:

Um artigo recente publicado no [medium]([https://towardsdatascience.com/which-deep-learning-framework-is-growing-fastest-3f77f14aa318](https://towardsdatascience.com/which-deep-learning-framework-is-growing-fastest-3f77f14aa318)) fez uma comparação entre as principais bibliotecas para Deep Learning e mostrou que Tensorflow é o que mais cresce e o que apresenta a maior demanda.

![enter image description here](https://cdn-images-1.medium.com/max/1600/1*c67KMUJj3waIlxnUJ1enTw.png)
*Resultado do estudo realizado por [Jeff Hale]([https://towardsdatascience.com/which-deep-learning-framework-is-growing-fastest-3f77f14aa318)

E porque então usar Keras? Keras roda em cima do Tensorflow e de vantagens possui uma linguagem muito mais amigável, fácil e rápido de escrever os modelos (muito mais fácil!). 
O Tensorflow anunciou recentemente sua versão [2.0](https://www.tensorflow.org/alpha), que usa Keras como sua API de alto nível já integrada dentro do `tf.keras`. Portanto, continuarem usando o Keras como framework para Deep Learning. (Ainda mais com esse novo mascote!)

![enter image description here](https://pbs.twimg.com/media/DtgmEDSV4AAIjLm.jpg:small)
*Keras mascote: Prof Keras*

## Convolutional Neural Network

### Como elas funcionam?

-   Filtros convolucionais são usados para extrair features de imagens.

![drawing](https://camo.githubusercontent.com/7513873c1d99957d9604d134564d7fc61d321fa4/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f313030302f312a54416f336173656c4a4e5677724c4c723635344d79672e676966)

Nas redes convolucionais esses filtros são aplicados em várias camadas:

![](https://camo.githubusercontent.com/385e37807ce5f3edce3890c52e2cc5cfed2bd808/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f313630302f312a5f333445747267596b366351786c4a326272353148512e676966)

Os valores dos filtros são aprendidos, portanto a própria rede aprende quais características são relevantes.
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTg4MDk5NzgwMl19
-->