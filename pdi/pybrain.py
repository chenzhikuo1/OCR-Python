__author__ = 'Wilson Junior'
import pybrain
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer

# UTILIZANDO REDES NEURAIS PARA SIMULAR PORTA LOGICA AND

# DEFININDO QUANTAS ENTRADAS E QUANTAS SAIDAS A REDE NEURAL DEVE POSSUIR
parametros = SupervisedDataSet(2,1);

# DEFININDO PARAMETROS DE ENTRADA E SAIDA
parametros.addSample([0,0],[0]);
parametros.addSample([0,1],[0]);
parametros.addSample([1,0],[0]);
parametros.addSample([1,1],[1]);

# CONSTRUINDO A REDE NEURAL
# 2 - PARAMETROS DE ENTRADA
# 10 - NEURONIOS NA CAMADA INTERMEDIARIA
# 1 - SAIDA
rede_neural = buildNetwork(2,10,1,bias=True,outputbias=True);

# TREINANDO A REDE NEURAL
treinamento = BackpropTrainer(rede_neural,parametros,momentum=0.5);
treinamento.trainEpochs(1000);

# SIMULANDO A REDE NEURAL
parametros_teste = SupervisedDataSet(2,1);

# DEFININDO PARAMETROS DE ENTRADA E SAIDA
parametros_teste.addSample([0,0],[0]);
treinamento.testOnData(parametros_teste,True);