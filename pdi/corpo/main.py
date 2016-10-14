__author__ = 'Wilson Junior'

# ALGORITMO DE RECONHECIMENTO OPTICO DE CARACTERES
# SETEMBRO/OUTUBRO 2016 SOBRAL CE
# UNIVERSIDADE FEDERAL DO CEARA
# PROCESSAMENTO DIGITAL DE IMAGENS
# REDES NEURAIS ARTIFICIAIS - BACKPROPAGATION

import WilsonPDI
import teste
import scipy.ndimage as nd
import numpy as np
from pybrain.supervised import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
import glob

# BUSCANDO LOCAL DAS IMAGENS
atributos = [];
for imagem in glob.glob("../imagens/*.jpg"):
    objeto = WilsonPDI.wocr(imagem);
    atributos.append(objeto.atributos());

maxima_entrada = max(max(atributos));
atributos = np.matrix(atributos);

dados_entrada_normatizados = atributos/maxima_entrada;

saidas = [];
saidas.append([0,0,0,0]);saidas.append([0,0,0,0]);saidas.append([0,0,0,0]);saidas.append([0,0,0,0]);saidas.append([0,0,0,0]);saidas.append([0,0,0,0]);saidas.append([0,0,0,0]);saidas.append([0,0,0,0]);saidas.append([0,0,0,0]);
saidas.append([0,0,0,1]);saidas.append([0,0,0,1]);saidas.append([0,0,0,1]);saidas.append([0,0,0,1]);saidas.append([0,0,0,1]);saidas.append([0,0,0,1]);saidas.append([0,0,0,1]);saidas.append([0,0,0,1]);saidas.append([0,0,0,1]);saidas.append([0,0,0,1]);saidas.append([0,0,0,1]);
saidas.append([0,0,1,0]);saidas.append([0,0,1,0]);saidas.append([0,0,1,0]);saidas.append([0,0,1,0]);saidas.append([0,0,1,0]);saidas.append([0,0,1,0]);saidas.append([0,0,1,0]);saidas.append([0,0,1,0]);saidas.append([0,0,1,0]);saidas.append([0,0,1,0]);saidas.append([0,0,1,0]);
saidas.append([0,0,1,1]);saidas.append([0,0,1,1]);saidas.append([0,0,1,1]);saidas.append([0,0,1,1]);saidas.append([0,0,1,1]);saidas.append([0,0,1,1]);saidas.append([0,0,1,1]);saidas.append([0,0,1,1]);saidas.append([0,0,1,1]);saidas.append([0,0,1,1]);saidas.append([0,0,1,1]);saidas.append([0,0,1,1]);
saidas.append([0,1,0,0]);saidas.append([0,1,0,0]);saidas.append([0,1,0,0]);saidas.append([0,1,0,0]);saidas.append([0,1,0,0]);saidas.append([0,1,0,0]);saidas.append([0,1,0,0]);saidas.append([0,1,0,0]);saidas.append([0,1,0,0]);saidas.append([0,1,0,0]);
saidas.append([0,1,0,1]);saidas.append([0,1,0,1]);saidas.append([0,1,0,1]);saidas.append([0,1,0,1]);saidas.append([0,1,0,1]);saidas.append([0,1,0,1]);saidas.append([0,1,0,1]);saidas.append([0,1,0,1]);saidas.append([0,1,0,1]);saidas.append([0,1,0,1]);
saidas.append([0,1,1,0]);saidas.append([0,1,1,0]);saidas.append([0,1,1,0]);saidas.append([0,1,1,0]);saidas.append([0,1,1,0]);saidas.append([0,1,1,0]);saidas.append([0,1,1,0]);saidas.append([0,1,1,0]);saidas.append([0,1,1,0]);saidas.append([0,1,1,0]);
saidas.append([0,1,1,1]);saidas.append([0,1,1,1]);saidas.append([0,1,1,1]);saidas.append([0,1,1,1]);saidas.append([0,1,1,1]);saidas.append([0,1,1,1]);saidas.append([0,1,1,1]);saidas.append([0,1,1,1]);saidas.append([0,1,1,1]);saidas.append([0,1,1,1]);
saidas.append([1,0,0,0]);saidas.append([1,0,0,0]);saidas.append([1,0,0,0]);saidas.append([1,0,0,0]);saidas.append([1,0,0,0]);saidas.append([1,0,0,0]);saidas.append([1,0,0,0]);saidas.append([1,0,0,0]);saidas.append([1,0,0,0]);saidas.append([1,0,0,0]);
saidas.append([1,0,0,1]);saidas.append([1,0,0,1]);saidas.append([1,0,0,1]);saidas.append([1,0,0,1]);saidas.append([1,0,0,1]);saidas.append([1,0,0,1]);saidas.append([1,0,0,1]);saidas.append([1,0,0,1]);saidas.append([1,0,0,1]);saidas.append([1,0,0,1]);
saidas = np.matrix(saidas);


contador = 0;
parametros_entrada = SupervisedDataSet(22,4);

for entrada in dados_entrada_normatizados:
    parametros_entrada.addSample(entrada,saidas[contador]);
    contador = contador + 1;

rede_neural = buildNetwork(22,10,4,bias=True,outputbias=True);

treinamento = BackpropTrainer(rede_neural,parametros_entrada,momentum=0.9);

treinamento.trainEpochs(1000);

objeto = WilsonPDI.wocr("../imagens_teste/um.jpg");

atributos_teste = [];
atributos_teste.append(objeto.atributos());
atributos_teste = np.matrix(atributos_teste);
atributos_teste = atributos_teste / maxima_entrada;

saidas = [];
saidas.append([0,0,0,0]);
saidas.append([0,0,0,1]);
saidas.append([0,0,1,0]);
saidas.append([0,0,1,1]);
saidas.append([0,1,0,0]);
saidas.append([0,1,0,1]);
saidas.append([0,1,1,0]);
saidas.append([0,1,1,1]);
saidas.append([1,0,0,0]);
saidas.append([1,0,0,1]);

caracteres = ['0','1','2','3','4','5','6','7','8','9'];
contador = -1;
for saida_desejavel in saidas:
    contador = contador + 1;
    teste = SupervisedDataSet(22,4);
    teste.addSample(atributos_teste,saida_desejavel);
    erro = treinamento.testOnData(teste,True);
    if(erro <= 0.059):
        print(caracteres[contador]);
        break;

