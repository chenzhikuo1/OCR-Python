__author__ = 'Wilson Junior'

import scipy.ndimage as nd
import teste
import scipy.misc as ms

class wocr:
    imagem = "";

    def __init__(self,imagem):
        self.imagem = nd.imread(imagem,1);
        self.imagem = 254 - self.imagem;
        self.imagem = nd.grey_dilation(self.imagem,2);
        self.imagem = self.prepara_imagem();


    def regiao(self,xi,xf,yi,yf):
        contador = 0;
        for i in range(xi,xf):
            for j in range(yi,yf):
                if(self.imagem[i,j] == 0):
                    contador = contador + 1;
        return float(contador);


    def prepara_imagem(self):
        x,y = self.imagem.shape;
        for i in range(0,x):
            for j in range(0,y):
                if(self.imagem[i,j] < 150):
                    self.imagem[i,j] = 0;
                else:
                    self.imagem[i,j] = 255;
        return self.imagem;

    def atributos(self):
        atb = [];
        x,y = self.imagem.shape;
        atb.append(self.regiao(0,x/4,0,y/4));
        atb.append(self.regiao(x/4,x/2,0,y/4));
        atb.append(self.regiao(x/2,(3*x)/4,0,y/4));
        atb.append(self.regiao((3*x)/4,x,0,y/4));

        atb.append(self.regiao(0,x/4,y/4,y/2));
        atb.append(self.regiao(x/4,x/2,y/4,y/2));
        atb.append(self.regiao(x/2,(3*x)/4,y/4,y/2));
        atb.append(self.regiao((3*x)/4,x,y/4,y/2));

        atb.append(self.regiao(0,x/4,y/2,(3*y)/4));
        atb.append(self.regiao(x/4,x/2,y/2,(3*y)/4));
        atb.append(self.regiao(x/2,(3*x)/4,y/2,(3*y)/4));
        atb.append(self.regiao((3*x)/4,x,y/2,(3*y)/4));

        atb.append(self.regiao(0,x/4,(3*y)/4,y));
        atb.append(self.regiao(x/4,x/2,(3*y)/4,y));
        atb.append(self.regiao(x/2,(3*x)/4,(3*y)/4,y));
        atb.append(self.regiao((3*x)/4,x,(3*y)/4,y));

        atb.append(teste.distancia_borda_esquerda(self.imagem));
        atb.append(teste.distancia_borda_topo(self.imagem));
        atb.append(teste.distancia_borda_direita(self.imagem));
        atb.append(teste.distancia_borda_baixo(self.imagem));

        return atb;


import os
import glob
import numpy as np
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer

os.chdir("../imagens/");
contador = 0;
atributos = [];
for file in glob.glob("*.jpg"):
    obj = wocr(file);
    atributos.append(obj.atributos());

maximo_entrada = max(max(atributos));
atributos = np.matrix(atributos);
atributos = atributos/maximo_entrada;

saidas = [float(0),float(0),float(0),float(0),float(0),float(0),float(0),float(0),float(0),
          float(5),float(5),float(5),float(5),float(5),float(5),float(5),float(5),float(5),float(5),float(5),
          float(10),float(10),float(10),float(10),float(10),float(10),float(10),float(10),float(10),float(10),float(10),
          float(15),float(15),float(15),float(15),float(15),float(15),float(15),float(15),float(15),float(15),float(15),float(15),
          float(20),float(20),float(20),float(20),float(20),float(20),float(20),float(20),float(20),float(20),
          float(25),float(25),float(25),float(25),float(25),float(25),float(25),float(25),float(25),float(25),float(25),
          float(30),float(30),float(30),float(30),float(30),float(30),float(30),float(30),float(30),float(30),
          float(35),float(35),float(35),float(35),float(35),float(35),float(35),float(35),float(35),float(35),
          float(40),float(40),float(40),float(40),float(40),float(40),float(40),float(40),float(40),float(40),
          float(45),float(45),float(45),float(45),float(45),float(45),float(45),float(45),float(45),float(45)];

maxima_saida = max(saidas);
saidas = np.matrix(saidas);
saidas = saidas/maxima_saida;

parametros_entrada = SupervisedDataSet(20,1);

i = 0;
for entrada in atributos:
    parametros_entrada.addSample(entrada,[saidas[0,i]]);
    i = i + 1;

rede_neural = buildNetwork(20,15,1,bias=True);
rede_neural.randomize();
treinamento = BackpropTrainer(rede_neural,parametros_entrada,momentum=0.99);
treinamento.trainEpochs(1000);

obj = wocr("../imagens_teste/um.jpg");

atributos_teste = obj.atributos();
atributos_teste = np.matrix(atributos_teste) / maximo_entrada;

caracteres = ['0','1','2','3','4','5','6','7','8','9'];
valores_desejaveis = [];
for i in range(0,50,5):
    valores_desejaveis.append(float(i));
valores_desejaveis = np.matrix(valores_desejaveis);
valores_desejaveis = valores_desejaveis / 45;
for indice in range(0,9):
    simular = SupervisedDataSet(20,1);
    simular.addSample(atributos_teste,valores_desejaveis[0,indice]);
    erro = treinamento.testOnData(simular,True);
    if(erro <= 0.005):
        print(caracteres[indice]);
        break;
