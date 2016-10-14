__author__ = 'Wilson Junior'


import numpy as np
import matplotlib.pyplot as plt

#INFORMACOES DO ALGORITMO
# A - 0 0
# B - 0 1
# 1 - 1 0
# 2 - 1 1

entrada = np.matrix([(0,0),(0,1),(1,0),(1,1)]);

saida = np.matrix([1,1,0,0]);

pesos = np.matrix(np.zeros((1,6)));

bias = np.matrix(np.zeros((1,3)));

taxa_aprendizagem = 0.3;

limite = 0.001;

iteracoes = 20;

erro = [];

erro_calculado = 1;
indice = 0;

while(indice<iteracoes or erro_calculado > limite):
    for i in range(0,4):
        neuronio1 = entrada[i,0] * pesos[0,0] + entrada[i,1] * pesos[0,1] + bias[0,0];
        neuronio2 = entrada[i,0] * pesos[0,2] + entrada[i,1] * pesos[0,3] + bias[0,1];
        y = neuronio1 * pesos[0,4] + neuronio2 * pesos[0,5] + bias[0,2];
        erro_calculado = saida[0,i] - y;
        erro.append(erro_calculado);
        bias[0,2] = bias[0,2] + taxa_aprendizagem * erro_calculado;
        bias[0,1] = bias[0,1] + taxa_aprendizagem * erro_calculado;
        bias[0,0] = bias[0,0] + taxa_aprendizagem * erro_calculado;
        pesos[0,5] = pesos[0,5] + taxa_aprendizagem*neuronio2*erro_calculado;
        pesos[0,4] = pesos[0,4] + taxa_aprendizagem*neuronio1*erro_calculado;
        pesos[0,3] = pesos[0,3] + taxa_aprendizagem*entrada[i,1]*erro_calculado;
        pesos[0,2] = pesos[0,2] + taxa_aprendizagem*entrada[i,0]*erro_calculado;
        pesos[0,1] = pesos[0,1] + taxa_aprendizagem*entrada[i,1]*erro_calculado;
        pesos[0,0] = pesos[0,0] + taxa_aprendizagem*entrada[i,0]*erro_calculado;
    indice = indice + 1;

plt.plot(erro);
plt.show();


def simula(x1,x2):
    neuronio1 = x1 * pesos[0,0] + x2 * pesos[0,1] + bias[0,0];
    neuronio2 = x1 * pesos[0,2] + x2 * pesos[0,3] + bias[0,1];
    y = neuronio1 * pesos[0,4] + neuronio2 * pesos[0,5] + bias[0,2];
    print(y);

print(simula(0,1));