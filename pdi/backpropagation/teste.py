__author__ = 'Wilson Junior'

import numpy as np
import scipy.ndimage as nd

def distancia(x1,y1,x2,y2):
    return np.sqrt( np.power(y2-y1,2) + np.power(x2-x1,2) );

def prepara_imagem(imagem):
        x,y = imagem.shape;
        for i in range(0,x):
            for j in range(0,y):
                if(imagem[i,j] < 150):
                    imagem[i,j] = 0;
                else:
                    imagem[i,j] = 255;
        return imagem;

def distancia_borda_esquerda(imagem):
    x,y = imagem.shape;
    posicoes = [];
    for i in range(0,x):
        indice = -1;
        for j in range(0,y):
            if(imagem[i,j]==0 and indice == -1):
                posicoes.append(distancia(i,0,i,j));
                indice = 10;
    return min(posicoes);

def distancia_borda_direita(imagem):
    x,y = imagem.shape;
    posicoes = [];
    for i in range(0,x):
        j = y-1;
        indice = -1;
        while(j>=0):
            if(imagem[i,j] == 0 and indice == -1):
                posicoes.append(distancia(i,y,i,j));
                indice = 10;
            j = j - 1;
    return min(posicoes);

def distancia_borda_topo(imagem):
    x,y = imagem.shape;
    for i in range(0,x):
        for j in range(0,y):
            if(imagem[i,j] == 0):
                return distancia(0,j,i,j);

def distancia_borda_baixo(imagem):
    x,y = imagem.shape;
    for i in range(x-1,0,-1):
        for j in range(y-1,0,-1):
            if(imagem[i,j] == 0):
                return distancia(x,j,i,j);


def  distancia_borda_esquerda_topo(imagem):
    x,y = imagem.shape;
    for i in range(0,x):
        for j in range(0,y):
            if(imagem[i,j] == 0):
                return int(distancia(0,0,i,j));

def distancia_borda_direita_topo(imagem):
    x,y = imagem.shape;
    for i in range(0,x):
        for j in range(y-1,0,-1):
            if(imagem[i,j]==0):
                return int(distancia(0,y,i,j));


def distancia_borda_direita_baixo(imagem):
    x,y = imagem.shape;
    for i in range(x-1,0,-1):
        for j in range(y-1,0,-1):
            if(imagem[i,j] == 0):
                return int(distancia(x,y,i,j));

def distancia_borda_esquerda_baixo(imagem):
    x,y = imagem.shape;
    for i in range(x-1,0,-1):
        for j in range(0,y):
            if(imagem[i,j] == 0):
                return int(distancia(x,0,i,j));

def conta_pixels(imagem):
    x,y = imagem.shape;
    contador = 0;
    for i in range(0,x):
        for j in range(0,y):
            if(imagem[i,j] == 0):
                contador = contador + 1;
    return contador;

def indice_circularidade(imagem):
    imagem = prepara_imagem(imagem);
    area = conta_pixels(imagem);
    ax = nd.sobel(imagem,0);
    ay = nd.sobel(imagem,1);
    imagem = np.hypot(ax,ay);
    imagem = nd.grey_erosion(imagem,2);
    imagem = prepara_imagem(imagem);
    perimetro = conta_pixels(imagem);
    return (12.57*area)/(perimetro**2);

