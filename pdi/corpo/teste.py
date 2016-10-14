from numpy.core.fromnumeric import mean

__author__ = 'Wilson Junior'

import numpy as np
import scipy.ndimage as nd
import scipy.misc as ms
import numpy as np

def distancia(x1,y1,x2,y2):
    return np.sqrt( np.power(y2-y1,2) + np.power(x2-x1,2) );

def prepara_imagem(imagem):
        x,y = imagem.shape;
        for i in range(0,x):
            for j in range(0,y):
                if(imagem[i,j] < 180):
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



def calcula_area(imagem):
    x,y = imagem.shape;
    contador = 0;
    for i in range(0,x):
        for j in range(0,y):
            if(imagem[i,j] == 0):
                contador = contador + 1;
    return float(contador);

def calcula_perimetro(imagem):
    imagem_borda = nd.grey_erosion(imagem,3);
    imagem_borda = imagem_borda - imagem;
    x,y = imagem_borda.shape;
    contador = 0;
    for i in range(0,x):
        for j in range(0,y):
            if(imagem_borda[i,j] == 0):
                contador = contador + 1;
    return float(contador);

def indice_circularidade(imagem):
    area = calcula_area(imagem);
    perimetro = calcula_perimetro(imagem);
    return 12.57*area/(perimetro*perimetro);

def percorre(imagem,xi,xf,step1,yi,yf,step2):
    for i in range(xi,xf,step1):
        for j in range(yi,yf,step2):
            if(imagem[i,j] == 0):
                return [i,j];

def media_altura(imagem):
    dsup = percorre(imagem,0,25,1,0,25,1);
    dinf = percorre(imagem,24,0,-1,24,0,-1);
    dsup2 = percorre(imagem,0,25,1,24,0,-1);
    dinf2 = percorre(imagem,24,0,-1,0,25,1);
    dist1 = distancia(dsup[0],dsup[1],dinf[0],dinf[1]);
    dist2 = distancia(dsup2[0],dsup2[1],dinf2[0],dinf2[1]);
    return (dist1+dist2)/2;

def hist_horizontal(imagem):
    x,y = imagem.shape;
    lista = [];
    for i in range(0,x):
        contador = 0;
        for j in range(0,y):
            if(imagem[i,j] == 0):
                contador = contador + 1;
        lista.append(float(contador));
    return max(lista);

def hist_vertical(imagem):
    x,y = imagem.shape;
    lista = [];
    for i in range(0,x):
        contador = 0;
        for j in range(0,y):
            if(imagem[j,i] == 0):
                contador = contador + 1;
        lista.append(float(contador));
    return max(lista);
import glob
imagem = nd.imread("../imagens_teste/um.jpg",1);
imagem = 254 - imagem;
imagem = nd.grey_dilation(imagem,2);
imagem = prepara_imagem(imagem);
print(hist_horizontal(imagem),hist_vertical(imagem));
for arq in glob.glob("../imagens/1*.jpg"):
    imagem = nd.imread(arq,1);
    imagem = 254 - imagem;
    imagem = nd.grey_dilation(imagem,2);
    imagem = prepara_imagem(imagem);
    print(hist_horizontal(imagem),hist_vertical(imagem));