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
        #regiao 4x4
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


        #distancias de bordas
        atb.append(teste.distancia_borda_esquerda(self.imagem));
        atb.append(teste.distancia_borda_topo(self.imagem));
        atb.append(teste.distancia_borda_direita(self.imagem));
        atb.append(teste.distancia_borda_baixo(self.imagem));

        #indice de circularidade
        atb.append(teste.indice_circularidade(self.imagem));

        #altura media da imagem
        atb.append(teste.media_altura(self.imagem));

        return atb;
