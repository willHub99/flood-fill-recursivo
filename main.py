#===============================================================================
# Autores: Bogdan T. Nassu, Eduarda Simonis Gavião (RA: 1879472), Willian Rodrigo Huber(RA: 1992910)
# Universidade Tecnológica Federal do Paraná
#===============================================================================

import sys
import timeit
import numpy as np
import cv2
from operator import itemgetter

#===============================================================================

INPUT_IMAGE =  'arroz.bmp'

NEGATIVO = False
THRESHOLD = 0.8
ALTURA_MIN = 1
LARGURA_MIN = 1
N_PIXELS_MIN = 1

#===============================================================================

def binariza (img, threshold):  
    return np.where(img < threshold, img*0, 1)


#-------------------------------------------------------------------------------

def floodFill(img, y, x, componente):
    #verifica se o pixel está dentro da imagem
    if y < img.shape[0] and y >=0 and x < img.shape[1] and x >= 0:

        #verifica se o pixel atual é fundo
        if img[y][x] == 0 or img[y][x] == componente['label']:
            return

        #armazena a coordenadas do lado direito do retângulo
        componente['R'] = max(componente['R'], x)
        #armazena a coordenadas do lado esquerdo do retângulo
        componente['L'] = min(componente['L'], x)
        #armazena a coordenadas do topo do retângulo
        componente['T'] = min(componente['T'], y)
        #armazena a coordenadas da base do retângulo
        componente['B'] = max(componente['B'], y)

        #pega o label do blob atual
        img[y][x] = componente['label']

        #incrementa a quantidade de pixels do blob
        componente['pixels'].append((y,x))  

        #chamada recursiva pro pixel a direita
        floodFill(img, y, x+1, componente)
        #chamada recursiva pro pixel acima
        floodFill(img, y+1, x, componente)
        #chamada recursiva pro pixel a esquerda
        floodFill(img, y, x-1, componente)
        #chamada recursiva pro pixel abaixo
        floodFill(img, y-1, x, componente)

    
def rotula (img, largura_min, altura_min, n_pixels_min):
    '''Rotulagem usando flood fill. Marca os objetos da imagem com os valores
[0.1,0.2,etc].
    

Parâmetros: img: imagem de entrada E saída.
            largura_min: descarta componentes com largura menor que esta.
            altura_min: descarta componentes com altura menor que esta.
            n_pixels_min: descarta componentes com menos pixels que isso.

Valor de retorno: uma lista, onde cada item é um vetor associativo (dictionary)
com os seguintes campos:


'label': rótulo do componente.
'n_pixels': número de pixels do componente.
'T', 'L', 'B', 'R': coordenadas do retângulo envolvente de um componente conexo,
respectivamente: topo, esquerda, baixo e direita.'''

    #img.shape retorna o tamanho das dimensões x, y e z
    row, col, _ = img.shape
    #cria a lista de dicionários
    componentes = []
    #define a label utilizada para rotular os objetos de interesse
    label = 2
    for y in range(row):
        for x in range(col):
            #garante que somente objetos de interrese sejam lidos
            if img[y][x] == 1:
                componente = {}
                componente['label'] = label
                componente['nPixel'] = 0
                componente['T'] = componente['B'] = y
                componente['L'] = componente['R'] = x
                componente['pixels'] = []
                floodFill(img, y, x, componente)
                label += 1
                componentes.append(componente)

    componentes_filtrados = []
    
    for componente in componentes:
        if len(componente['pixels']) >= n_pixels_min and abs(componente['B'] - componente['T']) >= altura_min  and abs(componente['R'] - componente['L']) >= largura_min:
            componentes_filtrados.append(componente)

    return componentes_filtrados
    

     


#===============================================================================

def main ():

    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.reshape((img.shape [0], img.shape [1], 1))
    img = img.astype(np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Segmenta a imagem.
    if NEGATIVO:
        img = 1 - img
    img = binariza(img, THRESHOLD)
    """ cv2.imshow('01 - binarizada', img)
    cv2.imwrite('01 - binarizada.png', img*255) """

    start_time = timeit.default_timer ()
    componentes = rotula(img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    n_componentes = len(componentes)
    print ('Tempo: %f' % (timeit.default_timer () - start_time))
    print ('%d componentes detectados.' % n_componentes)

    # Mostra os objetos encontrados.
    for c in componentes:
        cv2.rectangle(img_out, (c['L'], c['T']), (c['R'], c['B']), (0,0,1))

    
    
    cv2.imshow('02 - out', img_out)
    cv2.imwrite('02 - out.png', img_out*255)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

#===============================================================================
