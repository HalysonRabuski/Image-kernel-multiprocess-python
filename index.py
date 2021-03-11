from skimage.io import imread, imsave
from multiprocessing import Pool
import skimage
import numpy as np
import math

im = imread("images/halyson.jpg", as_gray=True)
numrows = len(im)
numcols = len(im[0])

def convolucao(i, j):
    n1 = get_value(i-1,j-1)
    n2 = get_value(i-1,j)
    n3 = get_value(i-1,j+1)
    n4 = get_value(i,j-1)
    n5 = get_value(i,j)
    n6 = get_value(i,j+1)
    n7 = get_value(i+1,j-1)
    n8 = get_value(i+1,j)
    n9 = get_value(i+1,j+1)
    # kernel = [[1,0,-1],
    #          [0,0,0],     #Bordas
    #          [-1,0,1]]
    kernel = [[-1,-1,-1],
             [-1,8,-1],   #Bordas 2
             [-1,-1,-1]]
    # kernel = [[1,2,1],
    #          [2,4,2],   #Blur gauss
    #          [1,2,1]]
    # kernel = [[0,-1,0],
    #          [-1,5,-1],     #sharpen
    #          [0,-1,0]]

    result = ((kernel[0][0]*n1) + (kernel[0][1]*n2) + (kernel[0][2]*n3)
             +(kernel[1][0]*n4) + (kernel[1][1]*n5) + (kernel[1][2]*n6) 
             + (kernel[2][0]*n7) + (kernel[2][1]*n8) + (kernel[2][2]*n9))

    return result

def get_value(i, j):
    try:
        return im[i][j]
    except IndexError:
        return 0

if __name__ == '__main__':
    import multiprocessing as mp
    import itertools

    num_cores = mp.cpu_count()
    pool = Pool(processes=num_cores)
    chunksize = math.floor((numrows * numcols) / 8)

    print('possui ' + str(numrows * numcols) + " partes")
    print('vai começar, com um chunksize de: ', chunksize)

    results = pool.starmap(convolucao, itertools.product(range(0, numrows), range(0, numcols)), chunksize=chunksize)

    print('acabou o multi processamento')
    
    final = np.array(results)
    nova = final.reshape(numrows,numcols)

    data = nova.astype(np.float64) / nova.max()

    img = skimage.img_as_ubyte(data)

    imsave("output/halyson.jpg", img)
