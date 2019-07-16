import os
import gzip
import numpy as np
import time
import matplotlib

# matplotlib.use('agg')

import matplotlib.pyplot as plt
import matplotlib.rcsetup as rcsetup
print(rcsetup.all_backends)

#type(forecast)=<class 'pandas.core.frame.DataFrame'>


def extract_train_img_data(file):
    image_list = []
    with gzip.open(file) as bytestream:
        
        head_bytes = bytestream.read(16)
        head = np.frombuffer(head_bytes, dtype = '>u4')
        
        for i in range(head[1]):
            image_list.append(bytestream.read(head[2] * head[3]))
        # images = bytestream.read(28 * 28)

    return head, image_list


def load_data(path, file):
    
    print('load fashion minst')

    if not os.path.exists(os.path.join(data_path, i)):
        print(os.path.join(data_path, i) + 'is not exists.')

    return 0


def show_image(image):
    plt.figure()
    for i in range(1,33):
        plt.subplot(4,8,i)
        plt.imshow(image[i - 1].reshape(28, 28))
        plt.xticks([])
        plt.yticks([])
    plt.show()


if __name__ == "__main__":

    data_path = '/mnt/d/git/python/ml/data/fashion'

    file_list = ['t10k-images-idx3-ubyte.gz',
                't10k-labels-idx1-ubyte.gz',
                'train-images-idx3-ubyte.gz',
                'train-labels-idx1-ubyte.gz']
    
    for i in file_list:
        load_data(data_path, i)
        
    labels, image = extract_train_img_data(os.path.join(data_path, 'train-images-idx3-ubyte.gz'))
    # print(image)
    print(type(image))
    
    label_array = np.frombuffer(labels, dtype = '>u4')
    print(label_array)
    
    array = []
    for i in range(label_array[1]):
        array.append(np.frombuffer(image[i], dtype = '>u1'))
    
    # print(array.reshape(28, 28))
    show_image(array)