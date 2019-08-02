import os
import gzip
import numpy as np
import time
import matplotlib 
from matplotlib.font_manager import FontProperties 

# matplotlib.use('agg')

import matplotlib.pyplot as plt
import matplotlib.rcsetup as rcsetup
print(rcsetup.all_backends)

font = FontProperties(fname = r"/mnt/c/Windows/Fonts/simsun.ttc", size = 6)


def extract_train_img_data(file):
    image_list = []
    
    with gzip.open(file) as bytestream:
        
        head_bytes = bytestream.read(16)
        head = np.frombuffer(head_bytes, dtype = '>u4')
        
        for i in range(head[1]):
            image_list.append(bytestream.read(head[2] * head[3]))
        # images = bytestream.read(28 * 28)

    return head, image_list


def extract_train_label_data(file):
    
    with gzip.open(file) as bytestream:
        
        labels = np.frombuffer(bytestream.read(), dtype = '>u1', offset = 8)
        
    return labels


def load_data(path, file):
    
    print('load fashion minst')

    if not os.path.exists(os.path.join(data_path, i)):
        print(os.path.join(data_path, i) + 'is not exists.')

    return 0


def show_image(class_names, labels, images):
    plt.figure()

    for i in range(1,33):
        plt.subplot(4,8,i)

        plt.xticks([])
        plt.yticks([])

        plt.imshow(images[i - 1].reshape(28, 28), cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i - 1]], fontproperties=font)

    plt.show()


if __name__ == "__main__":

    data_path = './data/fashion'
    class_names = ['短袖圆领T恤', '裤子', '套衫', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋','包', '短靴']

    file_list = ['t10k-images-idx3-ubyte.gz',
                't10k-labels-idx1-ubyte.gz',
                'train-images-idx3-ubyte.gz',
                'train-labels-idx1-ubyte.gz']
    
    for i in file_list:
        load_data(data_path, i)
        
    headers, image = extract_train_img_data(os.path.join(data_path, 'train-images-idx3-ubyte.gz'))
    # print(image)
    print(type(image))
    
    header_array = np.frombuffer(headers, dtype = '>u4')
    print(header_array)
    
    img_array = []
    for i in range(header_array[1]):
        img_array.append(np.frombuffer(image[i], dtype = '>u1'))
    
    label = extract_train_label_data(os.path.join(data_path, 'train-labels-idx1-ubyte.gz'))
    print(label[:100])
    # print(array.reshape(28, 28))
    show_image(class_names, label, img_array)