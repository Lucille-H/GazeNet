import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import categorical_accuracy

import gzip
import pandas as pd
from PIL import Image
import re
import os
import numpy as np
import pickle

import cv2
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import gzip
import os

IMG_DIR  = "Demo_img"
OUT_DIR = "Demo_res"


from Face_Detection import image, load_test

def load_model():
    model = keras.models.load_model("Model/1", custom_objects={'euclideanLoss': euclideanLoss,
                                                               'categorical_accuracy': categorical_accuracy})
    return model

def euclideanLoss(y_true, y_pred):
    return K.mean(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)))


def predict_gaze(model, images, faces, heads):
    preds = model.predict([np.array(images),np.array(faces),np.array(heads)])
    return preds

def visualize(filenames,heads, preds):
    df_valfnames = pd.DataFrame(zip(filenames, range(len(filenames))), columns=['filenames', 'index'])
    grouped_df = df_valfnames.groupby(['filenames'], as_index=False).groups
    for i, k in enumerate(grouped_df.keys()):
        ima = Image.open(IMG_DIR + k)
        w = np.size(ima)[0]
        h = np.size(ima)[1]

        fa_in = grouped_df[k]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for res in fa_in:
            plt.arrow(heads[res][0] * w, heads[res][1] * h, preds[0][res][0] * w - heads[res][0] * w,
                      preds[0][res][1] * h - heads[res][1] * h, color="red", width=1, head_width=20)
        plt.imshow(ima.convert('RGB'))
        # plt.show()
        plt.save(OUT_DIR+k)

if __name__=="__main__":
    images_dic = image(IMG_DIR)
    filenames,images,faces,heads=load_test(IMG_DIR,images_dic)
    model = load_model()
    preds = predict_gaze(model, images, faces, heads)
    visualize(filenames, heads, preds)