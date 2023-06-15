import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

from tensorflow import keras
from segmentation_models.losses import jaccard_loss,dice_loss
from segmentation_models.metrics import  iou_score, f1_score, f2_score, precision, recall
import cv2
import numpy as np
from matplotlib import colors


# Chargement du modèle
loaded_model_from_h5 = keras.models.load_model('best_model/mymodel.h5',custom_objects={'dice_loss': dice_loss, 'jaccard_loss': jaccard_loss, 'precision': precision, 'recall': recall,'f1-score': f1_score})


id2category={0: 'void', 1: 'flat', 2: 'construction', 3: 'object', 4: 'nature', 5: 'sky', 6: 'human', 7: 'vehicle'}



'''Fonction qui retourne l'image des segments identifiés par le modèle'''
def generate_img_from_mask(mask,colors_palette=['b','g','r','c','m','y','k','w']):
    
    img_seg = np.zeros((mask.shape[0],mask.shape[1],3),dtype='float')
    
    for cat in id2category.keys():
        img_seg[:,:,0] += mask[:,:,cat]*colors.to_rgb(colors_palette[cat])[0]
        img_seg[:,:,1] += mask[:,:,cat]*colors.to_rgb(colors_palette[cat])[1]
        img_seg[:,:,2] += mask[:,:,cat]*colors.to_rgb(colors_palette[cat])[2]
    
        
    return img_seg


'''Prédiction du mask prédit par le modèle'''
def affichage_model_result(img_test):
    
    img_test_array  = cv2.imread(img_test)
    
    resized = cv2.resize(img_test_array,(288,144))
    
    resized = np.expand_dims(resized, axis = 0)
    
    mask_u = loaded_model_from_h5.predict(resized)
    
    
    
    '''for i in range(8):
        plt.title("Mask : {}".format(id2category[i]))
        plt.imshow(mask_u[0,:,:,i])
        plt.show()'''
        
    a_s = np.squeeze(mask_u)
    
    mask_t=generate_img_from_mask(a_s)
    
    return mask_t