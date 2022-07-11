import cv2 as cv
import os
import glob
import numpy as np

class Manipulation():
    def __init__(self, file):
        self.file = file

    def __getPathForFile__(self):
        diretorio = os.getcwd()
        path_for_file = os.path.join(diretorio,self.file)

        return path_for_file
    
    def __getImage__(self):
        path_for_file = self.__getPathForFile__()
        img = cv.imread(path_for_file)
        

    def __normalizationMinMax__(self,img):
        minimo = min(img.flatten())
        maximo = max(img.flatten())    
        min_max_norm = lambda px:(px-minimo)/maximo
        img = min_max_norm(img)

        return img
    
    def __normalizationZscore__(self,img):
        media = np.mean(img.flatten())
        desvio_padrao = np.std(img.flatten())
        z_score = lambda px:(px-media)/desvio_padrao
        img = z_score(img)

        return img
    
    
    def __preProcessMinMax__(self):
        img = self.__getImage__()
        b,g,r = cv.split(img)

        new_channels = []
        for channel in [b,g,r]:
            norm_channel = self.__normalizationMinMax__(channel)
            new_channels.append(norm_channel)
        
        norm_img = cv.merge(new_channels)

        return norm_img
    
    def __preProcessZscore__(self):
        img = self.__getImage__()
        b,g,r = cv.split(img)

        new_channels = []
        for channel in [b,g,r]:
            norm_channel = self.__normalizationZscore__(channel)
            new_channels.append(norm_channel)
        
        norm_img = cv.merge(new_channels)

        return norm_img
    

class Standartization(Manipulation):
    def __init__(self, datasets):
        self.dataset = datasets

    def __getPathForImages__(self):
        diretorios = os.listdir(self.dataset)

        all_path = []
        for diretorio in diretorios:
            dataset_img = glob.glob(os.path.join(self.dataset,diretorio),"*")
            
            
            for img in dataset_img:
                dataset = {}
                dataset['path'] = img
                dataset['label'] = diretorio
                all_path.append(dataset)

        return all_path
    
    