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
        #path_for_file = self.__getPathForFile__()
        img = cv.resize(cv.imread(self.file),(224,224),interpolation = cv.INTER_AREA)
        return img
    
    def __getGrayImage__(self):
        img = self.__getImage__()
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        return img_gray
        
    

    def __normalizationMinMax__(self,img):
        minimo = min(img.flatten())
        maximo = max(img.flatten())    
        min_max_norm = lambda px:(px-minimo)/maximo
        img = min_max_norm(img)

        return img
    
    def __cor_gamma__(self):
        image = self.__getImage__()
        gamma = 0.55
        invGamma = 1 / gamma
        table = [((i / 255) ** invGamma) * 255 for i in range(256)]
        table = np.array(table, np.uint8)
        gamma_image = cv.LUT(image, table)
        return gamma_image
    
    def __normalizationZscore__(self,img):
        media = np.mean(img.flatten())
        desvio_padrao = np.std(img.flatten())
        z_score = lambda px:(px-media)/desvio_padrao
        img = z_score(img)

        return img
    
    
    def  __removeBackground__(self):
       img = self.__getImage__()
       img_gray = self.__getGrayImage__()

       T, img_bin = cv.threshold(img_gray,0,255,cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
       kernel = np.ones((7,7),np.uint8) 
       img_bin = cv.dilate(img_bin,kernel,iterations=2)
       img_bin = cv.erode(img_bin,kernel,iterations=2)
       img_result = cv.bitwise_and(img,img,mask = img_bin)

       return img_result 

    def __del__(self):
        del(self)

class Standartization(Manipulation):
    def __init__(self, datasets):
        self.dataset = datasets

    def __getPathForImages__(self):
        diretorios = os.listdir(self.dataset)

        all_path = []
        for categoria,diretorio in enumerate(diretorios):
            path_for_diretorio = os.path.join(self.dataset,diretorio)+os.sep+"*"
            dataset_img = glob.glob(path_for_diretorio)
            
            
            for img in dataset_img:
                dataset = {}
                dataset['path'] = img
                dataset['pessoa'] = diretorio
                dataset['label'] = categoria
                all_path.append(dataset)

        return all_path
    
    def __getAllImages__(self):
        all_samples = self.__getPathForImages__()

        images = []
        labels = []
        members = []
        for sample in all_samples:
            path = sample['path']
            label = sample['label']
            member = sample['pessoa']

            super().__init__(path)
            img = super().__getImage__()
            super().__del__()

            images.append(img)
            labels.append(label)
            members.append(member)

        images = np.array(images)

        dataset  = {'images':images, 'labels':labels,'members':members}

        return  dataset

    def __getAllImagesRemovedBackground__(self):
        all_samples = self.__getPathForImages__()

        images = []
        labels = []
        members = []
        for sample in all_samples:
            path = sample['path']
            label = sample['label']
            member = sample['pessoa']

            super().__init__(path)
            img = super().__removeBackground__()
            super().__del__()

            images.append(img)
            labels.append(label)
            members.append(member)

        images = np.array(images)

        dataset  = {'images':images, 'labels':labels,'members':members}

        return  dataset


