import os
import glob
import pandas as pd
import numpy as np

class DataPreprocessing():
    def __init__(self, img_dir:str, train_val_test_ratio:list):
        super().__init__()
        self.img_dir = img_dir
        if sum(train_val_test_ratio) != 1:
            raise ValueError("Elements in train val test ratio should sum to 1")
        else:
            self.train_val_test_ratio = train_val_test_ratio
    
    def train_val_test_split(self):
        all_images = glob.glob(os.path.join(
            *[self.img_dir, "*.jpg"]
        ))
        all_images = [os.path.basename(path) for path in all_images]

        train_images = []
        val_images = []
        test_images = []
        for i in range(17):
            permutated_images = np.random.permutation(all_images[i*80 : (i+1)*80])
            train_end = int(self.train_val_test_ratio[0]*len(permutated_images))
            val_end = train_end + int(self.train_val_test_ratio[1]*len(permutated_images))
            train_images.append(permutated_images[:train_end])
            val_images.append(permutated_images[train_end:val_end])
            test_images.append(permutated_images[val_end:])
        
        train_dir = os.path.join(self.img_dir, "train")
        val_dir = os.path.join(self.img_dir, "val")
        test_dir = os.path.join(self.img_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        cnt = 0
        for class_images in train_images:
            for img in class_images:                                 
                img_path = os.path.join(self.img_dir, img)
                os.system(f"copy {img_path} {train_dir}")
            cnt += 1
            
        cnt = 0
        for class_images in val_images:
            for img in class_images:                                 
                img_path = os.path.join(self.img_dir, img)
                os.system(f"copy {img_path} {val_dir}")
            cnt += 1
        cnt = 0
        for class_images in test_images:
            for img in class_images:
                img_path = os.path.join(self.img_dir, img)
                os.system(f"copy {img_path} {test_dir}")
            cnt += 1
        
    def create_annotations_file(self, subfolder:str=None):
        if subfolder is None:
            jpg_filenames_list = glob.glob(os.path.join(
                *[self.img_dir, "*.jpg"]
            ))
        else:
            jpg_filenames_list = glob.glob(os.path.join(
                *[self.img_dir, subfolder, "*jpg"]
            ))
        jpg_filenames_list = [os.path.basename(path) for path in jpg_filenames_list]
        
        # there are 1360 images totally with 17 classes, each class has 80 images
        # 1-80 belongs to class 1; 81-160 belongs to class 2...
        labels = [int(x[6:10])//80+1 if int(x[6:10])%80 !=0 else int(x[6:10])//80 
                    for x in jpg_filenames_list]
        labels = [i - 1 for i in labels]
        
        labels = pd.DataFrame({
            "file_name": jpg_filenames_list,
            "labels": labels,
        })
        if subfolder == None:
            csv_path = os.path.join(self.img_dir, "labels.csv")
        else:
            csv_path = os.path.join(*[self.img_dir, subfolder, "labels.csv"])
        labels.to_csv(
            csv_path,
            header = False,
            index = False
        )
    
    def process_data(self):
        self.train_val_test_split()
        self.create_annotations_file(subfolder="train")
        self.create_annotations_file(subfolder="val")
        self.create_annotations_file(subfolder="test")

# ==============================================================================
# Process Data
img_dir = "flowers"
train_dir = os.path.join(img_dir, "train")
test_dir = os.path.join(img_dir, "test")
val_dir = os.path.join(img_dir, "val")
train_val_test_ratio = [0.8, 0.1, 0.1]

# Data Preprocssing
data_preprocessing = DataPreprocessing(
    img_dir = img_dir,
    train_val_test_ratio = train_val_test_ratio
)
data_preprocessing.process_data()