from jittor.dataset.dataset import Dataset
import matplotlib.pyplot as plt
import os

class SymbolDataset(Dataset):
    def __init__(self, root_path, transform, max_size=-1):
        super().__init__()
        
        resolution_path = os.path.join(root_path)
        train_image = []
        size = 0
        for image_file in os.listdir(resolution_path):
            image_path = os.path.join(resolution_path, image_file)
            ext = os.path.splitext(image_path)[-1]
            
            if ext not in ['.png', '.jpg']:
                continue
                
            image = plt.imread(image_path)
            
            if ext == '.png':
                image = image * 255
                
            image = image.astype('uint8')
            train_image.append(image)
            size += 1
            if size % 1000 == 0:
                print("loaded " + str(size) + " images")
            if size >= max_size and max_size > 0:
                break
        self.train_image = train_image
        self.transform  = transform
        
    def __len__(self):
        return len(self.train_image)
    
    def __getitem__(self, index):
        X = self.train_image[index]
        return self.transform(X)