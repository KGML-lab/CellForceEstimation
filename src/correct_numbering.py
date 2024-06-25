import os
import argparse
from natsort import natsorted

ap = argparse.ArgumentParser(description='Cross Validation')
ap.add_argument('--path', default='Fold-1', type=str, metavar='PATH')
args = ap.parse_args()


images_resized = os.path.join(args.path, "images_resized") #"./Data/Fig3C-Control/images_resized/"
images = os.path.join(args.path, "images") #"./Data/Fig3C-Control/images/"
count = 0
for img in natsorted(os.listdir(images_resized)):
    img_type = img.split(".")[1]
    #print(img)
    new_name = str(count)+"."+img_type
    #print(new_name)
    os.rename(os.path.join(images_resized, img), os.path.join(images_resized,new_name))
    count +=1
    