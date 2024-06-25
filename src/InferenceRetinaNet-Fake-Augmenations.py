# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import argparse
# import some common libraries
import numpy as np
import os, json, cv2, random
import cv2
import pandas as pd
import shutil
import numpy as np

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import json
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

from natsort import natsorted
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.config import get_cfg
import os
import imageio
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imgaug.augmenters as iaa

ap = argparse.ArgumentParser(description='Cross Validation')
ap.add_argument('--fold', default='Fold-1', type=str, metavar='PATH')
ap.add_argument('--dest_test', default='Cell-25', type=str, metavar='CELL PATH')
args = ap.parse_args()

coco_format_train = pd.DataFrame(columns=["file_name","height","width","annotations"])
coco_format_train["annotations"] = coco_format_train["annotations"].astype('object')

coco_format_test = pd.DataFrame(columns=["file_name","height","width","annotations"])
coco_format_test["annotations"] = coco_format_test["annotations"].astype('object')

fold = args.fold
dest_test = args.dest_test +"/"
#fold = "Fold-ALL"
source = f"../CrossValidation2/{fold}/"   #./data_fluoro_nodes_fake/"
dest_train = source + "train/"  #"./data_fluoro_nodes_fake/train/"


output_dir = ("./model")
from contextlib import suppress

with suppress(OSError):
    os.remove(output_dir +'/boardetect_test_coco_format.json.lock')
    os.remove(output_dir +'/boardetect_test_coco_format.json')
    os.remove(source + 'test.json')
shutil.rmtree(dest_test+"images_resized", ignore_errors=True)


annotations_source = source + "annotations_text/"
images_source = source +"images/"
annotations_train = dest_train +"annotations_text/"
annotations_test = dest_test +"annotations_text/"
images_train = dest_train +"images/"
images_resized_train = dest_train +"images_resized/"
images_augmented_train = dest_train +"images_augmented/"
images_test = dest_test +"images/"
images_resized_test = dest_test +"images_resized/"
images_augmented_test = dest_test +"images_augmented/"

if os.path.isdir(images_resized_test) == False:
        os.mkdir(images_resized_test)
        
        
factor_w = 1024/1388
factor_h = 1024/1040

# def add_image_annotation(txt_file, xy_coords, filename):
#     res=pd.DataFrame(columns=["file_name","height","width","annotations"])
#     res.at[0,"height"] = 1024
#     res.at[0,"width"] = 1024
#     res.at[0,"file_name"] = filename
#     boxes = []
#     bboxes = []
#     imguag_boxes = []
#     bbox_mode = 0
#     category_id = 0
#     #rotate the image by 90* and save 
#     for xy in xy_coords:
#         box = []
#         x = float(xy.split(" ")[0])
#         y = float(xy.split(" ")[1])
       
#         x1 = int(x*factor_w - (width // 2))
#         y1 = int(y*factor_h - (width // 2))
#         x2 = int(x*factor_w + (width // 2))
#         y2 = int(y*factor_h + (width // 2))      
#         w = h = 31
#         box = [x1, y1, x2, y2]
#         boxes.append(np.array(box))
#         imguag_boxes.append( BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2))
#         #print(np.array(box))
            
   
#     res["annotations"]=res["annotations"].astype('object')
#     annotation_df = pd.DataFrame(columns=["bbox","bbox_mode","category_id"])
#     annotation_df["bbox"] = boxes
#     annotation_df["bbox_mode"] = bbox_mode
#     annotation_df["category_id"] = category_id
#     annotations = annotation_df.T.to_dict().values()
#     l = []
#     for j in annotations:
#         l.append(j)
#     res.at[0,"annotations"] = l
#     return res, imguag_boxes,boxes
    

# def add_augmented_image_annotation( boxes, filename ):
#     res=pd.DataFrame(columns=["file_name","height","width","annotations"])
#     res.at[0,"height"] = 1024
#     res.at[0,"width"] = 1024
#     res.at[0,"file_name"] = filename
#     bbox_mode = 0
#     category_id = 0         
   
#     res["annotations"]=res["annotations"].astype('object')
#     annotation_df = pd.DataFrame(columns=["bbox","bbox_mode","category_id"])
#     annotation_df["bbox"] = boxes
#     annotation_df["bbox_mode"] = bbox_mode
#     annotation_df["category_id"] = category_id
#     annotations = annotation_df.T.to_dict().values()
#     l = []
#     for j in annotations:
#         l.append(j)
#     res.at[0,"annotations"] = l
#     return res
    
# count = 0
# for txt_file in natsorted(os.listdir(annotations_train)):
#         width = 31
#         text_file = open(annotations_train + txt_file, 'r')
#         xy_coords = text_file.readlines()
       
#         image = PIL.Image.open(images_train + txt_file[:-4] + ".jpg")
#         #print(image.size)
        
#         filename = str(count) + ".jpg"
#         filename_path = images_augmented_train 
#         image2 = image.resize((1024,1024))
#         image2.save(filename_path + filename)
#         count = count+1
#         res_image, imguag_boxes, boxes= add_image_annotation(txt_file, xy_coords, filename)
#         coco_format_train = coco_format_train.append(res_image)
                
#         # image_resized = cv2.imread(images_augmented_train + str(count) + ".jpg")
#         # image_resized  = cv2.cvtColor(image_resized , cv2.COLOR_BGR2RGB)
#         # visualize(image_resized, bboxes)
        
#         image = imageio.imread(filename_path + filename)
#         bbs = BoundingBoxesOnImage(imguag_boxes, shape=image.shape)
#         #ia.imshow(bbs.draw_on_image(image))
#        # print(boxes)
        
#         seq = iaa.Sequential([
#         iaa.Affine(rotate=90)])
#         image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
#         #ia.imshow(bbs_aug.draw_on_image(image_aug))
#         #print(bbs_aug.to_xyxy_array())
#         imageio.imwrite(images_augmented_train + str(count) + ".jpg", image_aug)
#         boxes = []
#         for box in bbs_aug.to_xyxy_array():
#             boxes.append(np.array(box))       
#         res_image = add_augmented_image_annotation(boxes, str(count) + ".jpg")
#         count = count+1
#         coco_format_train = coco_format_train.append(res_image)
        
#         seq = iaa.Sequential([
#         iaa.Affine(rotate=180)])
#         image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
#         #ia.imshow(bbs_aug.draw_on_image(image_aug))
#         #print(bbs_aug.to_xyxy_array())
#         imageio.imwrite(images_augmented_train + str(count) + ".jpg", image_aug)
#         boxes = []
#         for box in bbs_aug.to_xyxy_array():
#             boxes.append(np.array(box))       
#         res_image = add_augmented_image_annotation(boxes, str(count) + ".jpg")
#         count = count+1
#         coco_format_train = coco_format_train.append(res_image)
               
#         seq = iaa.Sequential([
#         iaa.Affine(rotate=270)])
#         image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
#        # ia.imshow(bbs_aug.draw_on_image(image_aug))
#         #print(bbs_aug.to_xyxy_array())
#         imageio.imwrite(images_augmented_train + str(count) + ".jpg", image_aug)
#         boxes = []
#         for box in bbs_aug.to_xyxy_array():
#             boxes.append(np.array(box))       
#         res_image = add_augmented_image_annotation(boxes, str(count) + ".jpg")
#         count = count+1
#         coco_format_train = coco_format_train.append(res_image)
        
#         #flip horizontally
#         seq = iaa.Sequential([
#         iaa.Fliplr(1.0)])
#         image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
#         #ia.imshow(bbs_aug.draw_on_image(image_aug))
#         #print(bbs_aug.to_xyxy_array())
#         imageio.imwrite(images_augmented_train + str(count) + ".jpg", image_aug)
#         boxes = []
#         for box in bbs_aug.to_xyxy_array():
#             boxes.append(np.array(box))       
#         res_image = add_augmented_image_annotation(boxes, str(count) + ".jpg")
#         count = count+1
#         coco_format_train = coco_format_train.append(res_image)     
        
#         #flip vertically
#         seq = iaa.Sequential([
#         iaa.Flipud(1.0)])
#         image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
#         #ia.imshow(bbs_aug.draw_on_image(image_aug))
#         #print(bbs_aug.to_xyxy_array())
#         imageio.imwrite(images_augmented_train + str(count) + ".jpg", image_aug)
#         boxes = []
#         for box in bbs_aug.to_xyxy_array():
#             boxes.append(np.array(box))       
#         res_image = add_augmented_image_annotation(boxes, str(count) + ".jpg")
#         count = count+1
#         coco_format_train = coco_format_train.append(res_image)      
        
# #         category_ids = [0]*len(bboxes)
# #         transformed = transform_h_flip(image=image_resized , bboxes=bboxes,category_ids = category_ids )
# #         visualize( transformed['image'], transformed['bboxes'])
# #         transformed = transform_v_flip(image=image_resized , bboxes=bboxes,category_ids = category_ids )
# #         visualize( transformed['image'], transformed['bboxes'])
# #         transformed = transform_rotate_90(image=image_resized , bboxes=bboxes,category_ids = category_ids )
# #         visualize( transformed['image'], transformed['bboxes'])
                 
#         coco_format_train.reset_index(drop=True,inplace=True)
        
        
# coco_format_train.reset_index(inplace=True)
# coco_format_train.rename(columns={"index":"image_id"},inplace=True)
# coco_format_train.to_json(source + "train.json",orient="records")


for txt_file in natsorted(os.listdir(images_test)):
        width = 31
        # text_file = open(annotations_test + txt_file, 'r')
        # xy_coords = text_file.readlines()
        # xy_coords = np.loadtxt(text_file)
        boxes = []
        res=pd.DataFrame(columns=["file_name","height","width","annotations"])
        image = PIL.Image.open(images_test + txt_file[:-4] + ".jpg")
        #print(image.size)
        res.at[0,"height"] = 1024
        res.at[0,"width"] = 1024
        res.at[0,"file_name"] = txt_file[:-4]+".jpg"
        bbox_mode = 0
        category_id = 0
        image2 = image.resize((1024,1024))
        image2.save(images_resized_test + txt_file[:-4] + ".jpg")
        # for xy in xy_coords:
        #     box = []
        #     # x = float(xy.split(" ")[0])
        #     # y = float(xy.split(" ")[1])
        #     x = float(xy[0])
        #     y = float(xy[1])
        #     x1 = int(x*factor_w - (width // 2))
        #     y1 = int(y*factor_h - (width // 2))
        #     x2 = int(x*factor_w + (width // 2))
        #     y2 = int(y*factor_h + (width // 2))
        #     w = h = 31
        #     box = [x1, y1, x2, y2]
        box = [0,0,0,0]
        boxes.append(np.array(box))
            #print(np.array(box))

        res["annotations"]=res["annotations"].astype('object')
        annotation_df = pd.DataFrame(columns=["bbox","bbox_mode","category_id"])
        annotation_df["bbox"] = boxes
        annotation_df["bbox_mode"] = bbox_mode
        annotation_df["category_id"] = category_id
        annotations = annotation_df.T.to_dict().values()
        l = []
        for j in annotations:
            l.append(j)
        res.at[0,"annotations"] = l
        coco_format_test = coco_format_test.append(res)
        coco_format_test.reset_index(drop=True,inplace=True)
        
coco_format_test.reset_index(inplace=True)
coco_format_test.rename(columns={"index":"image_id"},inplace=True)
coco_format_test.to_json(dest_test + "test.json",orient="records")

def get_board_dicts(imgdir, mode):
    if mode is 'train':
        #json_file = "../data_fluoro_nodes_fake"+"/train.json" #Fetch the json file
        json_file = imgdir+"/test.json" #Fetch the json file
    if mode is 'test':
         json_file = dest_test+"/test.json" #Fetch the json file
    with open(json_file) as f:
        dataset_dicts = json.load(f)
    for i in dataset_dicts:
        filename = i["file_name"] 
        if mode is 'train':
            i["file_name"] = imgdir+ "/" + mode + "/images_augmented/"+filename 
            print(i["file_name"] )
        if mode is 'test':
            i["file_name"] = dest_test+ "/images_resized/"+filename 
        for j in i["annotations"]:
            j["bbox_mode"] = BoxMode.XYXY_ABS #Setting the required Box Mode
            j["category_id"] = int(j["category_id"])
    return dataset_dicts

#Registering the Dataset
for d in ["test"]:
    DatasetCatalog.register("boardetect_" + d, lambda d=d: get_board_dicts(dest_test, d))
    MetadataCatalog.get("boardetect_" + d).set(thing_classes=["node"])
#board_metadata = MetadataCatalog.get("boardetect_train")
val_metadata = MetadataCatalog.get("boardetect_test")

class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        output_folder = cfg.OUTPUT_DIR
        

    return COCOEvaluator(dataset_name, cfg, False, output_folder)
cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu' 
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml")) #Get the basic model configuration from the model zoo 
#Passing the Train and Validation sets
cfg.DATASETS.TRAIN = ("boardetect_test",)
cfg.DATASETS.TEST = ("boardetect_test",)
cfg.OUTPUT_DIR = output_dir
if os.path.exists(cfg.OUTPUT_DIR + "/boardetect_test_coco_format.json") == True:
    os.remove(cfg.OUTPUT_DIR + "/boardetect_test_coco_format.json")
# Number of data loading threads
cfg.DATALOADER.NUM_WORKERS = 4
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0039999.pth")
# Number of images per batch across all machines.
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.0125  # pick a good LearningRate
cfg.SOLVER.MAX_ITER = 50000 #50000  #No. of iterations
# cfg.SOLVER.STEPS = (300,600)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024 * 20
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # No. of classes = [HINDI, ENGLISH, OTHER]
cfg.TEST.EVAL_PERIOD = 500 # No. of iterations after which the Validation Set is evaluated. 
cfg.TEST.DETECTIONS_PER_IMAGE = 600
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
cfg.VIS_PERIOD = 500
cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN = 5000
cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST = 5000
cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 5000
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 5000
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg) 
trainer.resume_or_load(resume=True)
trainer.train()
os.system(f'cp {output_dir}/coco_instances_results.json {dest_test}/coco_instances_results.json')  
os.system(f'cp {output_dir}/boardetect_test_coco_format.json {dest_test}/boardetect_test_coco_format.json')
