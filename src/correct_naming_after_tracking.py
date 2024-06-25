import os
import cv2
import shutil
import argparse
from natsort import natsorted

ap = argparse.ArgumentParser(description='Cross Validation')
ap.add_argument('--path', default='Fold-1', type=str, metavar='PATH')
args = ap.parse_args()

path = args.path + "/"
print("path",path)
#put original phase back in the frame folder and rename folder acc to original naming
org_phase_path = os.path.join(path, "phase-images")
if not os.path.exists(org_phase_path):
    org_phase_path = os.path.join(path, "images")
output_path2 = os.path.join(path, "Correction Files/ObjectDetection+CrossCorr-Forward+BackwardPass")
dest_path = os.path.join(path, "Correction Files/Forward_Backward_Tracking_Results")
print(dest_path)
shutil.rmtree(dest_path, ignore_errors=True)

frame_count = 0
frame_dir_list = natsorted(os.listdir(output_path2))

for image in natsorted(os.listdir(org_phase_path)):
    if "checkpoint" in image:
        continue
    #if image.endswith("real_A.png"):
    #image_id = image.split("frame")[1].split("_")[0]
    image_id = image.split(".")[0]
    image_type = image.split(".")[1]
    print("image_id: ", image_id)
    frame = frame_dir_list[frame_count]
    print("frame: ", frame)
    frame_id = frame.split("frame")[1]
    new_frame = "frame" + str(image_id)
    print("new frame: ", new_frame)
    #rename frame name
    shutil.copytree(os.path.join(output_path2,frame), os.path.join(dest_path, new_frame))
    #os.rename(output_path2 + frame, output_path2 + new_frame)
    frame_count = frame_count + 1
    #rename other files
    files = os.listdir(os.path.join(dest_path,new_frame))
    for file in files:
        new_file = str(file.split(f"{int(frame_id):02}")[0]) + str(f"{int(image_id):02}") + str(file.split(f"{int(frame_id):02}")[1])
        os.rename(os.path.join(dest_path, new_frame, file), os.path.join(dest_path, new_frame, new_file))
    #include image
    img = cv2.imread(os.path.join(org_phase_path, image))
    dim = img.shape[1]
    if dim == 2048:
        img = img[:, 0:1024]
    img = cv2.resize(img, (1388,1040))
    image_name_new = "frame" + str(image_id) + "."+image_type
    cv2.imwrite(os.path.join(dest_path, new_frame, image_name_new),img)
    #shutil.copy(org_phase_path + image, os.path.join(output_path2, new_frame, image) )
    # else:
    #     continue
    
    