import os
import shutil
import glob
#update pix2pix path
pix2pix_path = '../Pix2Pix/pytorch-CycleGAN-and-pix2pix/'
video_dir_name = "Fig1-CNFMSchematic-Series"
#update phase images path, num_test
fake_fluro_path = "FluroData/" + video_dir_name + "/FakeFluro/"
phase_path = "FluroData/" + video_dir_name + "/" 
num_images = 24
## no update needed
src_folder = "./results/Unet_pristine/test_latest/images" 
dst_folder = fake_fluro_path

#change dir to pix2pix
os.chdir(pix2pix_path)
os.system('pwd')
shutil.rmtree(src_folder, ignore_errors=True)

#running pix2pix to test
os.system(f'python test.py --dataroot {phase_path} --model pix2pix --name Unet_pristine --direction BtoA --preprocess none --load_size 1024 --num_test {num_images}')

os.makedirs(fake_fluro_path, exist_ok=True)
#update final fluro location



pattern = "/*fake_B.png"
files = glob.glob(src_folder + pattern)
print(src_folder+pattern)
os.system('pwd')
print(files)

# move the files with txt extension
for file in files:
    if "checkpoint" not in file:
        # extract file name form file path
        file_name = os.path.basename(file).split("_fake_B")[0] + os.path.basename(file).split("_fake_B")[1]
        shutil.move(file, dst_folder + file_name)
        print('Moved:', file)