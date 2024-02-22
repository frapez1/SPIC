# This is the code to evaluate the performances, 
# in terms of BPP, mIoU and FID of the proposed SPIC framework

import os
import subprocess
import metrics_functions as mfunc
import numpy as np

##################
#### Parameters
##################
relust_path = "path/to/Result"
internimage_path = "path/to/InternImage"


##################
#### Main
##################
if __name__ == '__main__':
    ##### evaluate the BPP for the coarse
    BPP_coarse = mfunc.calculate_bpp(os.path.join(relust_path, "compressed"), ext=".bpg")
    
    ##### evaluate the BPP for the input SSM
    mfunc.flif_compression(f"{relust_path}/labels", f"{relust_path}/labels")
    BPP_SSM = mfunc.calculate_bpp(os.path.join(relust_path, "labels"), ext=".flif")
    
    ##### generate the SSM with internimage
    command = f"cd {internimage_path} && python3 segmentation/image_demo.py {relust_path}/samples segmentation/configs/cityscapes/upernet_internimage_xl_512x1024_160k_mapillary2cityscapes.py segmentation/checkpoint_dir/seg/upernet_internimage_xl_512x1024_160k_mapillary2cityscapes.pth --out {relust_path}/sem_generated"
    mfunc.run_command_with_conda_env("internimage", command)
    
    ##### evaluate the mIoU 
    true_SSM = mfunc.get_semantic_maps(os.path.join(relust_path, "labels/"), interimage=False)
    generated_SSM = mfunc.get_semantic_maps(os.path.join(relust_path, "sem_generated/"), interimage=True)
    mIoU = mfunc.calculate_mIoU(true_SSM, generated_SSM)
    
    ##### evaluate the FID
    true_img_path = os.path.join(relust_path, "images/")
    sampled_img_path = os.path.join(relust_path, "samples/")
    fid = mfunc.calculate_FID(true_img_path, sampled_img_path)
    
    ##### print the results
    print(f"BPP: {round(BPP_coarse+BPP_SSM,4)}")
    print(f"mIoU: {round(mIoU,4)}")
    print(f"FID: {round(fid,4)}")
    
