# this is the file that contains the metric functions for evaluating the model


import os
import glob
import subprocess
import blobfile as bf
import numpy as np
from PIL import Image
import json
import torch
import threading

##############
#### BPP
##############
# calculate the BPP
def calculate_bpp(coarse_path, ext=".bpg", size=512*256):
    bpp = 0
    count = 0
    for root, dirs, files in os.walk(coarse_path):
        for file in files:
            if file.endswith(ext):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                bpp += file_size*8/(size)
                count += 1
    return bpp/count

def flif_compression(input_path, output_path):
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".png"):
                file_path = os.path.join(root, file)
                # join output_path with the file name replacinv .png with .flif
                output_flif = os.path.join(output_path, file.replace(".png", ".flif"))
                command = f"flif -e {file_path} {output_flif} --overwrite"
                os.system(command)

##############
#### SSM
##############
# get the name of the conda environment
def find_conda_python(env_name):
    """Find the Python executable in a specified conda environment."""
    try:
        envs = subprocess.check_output(['conda', 'env', 'list', '--json'], text=True)
        envs_json = json.loads(envs)
        for env in envs_json['envs']:
            if env_name in env:
                # Assuming the standard location of the Python executable within the environment
                python_path = f"{env}/bin/python"
                return python_path
    except subprocess.CalledProcessError as e:
        print(f"Error querying conda environments: {e}")
    return None

# run a command in a conda environment
def run_command_with_conda_env(env_name, command):
    """Run a command using the Python executable of the specified conda environment."""
    python_executable = find_conda_python(env_name)
    if not python_executable:
        print(f"Python executable for environment '{env_name}' not found.")
        return

    # Modify your command to use the found Python executable
    modified_command = command.replace("python3", python_executable, 1)

    try:
        # Execute the modified command
        process = subprocess.Popen(modified_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        
        if process.returncode == 0:
            print("Command executed successfully.")
        else:
            print("Command execution failed.")
    except Exception as e:
        print(f"Failed to run command: {e}")



##############
#### mIoU
##############
def get_pixacc_miou(total_correct, total_label, total_inter, total_union):
    pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
    IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
    mIoU = IoU.mean()
    return pixAcc, mIoU

# class to compute the mIoU
class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scroes
    """
    def __init__(self, nclass):
        self.nclass = nclass
        self.lock = threading.Lock()
        self.reset()

    def update(self, labels, preds):
        def evaluate_worker(self, label, pred):
            correct, labeled = batch_pix_accuracy(
                pred, label)
            inter, union = batch_intersection_union(
                pred, label, self.nclass)
            with self.lock:
                self.total_correct += correct
                self.total_label += labeled
                self.total_inter += inter
                self.total_union += union
            return inter, union

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, labels, preds)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker,
                                        args=(self, label, pred),
                                       )
                       for (label, pred) in zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            raise NotImplemented

    def get_all(self):
        return self.total_correct, self.total_label, self.total_inter, self.total_union

    def get(self):
        return get_pixacc_miou(self.total_correct, self.total_label, self.total_inter, self.total_union)
 
    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0
        return

def batch_pix_accuracy(output, target):
    """Batch Pixel Accuracy
    Args:
        predict: input 4D tensor
        target: label 3D tensor
    """
    predict = output

    predict = predict.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1

    pixel_labeled = np.sum(target !=20)
    pixel_correct = np.sum((predict == target)*(target > 0))
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """
    predict = output
    mini = 1
    maxi = nclass+1
    nbins = nclass
    predict = predict.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1

    predict = predict * (target != 20).astype(predict.dtype)
    intersection = predict * (predict == target)
    # areas of intersection and union
    area_inter, a = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_inter = area_inter
    area_pred, b = np.histogram(predict, bins=nbins+1, range=(mini, maxi+1))
    area_pred = area_pred[:-1]
    area_lab, c = np.histogram(target, bins=nbins+1, range=(mini, maxi+1))
    area_lab = area_lab[:-1]
    
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union


# read the SSM images
def get_semantic_maps(path, interimage=False):
    # get the semantic maps paths
    semantics_files = sorted(glob.glob(path + "*.png"))
    
    # colour dictionary in case of colour images
    train_dict_colors = {(128, 64, 128): 0, (244, 35, 232): 1, (70, 70, 70): 2, (102, 102, 156): 3, (190, 153, 153): 4, (153, 153, 153): 5, (250, 170, 30): 6, (220, 220, 0): 7, (107, 142, 35): 8, (152, 251, 152): 9, (70, 130, 180): 10, (220, 20, 60): 11, (255, 0, 0): 12, (0, 0, 142): 13, (0, 0, 70): 14, (0, 60, 100): 15, (0, 80, 100): 16, (0, 0, 230): 17, (119, 11, 32): 18}
    
    # read the semantic maps and save them in a list
    semantics = []
    for semantic in semantics_files: ## TODO: remove the [:32] to the right number
        with bf.BlobFile(semantic, "rb") as f:
            pil_class = Image.open(f)
            pil_class.load()
        # if the SSM is in grayscale no problem
        if not interimage:
            pil_class = pil_class.convert("L")
            pil_class = pil_class.resize((512,256), resample=Image.NEAREST)
            # rescale from 0-255 to 0-19
            semantic_img = np.array(pil_class)//7
            semantic_img[semantic_img == 36] = 19
        # otherwise if it is generated with interimage, I have to adjust the colors
        else:
            pil_class = pil_class.convert("RGB")
            pil_class = pil_class.resize((512,256), resample=Image.NEAREST)
            # Convert the image to a list of pixel values
            pixels = list(pil_class.getdata())

            # Replace RGB values with corresponding integers from the dictionary
            converted_pixels = [train_dict_colors[pixel] if pixel in train_dict_colors else 19 for pixel in pixels]  # Replace with 0 if not found

            # Create a new image with the same size as the original one
            new_image = Image.new('L', pil_class.size)  # 'L' mode for (8-bit pixels, black and white)

            # Put the updated data back into the image
            new_image.putdata(converted_pixels)

            semantic_img = np.array(new_image)
        semantics.append(semantic_img)
    return {'name': semantics_files, 'semantics' : semantics}


def calculate_mIoU(true_SSM, generated_SSM):
    # get the semantic maps
    true_semantics = true_SSM['semantics']
    generated_semantics = generated_SSM['semantics']
    
    metric = SegmentationMetric(19)
    metric.update( torch.tensor(true_semantics),torch.tensor(generated_semantics))
    mIoU = 1.0 * metric.total_inter / (np.spacing(1) + metric.total_union)
    
    return np.mean(mIoU)


##############
#### FID
##############
# calculate the FID
def calculate_FID(true_img_path, sampled_img_path):
    # command to calculate the FID
    command = f"python -m pytorch_fid {true_img_path} {sampled_img_path}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print("FID evaluation failed.")
    # extract the FID from the output "FID: value"
    stdout = stdout.split("\n")[-2].split(" ")[-1]
    return float(stdout)
