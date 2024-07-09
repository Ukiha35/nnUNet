import numpy as np
import os,sys
import SimpleITK as sitk
import shutil
from tqdm import tqdm
import json

if __name__ == '__main__':
    root_COVID = "/media/ps/passport2/ltc/COVID-19-CT-Seg/"
    root_MosMedData = "/media/ps/passport2/ltc/MosMedData/"
    nnUNet_dest = "/media/ps/passport2/ltc/nnUNetv2/nnUNet_raw/Dataset101_COVID"
    
    COVID_data_path = os.path.join(root_COVID, "COVID-19-CT-Seg_20cases")
    COVID_label_path = os.path.join(root_COVID, "Infection_Mask")
    MosMedData_data_path = os.path.join(root_MosMedData, "studies", "CT-1")
    MosMedData_label_path = os.path.join(root_MosMedData, "masks")
    labelsTr_path = os.path.join(nnUNet_dest, "labelsTr")
    labelsTs_path = os.path.join(nnUNet_dest, "labelsTs")
    imagesTr_path = os.path.join(nnUNet_dest, "imagesTr")
    imagesTs_path = os.path.join(nnUNet_dest, "imagesTs")
    
    # remove original files
    if os.path.exists(nnUNet_dest):
        shutil.rmtree(nnUNet_dest)
    os.makedirs(nnUNet_dest)
    
    # imagesTr
    os.makedirs(imagesTr_path)
    print("processing imagesTr...")
    ls = [file for file in os.listdir(COVID_data_path) if file.endswith(".nii.gz")]
    for file in tqdm(ls):
        ori_file = os.path.join(COVID_data_path,file)
        new_file = os.path.join(imagesTr_path,file.split('.nii.gz')[0]+'_0000.nii.gz')
        
        # read image
        itk = sitk.ReadImage(ori_file)
        data = sitk.GetArrayFromImage(itk)
        
        # preprocess image
        if 'coronacases' in file:
            data = np.clip(data, -1250, 250)
            data = np.array((data + 1250) / 1500 * 255, dtype=np.int16) # from -1250~250 to 0~255
        
        # copy image
        new_itk = sitk.GetImageFromArray(data)
        new_itk.CopyInformation(itk)
        sitk.WriteImage(new_itk, new_file)
        
    # labelsTr
    os.makedirs(labelsTr_path)
    ls = [file for file in os.listdir(COVID_label_path) if file.endswith(".nii.gz")]
    print("processing labelsTr...")
    for file in tqdm(ls):
        ori_file = os.path.join(COVID_label_path,file)
        new_file = os.path.join(labelsTr_path,file)
        
        # copy label
        shutil.copyfile(ori_file, new_file)
        
    # imagesTs
    os.makedirs(imagesTs_path)
    print("processing imagesTs...")
    ls = [file for file in os.listdir(MosMedData_data_path) if file.endswith(".nii.gz") and int(file[6:10])>=255 and int(file[6:10])<=304]
    for file in tqdm(ls):
        ori_file = os.path.join(MosMedData_data_path,file)
        new_file = os.path.join(imagesTs_path,file.split('.nii.gz')[0]+'_0000.nii.gz')
        
        # read image
        itk = sitk.ReadImage(ori_file)
        data = sitk.GetArrayFromImage(itk)
        
        # preprocess image
        data = np.clip(data, -1250, 250)
        data = np.array((data + 1250) / 1500 * 255, dtype=np.int16) # from -1250~250 to 0~255
        
        # copy image
        new_itk = sitk.GetImageFromArray(data)
        new_itk.CopyInformation(itk)
        sitk.WriteImage(new_itk, new_file)
        
    # labelsTs
    os.makedirs(labelsTs_path)
    print("processing labelsTs...")
    ls = [file for file in os.listdir(MosMedData_label_path) if file.endswith(".nii.gz")]
    for file in tqdm(ls):
        ori_file = os.path.join(MosMedData_label_path,file)
        new_file = os.path.join(labelsTs_path,file.split('_mask')[0]+'.nii.gz')
        
        # copy label
        shutil.copyfile(ori_file, new_file)
    
    # dataset.json
    print("storing dataset.json...")
    dataset_settings = {
        "name": "COVID-19-CT-Seg & MosMedData",
        "description": "MosMedData dataset contains anonymised human lung computed tomography (CT) scans with COVID-19 related findings, as well as without such findings. COVID-19-CT-Seg CT scans are from the Coronacases Initiative and Radiopaedia that can be freely downloaded from https://academictorrents.com/details/136ffddd0959108becb2b3a86630bec049fcb0ff.",
        "licence": "CC BY-NC-SA & CC BY-NC-ND",
        "labels": {
            "background": 0,
            "infection": 1
        },
        "numTraining": 20,
        "channel_names": {
            "0": "CT"
        },
        "file_ending": ".nii.gz"
    }
    with open(os.path.join(nnUNet_dest, "dataset.json"), "w") as f:
        json.dump(dataset_settings, f, indent=4)
    
    print("done!")
    
    
    
    
    
    