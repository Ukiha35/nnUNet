import numpy as np
import os,sys
import SimpleITK as sitk
import shutil
from tqdm import tqdm
import json

if __name__ == '__main__':
    root_COVID = "/media/ps/passport2/ltc/COVID-19-CT-Seg/"
    root_MosMedData = "/media/ps/passport2/ltc/MosMedData/"
    nnUNet_dest = "/media/ps/passport2/ltc/nnUNetv2/nnUNet_raw/Dataset102_COVID"
    
    COVID_data_path = os.path.join(root_COVID, "COVID-19-CT-Seg_20cases")
    COVID_label_path = os.path.join(root_COVID, "Lung_and_Infection_Mask")
    labelsTr_path = os.path.join(nnUNet_dest, "labelsTr")
    imagesTr_path = os.path.join(nnUNet_dest, "imagesTr")
    
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
        
        # read label
        itk = sitk.ReadImage(ori_file)
        data = sitk.GetArrayFromImage(itk)
        
        # preprocess label
        data[data==2] = 1
        data[data==3] = 2
        
        # copy label
        new_itk = sitk.GetImageFromArray(data)
        new_itk.CopyInformation(itk)
        sitk.WriteImage(new_itk, new_file)
   
    # dataset.json
    print("storing dataset.json...")
    dataset_settings = {
        "name": "COVID-19-CT-Seg",
        "labels": {
            "background": 0,
            "lung": [1,2],
            "infection":2
        },
        "regions_class_order": [1,2],
        "numTraining": 20,
        "channel_names": {
            "0": "CT"
        },
        "file_ending": ".nii.gz"
    }
    with open(os.path.join(nnUNet_dest, "dataset.json"), "w") as f:
        json.dump(dataset_settings, f, indent=4)
    
    print("done!")
    
    
    
    
    
    