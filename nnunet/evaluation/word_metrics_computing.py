# -*- coding: utf-8 -*-
# Author: Xiangde Luo (https://luoxd1996.github.io).
# Date:   16 Dec. 2021
# Implementation for computing the DSC and HD95 in the WORD dataset.
# # Reference:
#   @article{luo2021word,
#   title={{WORD}: A large scale dataset, benchmark and clinical applicable study for abdominal organ segmentation from CT image},
#   author={Xiangde Luo, Wenjun Liao, Jianghong Xiao, Jieneng Chen, Tao Song, Xiaofan Zhang, Kang Li, Dimitris N. Metaxas, Guotai Wang, Shaoting Zhang},
#   journal={arXiv preprint arXiv:2111.02403},
#   year={2021}
# }

import os
import numpy as np
import SimpleITK as sitk
from medpy import metric
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import json

def cal_metric(gt, pred, voxel_spacing):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=voxel_spacing)
        return np.array([dice, hd95])
    else:
        return np.array([0.0, 50])

def each_cases_metric(gt, pred, voxel_spacing):
    classes_num = gt.max() + 1
    class_wise_metric = np.zeros((classes_num-1, 2))
    for cls in tqdm(range(1, classes_num)):
        class_wise_metric[cls-1, ...] = cal_metric(pred==cls, gt==cls, voxel_spacing)
    print(class_wise_metric)
    return class_wise_metric


cal_filename = "predictionsTs_DSC_HD95"

class_list = ['Liver','Spleen','Kidney(L)','Kidney(R)','Stomach','Gallbladder','Esophagus','Pancreas','Duodenum','Colon','Intestine','Adrenal','Rectum','Bladder','Head of Femur(L)','Head of Femur(R)']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--GT_dir", default="/media/ps/passport2/ltc/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task100_WORD/labelsTs/",required=False)
    parser.add_argument("--Pred_dir_ori", default="/media/ps/passport2/ltc/nnUNet/nnUNet_outputs",required=False)
    parser.add_argument('-m', '--mode', help="analyze, calculate",
                        default="calculate", required=False)
    parser.add_argument('-p', '--pred_dir', help="pred_exp_name",
                        default="3Dv2", required=False)
    
    args = parser.parse_args()
    
    Pred_dir = os.path.join(args.Pred_dir_ori,args.pred_dir)
    
    assert args.mode in ["analyze", "calculate"]
    
    if args.mode == "calculate":
        all_results = np.zeros((30,16,2))
        # for ind, case in (enumerate(sorted(os.listdir(args.GT_dir)))):
        
        ind=0
        case="word_0057.nii.gz"
        
        gt_itk = sitk.ReadImage(args.GT_dir+case)
        voxel_spacing = (gt_itk.GetSpacing()[2], gt_itk.GetSpacing()[0], gt_itk.GetSpacing()[1])
        gt_array = sitk.GetArrayFromImage(gt_itk)
        pred_itk = sitk.ReadImage(os.path.join(Pred_dir,case))
        pred_array = sitk.GetArrayFromImage(pred_itk)
        print(f'{ind}/30: evaluating {case}...')
        print()
        all_results[ind, ...] = each_cases_metric(gt_array, pred_array, voxel_spacing)

        np.save(os.path.join(Pred_dir,cal_filename)+'.npy', all_results)
    else:
        all_results = np.load(os.path.join(Pred_dir,cal_filename)+'.npy')
    
    mean_results = np.mean(all_results,0)
    
    result_dict = {
        "dice": {},
        "HD95": {},
        "mean": {"dice":np.mean(mean_results[:,0]),
                 "HD95":np.mean(mean_results[:,1])}
    }
    
    for i in range(len(class_list)):
        result_dict['dice'][class_list[i]] = mean_results[i,0]
        result_dict['HD95'][class_list[i]] = mean_results[i,1]

    with open(os.path.join(Pred_dir,cal_filename)+'.json', 'w') as json_file:
        json.dump(result_dict, json_file, indent=4)

    print("done")

    '''
    plt.imsave("gt.jpg",gt_array[100,:,:])
    plt.imsave("pred.jpg",pred_array[100,:,:])
    '''
    
if __name__ == "__main__":
    main()