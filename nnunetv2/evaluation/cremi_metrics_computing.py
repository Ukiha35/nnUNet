#!/usr/bin/python

from cremi.io import CremiFile
from cremi.evaluation import NeuronIds, Clefts, SynapticPartners
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import SimpleITK as sitk
import shutil
import os
import argparse
import pickle
import json
from cremi.io import CremiFile
from cremi.Volume import Volume
from medpy import metric
try:
    import h5py
except ImportError:
    h5py = None

def evaluate_one(Pred_dir,GT_dir,file_name):
    if not os.path.exists(join(GT_dir, "imagesTs", f"{file_name}.hdf")):
        out_a = CremiFile(join(GT_dir, "imagesTs", f"{file_name}.hdf"), 'w')
        
        gt_image = sitk.GetArrayFromImage(sitk.ReadImage(join(GT_dir, "imagesTs", f"{file_name}_0000.nii.gz"))).astype(np.uint64)
        gt_image[gt_image == 0] = 0xffffffffffffffff
        volume = Volume(gt_image, (40., 4., 4.))
        out_a.write_raw(volume)
        
        gt_label = sitk.GetArrayFromImage(sitk.ReadImage(join(GT_dir, "labelsTs", f"{file_name}.nii.gz"))).astype(np.uint64)
        gt_label[gt_label == 0] = 0xffffffffffffffff
        clefts = Volume(gt_label, (40., 4., 4.))
        out_a.write_clefts(clefts)
        
        out_a.close()
    
    # if not os.path.exists(join(Pred_dir, f"{file_name}.hdf")):
    pred = sitk.GetArrayFromImage(sitk.ReadImage(join(Pred_dir, f"{file_name}.nii.gz"))).astype(np.uint64)
    pred[pred == 0] = 0xffffffffffffffff
    out_a = CremiFile(join(Pred_dir, '{file_name}.hdf'), 'w')
    clefts = Volume(pred, (40., 4., 4.))
    out_a.write_clefts(clefts)
    out_a.close()
    
    test = CremiFile(join(Pred_dir, f'{file_name}.hdf'), 'r')
    truth = CremiFile(join(GT_dir, "imagesTs", f'{file_name}.hdf'), 'r')

    clefts_evaluation = Clefts(test.read_clefts(), truth.read_clefts())

    # test_array = np.array(test.read_clefts().data,dtype=np.uint8)
    # truth_array = np.array(truth.read_clefts().data,dtype=np.uint8)
    # dice = metric.binary.dc(test_array, truth_array)

    false_positive_count = clefts_evaluation.count_false_positives()
    false_negative_count = clefts_evaluation.count_false_negatives()

    false_positive_stats = clefts_evaluation.acc_false_positives()
    false_negative_stats = clefts_evaluation.acc_false_negatives()

    print (f"{file_name}:")
    print ("======")

    print ("\tfalse positives: " + str(false_positive_count))
    print ("\tfalse negatives: " + str(false_negative_count))

    print ("\tdistance to ground truth: " + str(false_positive_stats))
    print ("\tdistance to proposal    : " + str(false_negative_stats))
    return false_positive_count,false_negative_count,false_positive_stats,false_negative_stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--GT_dir", default="/media/ps/passport2/ltc/nnUNetv2/nnUNet_raw/Dataset061_CREMI",required=False)
    parser.add_argument("--Pred_dir_ori", default="/media/ps/passport2/ltc/nnUNetv2/nnUNet_outputs/CREMI",required=False)
    parser.add_argument('-p', '--pred_dir', help="pred_exp_name",default="Test", required=False)
    parser.add_argument('--mode',type=str,default='abc')
    
    args = parser.parse_args()
    assert args.mode in ['abc', 'ab']
    Pred_dir = os.path.join(args.Pred_dir_ori,args.pred_dir)    
    
    result_dict = {"name": args.pred_dir}
    
    if args.mode == 'ab':
        names = ['sample_c']
    else:
        names = ['sample_a_test','sample_b_test','sample_c_test']
    
    for name in names:
        false_positive_count, false_negative_count, false_positive_stats, false_negative_stats = evaluate_one(Pred_dir,args.GT_dir,name)
        result_dict['name'] = {
            "false positives": false_positive_count,
            "false negatives": false_negative_count,
            "distance to ground truth": false_positive_stats,
            "distance to proposal": false_negative_stats,
            "cremi score": (false_positive_stats['mean']+false_negative_stats['mean'])/2,
        }
        
        
    # print("\tdice: " + str(dice))
    
    result_dict['average cremi score'] = np.array([result_dict[name]['cremi score'] for name in names]).mean()
    
    with open(os.path.join(Pred_dir,"predictionsTs_CREMIScore")+'.json', 'w') as json_file:
        json.dump(result_dict, json_file, indent=4)
        
if __name__ == '__main__':
    main()