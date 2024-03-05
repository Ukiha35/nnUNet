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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--GT_dir", default="/media/ps/passport2/ltc/nnUNetv2/nnUNet_raw/Dataset061_CREMI",required=False)
    parser.add_argument("--Pred_dir_ori", default="/media/ps/passport2/ltc/nnUNetv2/nnUNet_outputs/CREMI",required=False)
    parser.add_argument('-p', '--pred_dir', help="pred_exp_name",
                        default="Test", required=False)
    
    args = parser.parse_args()
    
    Pred_dir = os.path.join(args.Pred_dir_ori,args.pred_dir)    
    
    if not os.path.exists(join(args.GT_dir, "imagesTs", "sample_c.hdf")):
        out_a = CremiFile(join(args.GT_dir, "imagesTs", 'sample_c.hdf'), 'w')
        
        gt_image = sitk.GetArrayFromImage(sitk.ReadImage(join(args.GT_dir, "imagesTs", "sample_c_0000.nii.gz"))).astype(np.uint64)
        gt_image[gt_image == 0] = 0xffffffffffffffff
        volume = Volume(gt_image, (40., 4., 4.))
        out_a.write_raw(volume)
        
        gt_label = sitk.GetArrayFromImage(sitk.ReadImage(join(args.GT_dir, "labelsTs", "sample_c.nii.gz"))).astype(np.uint64)
        gt_label[gt_label == 0] = 0xffffffffffffffff
        clefts = Volume(gt_label, (40., 4., 4.))
        out_a.write_clefts(clefts)
        
        out_a.close()
    
    # if not os.path.exists(join(Pred_dir, "sample_c.hdf")):
    pred = sitk.GetArrayFromImage(sitk.ReadImage(join(Pred_dir, "sample_c.nii.gz"))).astype(np.uint64)
    pred[pred == 0] = 0xffffffffffffffff
    out_a = CremiFile(join(Pred_dir, 'sample_c.hdf'), 'w')
    clefts = Volume(pred, (40., 4., 4.))
    out_a.write_clefts(clefts)
    out_a.close()

    
    
    
    test = CremiFile(join(Pred_dir, 'sample_c.hdf'), 'r')
    truth = CremiFile(join(args.GT_dir, "imagesTs", 'sample_c.hdf'), 'r')

    clefts_evaluation = Clefts(test.read_clefts(), truth.read_clefts())

    # test_array = np.array(test.read_clefts().data,dtype=np.uint8)
    # truth_array = np.array(truth.read_clefts().data,dtype=np.uint8)
    # dice = metric.binary.dc(test_array, truth_array)

    false_positive_count = clefts_evaluation.count_false_positives()
    false_negative_count = clefts_evaluation.count_false_negatives()

    false_positive_stats = clefts_evaluation.acc_false_positives()
    false_negative_stats = clefts_evaluation.acc_false_negatives()

    print ("Clefts")
    print ("======")

    print ("\tfalse positives: " + str(false_positive_count))
    print ("\tfalse negatives: " + str(false_negative_count))

    print ("\tdistance to ground truth: " + str(false_positive_stats))
    print ("\tdistance to proposal    : " + str(false_negative_stats))

    # print("\tdice: " + str(dice))
    
    result_dict = {
        "name": args.pred_dir,
        "false positives": false_positive_count,
        "false negatives": false_negative_count,
        "distance to ground truth": false_positive_stats,
        "distance to proposal": false_negative_stats,
        "cremi score": (false_positive_stats['mean']+false_negative_stats['mean'])/2,
        # "dice": dice
    }
    
    with open(os.path.join(Pred_dir,"predictionsTs_CREMIScore")+'.json', 'w') as json_file:
        json.dump(result_dict, json_file, indent=4)
        
if __name__ == '__main__':
    main()