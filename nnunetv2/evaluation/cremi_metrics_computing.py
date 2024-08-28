#!/usr/bin/python

from cremi.io import CremiFile
# from cremi.evaluation import NeuronIds, Clefts, SynapticPartners
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
from scipy import ndimage
import SimpleITK as sitk
import shutil
import os
import argparse
import pickle
import json
from cremi.io import CremiFile
from cremi.Volume import Volume
from medpy import metric
from sklearn import metrics
try:
    import h5py
except ImportError:
    h5py = None

class Clefts:

    def __init__(self, test, truth):

        # background is True, foreground is False
        self.test_clefts_mask = ~(sitk.GetArrayFromImage(test).astype(bool))
        self.truth_clefts_mask = ~(sitk.GetArrayFromImage(truth).astype(bool))
	
        # distance to foreground
        self.test_clefts_edt = ndimage.distance_transform_edt(self.test_clefts_mask, sampling=np.flip(test.GetSpacing()))
        self.truth_clefts_edt = ndimage.distance_transform_edt(self.truth_clefts_mask, sampling=np.flip(test.GetSpacing()))

        self.all_zero = False
        
    def count_false_positives(self, threshold = 200):
        # distance to gt_foreground > 200 is gt_negative
        mask1 = np.invert(self.test_clefts_mask)
        mask2 = self.truth_clefts_edt > threshold
        false_positives = self.truth_clefts_edt[np.logical_and(mask1, mask2)]
        return false_positives.size

    def count_false_negatives(self, threshold = 200):

        mask1 = np.invert(self.truth_clefts_mask)
        mask2 = self.test_clefts_edt > threshold
        false_negatives = self.test_clefts_edt[np.logical_and(mask1, mask2)]
        return false_negatives.size

    def acc_false_positives(self):
        if self.all_false:
            stats = {
                'mean': 100,
                'std': None,
                'max': None,
                'count': None,
                'median': None}
            return stats 
        mask = np.invert(self.test_clefts_mask)
        false_positives = self.truth_clefts_edt[mask]
        stats = {
            'mean': np.mean(false_positives),
            'std': np.std(false_positives),
            'max': np.amax(false_positives),
            'count': false_positives.size,
            'median': np.median(false_positives)}
        return stats

    def acc_false_negatives(self):
        if self.all_false:
            stats = {
                'mean': 100,
                'std': None,
                'max': None,
                'count': None,
                'median': None}
            return stats 
        mask = np.invert(self.truth_clefts_mask)
        false_negatives = self.test_clefts_edt[mask]
        stats = {
            'mean': np.mean(false_negatives),
            'std': np.std(false_negatives),
            'max': np.amax(false_negatives),
            'count': false_negatives.size,
            'median': np.median(false_negatives)}
        return stats

    def all_false_check(self):
        self.all_false = (np.invert(self.test_clefts_mask).sum() == 0) or (np.invert(self.truth_clefts_mask).sum() == 0)

cal_filename = "predictionsTs"


def evaluate_one(pred,gt_label,file_name):

    clefts_evaluation = Clefts(pred, gt_label)

    false_positive_count = clefts_evaluation.count_false_positives()
    false_negative_count = clefts_evaluation.count_false_negatives()

    clefts_evaluation.all_false_check()

    false_positive_stats = clefts_evaluation.acc_false_positives()
    false_negative_stats = clefts_evaluation.acc_false_negatives()

    print (f"{file_name}:")
    print ("======")

    print ("\tfalse positives: " + str(false_positive_count))
    print ("\tfalse negatives: " + str(false_negative_count))

    print ("\tdistance to ground truth: " + str(false_positive_stats))
    print ("\tdistance to proposal    : " + str(false_negative_stats))
    
    return false_positive_count,false_negative_count,false_positive_stats,false_negative_stats

def evaluation(pred_dir,mode,c):
    result_dict = {
        "name": pred_dir,
        "mean": {},
        "detailed": {}}

    if mode == 'ab':
        names = ['sample_c.nii.gz']
        GT_dir="/media/ps/passport2/ltc/nnUNetv2/nnUNet_raw/Dataset061_CREMI/labelsTs"
    elif mode == 'abc':
        names = ['sample_a_test.nii.gz','sample_b_test.nii.gz','sample_c_test.nii.gz']
        GT_dir="/media/ps/passport2/ltc/nnUNetv2/nnUNet_raw/Dataset062_CREMI/labelsTs"
    elif mode == 'fafb':
        GT_dir="/media/ps/passport1/ltc/nnUNetv2/nnUNet_raw/Dataset062_CREMI/labelsTs_fafb"
        names = os.listdir(GT_dir)
    
    for name in names:
        try:
            gt_label = sitk.ReadImage(join(GT_dir, name))
            pred = sitk.ReadImage(join(pred_dir, name))
        except:
            continue
        
        if c:
            try:
                with open(os.path.join(pred_dir,cal_filename)+'.json', 'r') as json_file:
                    result_dict = json.load(json_file)
            except:
                pass
        if name not in result_dict['detailed'].keys():
            false_positive_count, false_negative_count, false_positive_stats, false_negative_stats = evaluate_one(pred,gt_label,name)
            result_dict['detailed'][name] = {
                "false positives": false_positive_count,
                "false negatives": false_negative_count,
                "distance to ground truth": false_positive_stats,
                "distance to proposal": false_negative_stats,
                "cremi score": (false_positive_stats['mean']+false_negative_stats['mean'])/2,
            }

        
    result_dict['mean']['average cremi score'] = np.array([result_dict['detailed'][name]['cremi score'] for name in names]).mean()
    print(f"average cremi score: {result_dict['mean']['average cremi score']}")

    with open(os.path.join(pred_dir,cal_filename)+'.json', 'w') as json_file:
        json.dump(result_dict, json_file, indent=4)
    return result_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred_dir', help="pred_exp_name",default="/media/ps/passport1/ltc/nnUNetv2/nnUNet_outputs/fafb_CREMI/3d_fullres/fold3/patch24_256_256_step0.5_chkfinal_down1.0_1.0_1.0/", required=False)
    parser.add_argument('--continue_evaluation', action='store_true', default=False)
    parser.add_argument('--mode',type=str,default='abc')
    
    args = parser.parse_args()
    assert args.mode in ['abc', 'ab', 'fafb']

    evaluation(pred_dir=args.pred_dir,mode=args.mode,c=args.continue_evaluation)
    
    
        
if __name__ == '__main__':
    main()