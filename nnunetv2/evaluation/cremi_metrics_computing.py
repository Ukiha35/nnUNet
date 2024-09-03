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
import multiprocessing 
import time
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
        if self.all_false == 1:
            stats = {
                'mean': 100,
                'std': None,
                'max': None,
                'count': None,
                'median': None}
            return stats 
        elif self.all_false == 2:
            stats = {
                'mean': 0,
                'std': 0,
                'max': 0,
                'count': 0,
                'median': 0}
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
        if self.all_false == 1:
            stats = {
                'mean': 100,
                'std': None,
                'max': None,
                'count': None,
                'median': None}
            return stats 
        elif self.all_false == 2:
            stats = {
                'mean': 0,
                'std': 0,
                'max': 0,
                'count': 0,
                'median': 0}
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
        self.all_false = int(np.invert(self.test_clefts_mask).sum() == 0) + int(np.invert(self.truth_clefts_mask).sum() == 0)

cal_filename = "predictionsTs"


def evaluate_one(pred,gt_label,file_name):
    print(f"evaluating {file_name}...")

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
    print ("\tcremi score             : " + str((false_negative_stats['mean']+false_positive_stats['mean'])/2))
    
    return false_positive_count,false_negative_count,false_positive_stats,false_negative_stats

def evaluate_one_and_save(pred,gt_label,file_name,pred_dir):
    
    false_positive_count, false_negative_count, false_positive_stats, false_negative_stats = evaluate_one(pred,gt_label,file_name)
    with open(os.path.join(pred_dir,cal_filename)+'.json', 'r') as json_file:
        result_dict = json.load(json_file)
    result_dict['detailed'][file_name] = {
        "false positives": false_positive_count,
        "false negatives": false_negative_count,
        "distance to ground truth": false_positive_stats,
        "distance to proposal": false_negative_stats,
        "cremi score": (false_positive_stats['mean']+false_negative_stats['mean'])/2}
    with open(os.path.join(pred_dir,cal_filename)+'.json', 'w') as json_file:
        json.dump(result_dict, json_file, indent=4)
        
def evaluation(pred_dir,c,num_workers):

    names = ['sample_a_test.nii.gz','sample_b_test.nii.gz','sample_c_test.nii.gz']
    GT_dir="/media/ps/passport2/ltc/nnUNetv2/nnUNet_raw/Dataset062_CREMI/labelsTs"
    
    with multiprocessing.get_context("spawn").Pool(num_workers) as export_pool:
        tasks_record = []
        
        if not c or not os.path.exists(os.path.join(pred_dir,cal_filename)+'.json'):
            result_dict = {
                "name": pred_dir,
                "mean": {},
                "detailed": {}}
            with open(os.path.join(pred_dir,cal_filename)+'.json', 'w') as json_file:
                json.dump(result_dict, json_file, indent=4)

        for name in sorted(names):
            try:
                gt_label = sitk.ReadImage(join(GT_dir, name))
                pred = sitk.ReadImage(join(pred_dir, name))
            except:
                continue
            
            if name not in result_dict['detailed'].keys():
                while len([r for r in tasks_record if not r.ready()]) >= num_workers:
                    time.sleep(0.1)  
                r = export_pool.starmap_async(evaluate_one_and_save, ((pred,gt_label,name,pred_dir),))
                tasks_record.append(r)
                
        ret = [i.get() for i in tasks_record]
    
    with open(os.path.join(pred_dir,cal_filename)+'.json', 'r') as json_file:
        result_dict = json.load(json_file)
    result_dict['mean']['average cremi score'] = np.array([result_dict['detailed'][name]['cremi score'] for name in names]).mean()
    with open(os.path.join(pred_dir,cal_filename)+'.json', 'w') as json_file:
        json.dump(result_dict, json_file, indent=4)
        
    print(f"average cremi score: {result_dict['mean']['average cremi score']}")
    
    return result_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred_dir', help="pred_exp_name",default="/media/ps/passport2/ltc/nnUNetv2/nnUNet_outputs/CREMI/3d_lowres/fold2/patch32_160_160_step0.5_chkfinal_down1.0_1.0_1.0/stage_2/save_stage2_roi_th_0.001_level_sample_3.0_16.0_16.0_patch_12_128_128_center_canvas_12_128_128_shuffle_False_nms_0.75_prior_False_expand0_itc_4.0_pix_6.0_child_nmm_0.35_crop_12_128_128_max_fullres/", required=False)
    parser.add_argument('--continue_evaluation', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=5)
    
    args = parser.parse_args()

    evaluation(pred_dir=args.pred_dir,c=args.continue_evaluation,num_workers=args.num_workers)
    
    
        
if __name__ == '__main__':
    main()