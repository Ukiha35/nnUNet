import sys, os
import numpy as np
import shutil
import json

if __name__ == '__main__':
    shutil.copytree("/media/ps/passport2/ltc/nnUNetv2/nnUNet_results/Dataset100_WORD/nnUNetTrainer__nnUNetPlans__3d_fullres/",
                "/media/ps/passport2/ltc/nnUNetv2/nnUNet_results/Dataset100_WORD/nnUNetTrainer__nnUNetPlans__3d_2stage/")
    os.remove("/media/ps/passport2/ltc/nnUNetv2/nnUNet_results/Dataset100_WORD/nnUNetTrainer__nnUNetPlans__3d_2stage/plans_ori.json")
    with open("/media/ps/passport2/ltc/nnUNetv2/nnUNet_results/Dataset100_WORD/nnUNetTrainer__nnUNetPlans__3d_2stage/plans.json",'r') as f:
        plan = json.load(f)
    plan["configurations"]['3d_2stage'] = {
            "inherits_from": "3d_fullres",
            "previous_stage": "3d_lowres",
            "properties": "/home/ps/ltc/CREMI/config/stage2/WORD_100_stage2.json"
            }
    with open("/media/ps/passport2/ltc/nnUNetv2/nnUNet_results/Dataset100_WORD/nnUNetTrainer__nnUNetPlans__3d_2stage/plans.json",'w') as f:
        json.dump(plan,f,indent=4)