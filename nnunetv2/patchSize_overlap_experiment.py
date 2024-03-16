import shutil
import os
import argparse
import pickle
import json
import SimpleITK as sitk
from tqdm import tqdm

dataset_dict = {61:'CREMI',100:'WORD'}


def input_resample(folder_path, output_folder_path,downsample):
    
    os.makedirs(output_folder_path,exist_ok=True)
    
    # 遍历文件夹中的.nii.gz文件
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".nii.gz"):
            # 构建完整的文件路径
            file_path = os.path.join(folder_path, filename)
        # 读取.nii.gz文件
        pred_itk = sitk.ReadImage(file_path)

        # 获取原始图像的尺寸和间隔
        original_size = pred_itk.GetSize()
        original_spacing = pred_itk.GetSpacing()

        # 定义新的尺寸和间隔
        new_size = [int(size / downsample[axis]) for axis,size in enumerate(original_size)]
        new_spacing = [spacing * downsample[axis] for axis,spacing in enumerate(original_spacing)]

        # 使用Resample函数进行降采样
        resampled_img = sitk.Resample(pred_itk, new_size, sitk.Transform(), sitk.sitkNearestNeighbor, pred_itk.GetOrigin(),
                                    new_spacing, pred_itk.GetDirection(), 0.0, pred_itk.GetPixelID())# sitk.sitkLinear
        
        output_file_path = os.path.join(output_folder_path, filename)
        
        # 将结果保存到新的.nii.gz文件
        sitk.WriteImage(resampled_img, output_file_path)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_num", default=0, required=False)
    parser.add_argument("--fold", '-f', default=0, required=False)
    parser.add_argument("--datasetnum", type=int, default=100, required=False)
    parser.add_argument('--patch_size', type=int, nargs='+', default=[64,192,160], required=False)
    parser.add_argument("--step", default=0.5, required=False)
    parser.add_argument("--overwrite", default=False, action="store_true", required=False)
    parser.add_argument("--chk", default="checkpoint_final.pth", required=False)
    parser.add_argument("--config", default="2d", required=False)
    parser.add_argument("--downsample", type=float, nargs='+', default=[1,1,1], required=False)
    args = parser.parse_args()
    
    assert args.datasetnum in dataset_dict
    if args.datasetnum == 61 and args.config == "2d":
        assert args.downsample == [1,1,1]
    
    
    print(f"checkpoint: %s" % args.chk)
    print(f"fold: %s" % args.fold)
    print(f"patch size: %s" % args.patch_size)
    print(f"step: %s" % args.step)
    print(f"downsample: %s" % args.downsample)
    
    ori_input_dir = f"/media/ps/passport2/ltc/nnUNetv2/nnUNet_raw/Dataset{args.datasetnum:03d}_{dataset_dict[args.datasetnum]}/imagesTs"
    
    if args.config == "2d":
        input_dir = f"/media/ps/passport2/ltc/nnUNetv2/nnUNet_raw/Dataset{args.datasetnum:03d}_{dataset_dict[args.datasetnum]}/imagesTs_{args.downsample[0]}_{args.downsample[1]}_{args.downsample[2]}"
        if not os.path.exists(input_dir):
            print("creating resampled input directory...")
            input_resample(ori_input_dir,input_dir,args.downsample)
        else:
            print("resampled input directory already exists, start predicting...")
    else:
        input_dir = ori_input_dir

    print("creating settings...")
    
    output_dir = f"/media/ps/passport2/ltc/nnUNetv2/nnUNet_outputs/{dataset_dict[args.datasetnum]}"
       
    # 生成settings
    settings_dir = f"/media/ps/passport2/ltc/nnUNetv2/nnUNet_results/Dataset{args.datasetnum:03d}_{dataset_dict[args.datasetnum]}"
    
    ori_plan_file = os.path.join(settings_dir,f"nnUNetTrainer__nnUNetPlans__{args.config}",f"plans_ori.json")
    plan_file = os.path.join(settings_dir,f"nnUNetTrainer__nnUNetPlans__{args.config}",f"plans.json")
    
    with open(ori_plan_file, 'r') as f:
        data = json.load(f)

    if args.config == '2d':
        data['configurations']['2d']['spacing'] = [s*args.downsample[axis+1] for axis, s in enumerate(data['configurations']['2d']['spacing'])]
        data['configurations'][args.config]['patch_size'] = args.patch_size
    elif args.config == '3d_cascade_fullres':
        data['configurations']['3d_fullres']['spacing'] = [s*args.downsample[axis] for axis, s in enumerate(data['configurations']['3d_fullres']['spacing'])]
        data['configurations']['3d_fullres']['patch_size'] = args.patch_size
    else:
        data['configurations'][args.config]['spacing'] = [s*args.downsample[axis] for axis, s in enumerate(data['configurations'][args.config]['spacing'])]
        data['configurations'][args.config]['patch_size'] = args.patch_size
    
    with open(plan_file, 'w') as new_file:
        json.dump(data, new_file, indent=4)

    # nnUNet_predict 命令
    if args.config == '2d':
        output_folder = os.path.join(args.config,f"fold{args.fold}",f"patch{args.patch_size[0]}_{args.patch_size[1]}_step{args.step}_chk{args.chk.split('.')[0].split('_')[-1]}_down{args.downsample[0]}_{args.downsample[1]}_{args.downsample[2]}")
    else:
        output_folder = os.path.join(args.config,f"fold{args.fold}",f"patch{args.patch_size[0]}_{args.patch_size[1]}_{args.patch_size[2]}_step{args.step}_chk{args.chk.split('.')[0].split('_')[-1]}_down{args.downsample[0]}_{args.downsample[1]}_{args.downsample[2]}")
           
    if args.overwrite:
        if os.path.exists(os.path.join(output_dir,output_folder)):
            print(f"removing {os.path.join(output_dir,output_folder)}")
            shutil.rmtree(os.path.join(output_dir,output_folder))
        overwrite = ''
    else:
        if os.path.exists(os.path.join(output_dir,output_folder)):
            print(f"The folder '{output_folder}' already exists. Continuing process...")
        overwrite = '--continue_prediction'
    if args.config == '3d_cascade_fullres':
        prev_output = '-prev_stage_predictions '+os.path.join(settings_dir,f'nnUNetTrainer__nnUNetPlans__3d_cascade_fullres/fold_{args.fold}/validation/')
    else:
        prev_output = ''
    
    command_predict = f"CUDA_VISIBLE_DEVICES={args.cuda_num}  nnUNetv2_predict --save_probabilities {prev_output} {overwrite} -chk {args.chk} --continue_prediction -i {input_dir} -o {os.path.join(output_dir,output_folder)} -d {args.datasetnum} -c {args.config} -f {args.fold} -step_size {args.step}"
    os.system(command_predict)

    # os.remove(plan_file)
    
    # nnUNet_WORDEvaluation 命令
    command_evaluation = f"nnUNetv2_{dataset_dict[args.datasetnum]}Evaluation -p {output_folder}"
        
    os.system(command_evaluation)

if __name__ == "__main__":
    main()