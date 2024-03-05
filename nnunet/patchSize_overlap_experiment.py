import shutil
import os
import argparse
import pickle
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_num", default=0, required=False)
    parser.add_argument('--settings_dir', default="/media/ps/passport2/ltc/nnUNet/nnUNet_trained_models/nnUNet/3d_fullres/Task100_WORD/nnUNetTrainerV2__nnUNetPlansv2.1/", required=False)
    parser.add_argument('--input_dir', default="$nnUNet_raw_data_base/nnUNet_raw_data/Task100_WORD/imagesTs/", required=False)
    parser.add_argument('--output_dir', default="/media/ps/passport2/ltc/nnUNet/nnUNet_outputs/", required=False)
    parser.add_argument('--settings_template', default="3Dv2", required=False)
    parser.add_argument('--settings_fold', default=1, required=False)
    parser.add_argument('--patch_size', type=int, nargs='+', default=[64,160,160], required=False)
    parser.add_argument("--overlap", default=0.5, required=False)
    parser.add_argument("--overwrite", default=False, action="store_true", required=False)
    args = parser.parse_args()
    
    print(f"settings fold: %s" % args.settings_fold)
    print(f"patch size: %s" % args.patch_size)
    print(f"overlap: %s" % args.overlap)
    print("creating settings...")
    
    
    # 拷贝文件夹
    try:
        source_folder = os.path.join(args.settings_dir,args.settings_template)
    except:
        raise FileNotFoundError(f"The folder '{source_folder}' does not exist.")
    
    destination_folder = os.path.join(args.settings_dir,f"fold_{args.settings_fold}")
    if os.path.exists(destination_folder):
        if args.overwrite:
            print(f"removing {destination_folder}")
            shutil.rmtree(destination_folder)
        else:
            raise FileExistsError(f"The folder '{destination_folder}' already exists.")
    
    shutil.copytree(source_folder, destination_folder)
    with open(os.path.join(destination_folder,f"3Dv2_{args.patch_size[0]}_{args.patch_size[1]}_{args.patch_size[2]}_{args.overlap}.txt"), 'w') as file:
        pass
    
    
    # 打开文件并读取数据
    pkl_file_path = os.path.join(destination_folder,"model_final_checkpoint.model.pkl")
    with open(pkl_file_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)

    data['plans']['plans_per_stage'][1]['patch_size'] = args.patch_size

    with open(pkl_file_path, 'wb') as new_file:
        pickle.dump(data, new_file)


    # nnUNet_predict 命令
    output_folder = f"3Dv2_{args.patch_size[0]}_{args.patch_size[1]}_{args.patch_size[2]}_{args.overlap}"
    if os.path.exists(os.path.join(args.output_dir,output_folder)):
        if args.overwrite:
            print(f"removing {os.path.join(args.output_dir,output_folder)}")
            shutil.rmtree(os.path.join(args.output_dir,output_folder))
        else:
            raise FileExistsError(f"The folder '{output_folder}' already exists.")
    command_predict = f"CUDA_VISIBLE_DEVICES={args.cuda_num}  nnUNet_predict -i {args.input_dir} -o {os.path.join(args.output_dir,output_folder)} -t 100 -m 3d_fullres -f {args.settings_fold} --save_npz --step_size {args.overlap}"
    os.system(command_predict)


    # nnUNet_WORDEvaluation 命令
    command_evaluation = f"nnUNet_WORDEvaluation -m calculate -p {output_folder}"
    os.system(command_evaluation)

if __name__ == "__main__":
    main()