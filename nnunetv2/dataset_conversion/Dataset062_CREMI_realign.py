import numpy as np
import os
import SimpleITK as sitk

def save_cremi_sitk(slide, dir, spacing=(4.0,4.0,40.0)):
    new_slide_itk = sitk.GetImageFromArray(np.array(slide,dtype=np.uint8))
    new_slide_itk.SetSpacing(spacing)
    sitk.WriteImage(new_slide_itk, dir)




if __name__ == '__main__':
    dir = "/media/ps/passport2/ltc/nnUNetv2/nnUNet_raw/Dataset062_CREMI"
    file_names = ["sample_a","sample_b","sample_c","sample_d","sample_e"]
    for file_name in file_names:
        slide_itk = sitk.ReadImage(os.path.join(dir,'imagesTr',file_name+'_0000.nii.gz'))
        slide = sitk.GetArrayFromImage(slide_itk)
        slide_train = slide[:100]
        slide_test = slide[100:]
        save_cremi_sitk(slide_train,os.path.join(dir,'imagesTr',file_name+'_train_0000.nii.gz'))
        save_cremi_sitk(slide_test,os.path.join(dir,'imagesTr',file_name+'_test_0000.nii.gz'))
        save_cremi_sitk(slide_test,os.path.join(dir,'imagesTs',file_name+'_test_0000.nii.gz'))

        label_itk = sitk.ReadImage(os.path.join(dir,'labelsTr',file_name+'.nii.gz'))
        label = sitk.GetArrayFromImage(label_itk)
        label_train = label[:100]
        label_test = label[100:]
        save_cremi_sitk(label_train,os.path.join(dir,'labelsTr',file_name+'_train.nii.gz'))
        save_cremi_sitk(label_test,os.path.join(dir,'labelsTr',file_name+'_test.nii.gz'))
        save_cremi_sitk(label_test,os.path.join(dir,'labelsTs',file_name+'_test.nii.gz'))
        


    print(1)

    
    
    
    
    
    