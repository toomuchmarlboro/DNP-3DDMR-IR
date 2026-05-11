import json
import re

def fix_all_paths():
    with open('breastnet3d_v5.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # 1. Fix Linux absolute paths to relative
            source = re.sub(r'r"/mnt/Data1/Peoples/faiz836b/DNP-3DDMR-IR/UNET_Segmentation/breast_segmentation_unet_best_gpu.pth"',
                            'r"../../breast_segmentation_unet_best_gpu.pth"', source)
            source = re.sub(r'r"/mnt/Data1/Peoples/faiz836b/DNP-3DDMR-IR/data/organized_by_patient"',
                            'r"../../data/organized_by_patient"', source)
            source = re.sub(r'r"/mnt/Data1/Peoples/faiz836b/DNP-3DDMR-IR/UNET_Segmentation/3DBreastnet/checkpoints_3d_v5/3dbreastnet_best.pth"',
                            'Path("checkpoints_3d_v5/3dbreastnet_best.pth")', source)
            source = re.sub(r'r"/mnt/Data1/Peoples/faiz836b/DNP-3DDMR-IR/UNET_Segmentation/3DBreastnet/checkpoints_3d/3dbreastnet_best.pth"',
                            'Path("checkpoints_3d_v5/3dbreastnet_best.pth")', source)
                            
            # 2. Fix hardcoded Windows paths
            source = re.sub(r'r"C:\\Users\\LENOVO THINKPAD T14\\Documents\\PROPOSAL TA\\files\\Rodriguez-Guerrero Dataset\\Breast Thermography\\3D Reconstruction\\UNET_Segmentation\\3DBreastnet\\checkpoints_3d\\3dbreastnet_best.pth"',
                            'Path("checkpoints_3d_v5/3dbreastnet_best.pth")', source)

            # 3. Ensure checkpoints_3d goes to v5 (prevent duplicates)
            source = source.replace('checkpoints_3d_v5/', 'checkpoints_3d/')
            source = source.replace('checkpoints_3d/', 'checkpoints_3d_v5/')
            source = source.replace('checkpoints_3d\\\\', 'checkpoints_3d_v5\\\\')

            # 4. Ensure exported_stls goes to v5 (prevent _v5_v5)
            source = source.replace('exported_stls_v5_v5', 'exported_stls')
            source = source.replace('exported_stls_v5', 'exported_stls')
            source = source.replace('exported_stls', 'exported_stls_v5')
            
            # 5. Fix projection_plots goes to v5
            source = source.replace('projection_plots_v5', 'projection_plots')
            source = source.replace('projection_plots', 'projection_plots_v5')

            cell['source'] = [line + '\n' for line in source.split('\n')]

    with open('breastnet3d_v5.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    fix_all_paths()
    print("Notebook paths fixed successfully!")
