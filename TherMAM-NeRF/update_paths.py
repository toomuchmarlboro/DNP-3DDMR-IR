import json
import re

notebook_path = r"c:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\DNP-3DDMR-IR\TherMAM-NeRF\thermalnerf_v1.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    source = "".join(cell['source'])
    
    # 1. Update paths
    if "DATA_ROOT        = './data'" in source:
        source = source.replace(
            "DATA_ROOT        = './data'           # <-- SET THIS\n",
            "TIFF_DIR         = '../data/organized_by_patient'\nUNET_DIR         = '../data/organized_by_patient_unet'\n"
        )
        source = source.replace(
            "UNET_CKPT        = './unet_best.pth'  # <-- path to frozen UNet checkpoint",
            "UNET_CKPT        = '../../UNET_Segmentation/breast_segmentation_unet_best_gpu.pth'"
        )
        # Fix the comments
        source = re.sub(r"# Expected structure:.*?#       LL_mask\.png\n", "", source, flags=re.DOTALL)
        
        lines = source.split('\n')
        cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

    # 2. Update discover_patients
    elif 'def discover_patients' in source:
        new_discover_code = """def get_view_key(filename):
    n = filename.lower()
    if 'right later' in n: return 'RL'
    if 'right obli'  in n: return 'RO'
    if 'frontal' in n or 'anterior' in n: return 'F'
    if 'left obliq'  in n: return 'LO'
    if 'left later'  in n: return 'LL'
    return None

def discover_patients_split(tiff_base, unet_base):
    from pathlib import Path
    tb = Path(tiff_base)
    ub = Path(unet_base)
    pd_ = {}
    
    for tp in tb.rglob('*.tiff'):
        parts = tp.relative_to(tb).parts
        if len(parts) < 3: continue
        pid, lab, fn = parts[0], parts[1], parts[-1]
        vk = get_view_key(fn)
        if not vk: continue
        key = (pid, lab)
        if key not in pd_: pd_[key] = {'tiffs': {}, 'masks': {}}
        pd_[key]['tiffs'][vk] = tp
        
    for mp in ub.rglob('*.png'):
        parts = mp.relative_to(ub).parts
        if len(parts) < 3: continue
        pid, lab, fn = parts[0], parts[1], parts[-1]
        vk = get_view_key(fn)
        if not vk: continue
        key = (pid, lab)
        if key in pd_:
            pd_[key]['masks'][vk] = mp
            
    patients = []
    for (pid, lab), d in pd_.items():
        if len(d['tiffs']) == 5 and len(d['masks']) == 5:
            patients.append({
                'id'   : pid,
                'tiffs': d['tiffs'],
                'masks': d['masks'],
            })
        else:
            print(f'  [SKIP] {pid} — missing views')
            
    patients.sort(key=lambda p: int(p['id'].split('_')[-1]) if '_' in p['id'] else p['id'])
    print(f'Found {len(patients)} complete patients.')
    return patients

patients = discover_patients_split(TIFF_DIR, UNET_DIR)"""
        
        source = re.sub(r'def discover_patients\(.*', new_discover_code, source, flags=re.DOTALL)
        
        lines = source.split('\n')
        cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook paths updated successfully.")
