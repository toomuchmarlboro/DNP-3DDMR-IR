import json

notebook_path = r"c:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\DNP-3DDMR-IR\TherMAM-NeRF\thermalnerf_v1.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    source = "".join(cell['source'])
    
    if "UNET_CKPT" in source:
        new_source = []
        for line in source.split('\n'):
            if "UNET_CKPT" not in line and "USE_PRECOMPUTED_MASKS" not in line and "Optional: frozen UNet" not in line:
                new_source.append(line)
        
        # Clean up double empty lines created by deletion
        clean_source = "\n".join(new_source)
        clean_source = clean_source.replace("\n\n\n", "\n\n")
        
        lines = clean_source.split('\n')
        cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Removed UNET_CKPT from configuration.")
