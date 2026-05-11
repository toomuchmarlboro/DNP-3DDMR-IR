import json

def patch_export_cells():
    with open('breastnet3d_v5.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            modified = False
            
            if 'export_all_patients_to_stl()' in source:
                # 1. Update export directory
                source = source.replace('Path("exported_stls")', 'Path("exported_stls_v5")')
                source = source.replace('exported_stls', 'exported_stls_v5')
                
                # 2. Update checkpoint path
                source = source.replace('checkpoints_3d/', 'checkpoints_3d_v5/')
                
                # 3. Add visual hull to dec()
                source = source.replace('vol = dec(enc(m5)).float()', 
                                      'hull = compute_visual_hull(m5, device)\n            vol = dec(enc(m5), hull).float()')
                modified = True
                
            if 'def evaluate_all_patients():' in source:
                # 1. Update export directory for plots (optional, but good for clarity)
                source = source.replace('Path("projection_plots")', 'Path("projection_plots_v5")')
                
                # 2. Update checkpoint path
                source = source.replace('checkpoints_3d/', 'checkpoints_3d_v5/')
                
                # 3. Add visual hull to dec()
                source = source.replace('vol = dec(enc(m5)).float()', 
                                      'hull = compute_visual_hull(m5, device)\n            vol = dec(enc(m5), hull).float()')
                modified = True
                
            if modified:
                cell['source'] = [line + '\n' for line in source.split('\n')]
                
    with open('breastnet3d_v5.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
if __name__ == "__main__":
    patch_export_cells()
    print("Notebook export cells updated to v5 successfully!")
