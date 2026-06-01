import json

def patch_notebook():
    path = 'KPE_(moura2023)/KPE_Current.ipynb'
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'rules = [' in source and 'C4:  45 -> P3' in source:
                new_source = source.replace(
                    '( 90.0, p3_0,  -90.0, "C7:  90 -> P3"),',
                    '( 90.0, p1_0,  -90.0, "C7:  90 -> P1"),'
                )
                new_source = new_source.replace(
                    '(-90.0, p1_0,   90.0, "C5: -90 -> P1"),',
                    '(-90.0, p3_0,   90.0, "C5: -90 -> P3"),'
                )
                
                # Convert back to list of strings with newlines for Jupyter format
                lines = new_source.splitlines(True)
                cell['source'] = lines

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

if __name__ == '__main__':
    patch_notebook()
