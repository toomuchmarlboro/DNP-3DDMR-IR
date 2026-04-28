import json
import re

file_path = 'KPE_Current.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    d = json.load(f)

for cell in d['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Replace extract_body_edge logic in generate_9_curves_3d
        if 'def generate_9_curves_3d' in source:
            new_source = source
            
            # Helper function to insert
            helper = """
def extract_body_edge_from_contour(contour, start_pt):
    idx = np.argmin(np.sum((contour - start_pt)**2, axis=1))
    n = len(contour)
    
    forward_y = contour[(idx + 10) % n, 1]
    backward_y = contour[(idx - 10 + n) % n, 1]
    
    step = 1 if forward_y < backward_y else -1
        
    path = []
    curr = idx
    min_y = np.min(contour[:, 1])
    while True:
        pt = contour[curr]
        path.append(pt)
        if pt[1] <= min_y + 5:
            break
        curr = (curr + step + n) % n
        if len(path) > n:
            break
            
    return np.array(path)

"""
            # Insert helper before generate_9_curves_3d
            new_source = new_source.replace('def generate_9_curves_3d', helper + 'def generate_9_curves_3d')
            
            # Replace C1 and C2 extraction
            c1_c2_old = """    # C1: all points RIGHT of P1 (Alg-1: includes breast + right body edge)
    c1_mask = contour_0[:, 0] >= p1_0[0]
    c1_pts = longest_contiguous_segment(
        np.column_stack((contour_0, np.zeros(len(contour_0)))), c1_mask)

    # C2: all points LEFT of P3 (Alg-1: includes breast + left body edge)
    c2_mask = contour_0[:, 0] <= p3_0[0]
    c2_pts = longest_contiguous_segment(
        np.column_stack((contour_0, np.zeros(len(contour_0)))), c2_mask)"""
            
            c1_c2_new = """    # C1: Right body edge (from P1 upwards)
    c1_2d = extract_body_edge_from_contour(contour_0, p1_0)
    c1_pts = np.column_stack((c1_2d, np.zeros(len(c1_2d))))

    # C2: Left body edge (from P3 upwards)
    c2_2d = extract_body_edge_from_contour(contour_0, p3_0)
    c2_pts = np.column_stack((c2_2d, np.zeros(len(c2_2d))))"""
            
            new_source = new_source.replace(c1_c2_old, c1_c2_new)
            
            # Replace C4-C9 rules
            rules_old = """    # C4-C9: rotated side-view curves
    rules = [
        ( 45.0, p3_0,   45.0, "C4:  45 -> P3"),
        (-90.0, p3_0,  -90.0, "C5: -90 -> P3"),
        (-45.0, p1_0,  -45.0, "C6: -45 -> P1"),
        ( 90.0, p1_0,   90.0, "C7:  90 -> P1"),
        ( 45.0, p3_0,  135.0, "C8: aux 135 -> P3"),
        (-45.0, p1_0, -135.0, "C9: aux-135 -> P1"),
    ]"""
            
            rules_new = """    # C4-C9: rotated side-view curves
    # Fix: Right breast views (45, 90) -> anchor P3 (min X). Rotate by -src_angle
    # Fix: Left breast views (-45, -90) -> anchor P1 (max X). Rotate by -src_angle
    rules = [
        ( 45.0, p3_0,  -45.0, "C4:  45 -> P3"),
        ( 90.0, p3_0,  -90.0, "C7:  90 -> P3"),
        (-45.0, p1_0,   45.0, "C6: -45 -> P1"),
        (-90.0, p1_0,   90.0, "C5: -90 -> P1"),
        ( 45.0, p3_0, -135.0, "C8: aux -135 -> P3"),
        (-45.0, p1_0,  135.0, "C9: aux 135 -> P1"),
    ]"""
            new_source = new_source.replace(rules_old, rules_new)
            
            # Split lines and reconstruct to ensure it's valid JSON
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
            # Remove trailing newline on last line
            if cell['source']:
                cell['source'][-1] = cell['source'][-1].rstrip('\n')

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=1)
