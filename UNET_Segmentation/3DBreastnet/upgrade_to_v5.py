import json

def patch_notebook():
    with open('breastnet3d_v5.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # 1. Patch Model Definitions Cell
            if 'class Decoder3D(nn.Module):' in source:
                # Replace Decoder3D forward to accept visual_hull
                new_dec = """class Decoder3D(nn.Module):
    def __init__(self, drop=0.25):
        super().__init__()
        self.fc = nn.Linear(1000, 512*2*2*2)
        self.up1=nn.ConvTranspose3d(512,256,2,stride=2); self.d1=DoubleConv3D(256,256,drop)
        self.up2=nn.ConvTranspose3d(256,128,2,stride=2); self.d2=DoubleConv3D(128,128,drop)
        self.up3=nn.ConvTranspose3d(128,64,2,stride=2);  self.d3=DoubleConv3D(64,64,drop)
        self.up4=nn.ConvTranspose3d(64,32,2,stride=2);   self.d4=DoubleConv3D(32,32,0)
        self.up5=nn.ConvTranspose3d(32,16,2,stride=2);   self.d5=DoubleConv3D(16,16,0)
        self.up6=nn.ConvTranspose3d(16,8,2,stride=2);    self.d6=DoubleConv3D(8,8,0)
        
        # v5: Output fusion layer to inject the visual hull
        self.fusion = nn.Sequential(
            nn.Conv3d(8 + 1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(8), nn.ReLU(True),
            nn.Conv3d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.apply(_init)
        nn.init.constant_(self.fusion[2].bias, -4.0)

    def _s4(self, x): return self.d4(self.up4(x))
    def _s5(self, x): return self.d5(self.up5(x))
    def _s6(self, x): return self.d6(self.up6(x))

    def forward(self, x, visual_hull):
        x = self.fc(x).view(x.size(0), 512, 2, 2, 2)
        x = self.d1(self.up1(x))
        x = self.d2(self.up2(x))
        x = self.d3(self.up3(x))
        if x.requires_grad:
            x = torch.utils.checkpoint.checkpoint(self._s4, x, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self._s5, x, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self._s6, x, use_reentrant=False)
        else:
            x = self._s4(x); x = self._s5(x); x = self._s6(x)
            
        # v5: Inject visual hull
        x = torch.cat([x, visual_hull], dim=1)
        return self.fusion(x)
"""
                # Find the old Decoder3D class and replace it
                start_idx = source.find('class Decoder3D(nn.Module):')
                end_idx = source.find('# ── Differentiable projection')
                source = source[:start_idx] + new_dec + "\n" + source[end_idx:]
                
                # Add boundary loss and visual hull functions
                additions = """
# ── v5 Additions ─────────────────────────────────────────────
def compute_visual_hull(m5, device):
    \"\"\"Computes a 3D visual hull intersection from the 5 view masks\"\"\"
    B = m5.shape[0]
    D = H = W = 128
    # Create 3D grid in range [-1, 1]
    z, y, x = torch.meshgrid(
        torch.linspace(-1, 1, D, device=device),
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    grid_pts = torch.stack([x, y, z], dim=0).view(3, -1)
    
    angles = [-90., -45., 0., 45., 90.]
    hull = torch.ones((B, 1, D*H*W), device=device)
    
    for i, angle in enumerate(angles):
        rad = angle * math.pi / 180.0
        c, s = math.cos(rad), math.sin(rad)
        X_cam = grid_pts[0]*c - grid_pts[2]*s
        Y_cam = grid_pts[1]
        
        sample_coords = torch.stack([X_cam, Y_cam], dim=-1).unsqueeze(0).unsqueeze(2).expand(B, -1, -1, -1)
        mask_view = m5[:, i:i+1, :, :]
        sampled = F.grid_sample(mask_view, sample_coords, mode='bilinear', padding_mode='zeros', align_corners=True)
        hull = hull * sampled.squeeze(3)
        
    return hull.view(B, 1, D, H, W)

def boundary_loss(pred, target):
    \"\"\"Simple Laplacian edge loss to force curve alignment\"\"\"
    weight = torch.tensor([[[[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]]], device=pred.device)
    pred_edge = F.conv2d(pred, weight, padding=1)
    target_edge = F.conv2d(target, weight, padding=1)
    return F.l1_loss(torch.abs(pred_edge), torch.abs(target_edge))

"""
                source = source + additions
                cell['source'] = [line + '\n' for line in source.split('\n')]
                
            # 2. Patch Training Cell
            if 'def train(cfg):' in source:
                # Inject visual hull computation
                source = source.replace('vol = dec(enc(m5))', 
                                      'hull = compute_visual_hull(m5, device)\n                vol = dec(enc(m5), hull)')
                
                # Add boundary loss
                source = source.replace('loss = loss + dice_loss(render_projection(vol, th),',
                                      'proj = render_projection(vol, th)\n                    gt_mask = m5[:, i:i+1]\n                    dl = dice_loss(proj, gt_mask)\n                    bl = boundary_loss(proj, gt_mask)\n                    loss = loss + dl + (2.0 * bl)')
                source = source.replace('m5[:, i:i+1])', '') # cleanup the replaced line suffix
                
                # Add boundary loss in val loop
                source = source.replace('dl = dice_loss(proj, m5[:, i:i+1])',
                                      'dl = dice_loss(proj, m5[:, i:i+1])\n                    bl = boundary_loss(proj, m5[:, i:i+1])')
                
                # Update checkpoint dir
                source = source.replace('"checkpoints_3d"', '"checkpoints_3d_v5"')
                
                cell['source'] = [line + '\n' for line in source.split('\n')]
                
    with open('breastnet3d_v5.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
if __name__ == "__main__":
    patch_notebook()
    print("Notebook upgraded to v5 successfully!")
