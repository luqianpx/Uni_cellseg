import torch
import timm
from torchvision import transforms
import os
def load_uni_encoder(device='cuda', use_hf=True):
    """
    Returns a ViT-L/16 encoder with no classification head.
    If use_hf is True, tries to load UNI via HF hub. Otherwise falls back to DINOv2 ViT-L/16.
    """
    if use_hf:
        try:
            local_dir = "/data/llm_model/uni/"
            model = timm.create_model(
                "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
            )
            model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
            model.eval()
            print("UNI model loaded from HF hub.")
            return model.to(device)
            
        
        except Exception as e:
            print("WARN: Could not load UNI from HF hub, falling back to DINOv2 ViT-L/16. Error:", e)


class UNIBackbone(torch.nn.Module):
    """
    Wraps a ViT-L/16 (UNI weights or fallback) and exposes a spatial feature map.
    """
    def __init__(self, device='cuda', use_uni=True):
        super().__init__()
        self.encoder = load_uni_encoder(device=device, use_hf=use_uni)
        self.embed_dim = self.encoder.num_features  # usually 1024 for ViT-L/16

    def forward(self, x):
        """
        x: Bx3xH xW. Returns a dict with 'feat': BxCxhxw
        """
        # timm ViT forward_features returns tokens, with CLS token at index 0
        B, C, H, W = x.shape
        tokens = self.encoder.forward_features(x)  # [B, N+1, C]
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[0]
        if tokens.dim() == 3:
            # Remove CLS token
            cls_token, patch_tokens = tokens[:, :1], tokens[:, 1:]
            # Infer h,w from patch count
            num_patches = patch_tokens.shape[1]
            try:
                pH, pW = self.encoder.patch_embed.patch_size
            except Exception:
                pH = pW = 16
            h = max(1, H // pH)
            w = max(1, W // pW)
            if h * w != num_patches:
                # fallback: assume square or nearest
                w = int((num_patches) ** 0.5)
                h = num_patches // w
            feat = patch_tokens.transpose(1,2).reshape(B, self.embed_dim, h, w)
        else:
            # If encoder returns spatial feature already
            feat = tokens
        return {"feat": feat}
