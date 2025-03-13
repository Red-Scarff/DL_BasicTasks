import torch
import torch.nn as nn
from einops import rearrange

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, 
                                   kernel_size=patch_size, 
                                   stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, int(mlp_ratio * embed_dim))
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # 分块嵌入 [B, C, H, W] -> [B, N+1, D]
        x = self.patch_embed(x)  # [B, D, H/P, W/P]
        x = rearrange(x, 'b d h w -> b (h w) d')
        
        # 添加类别标记和位置编码
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.pos_embed
        
        # Transformer编码器
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x[:, 0]  # 返回类别标记

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

# 创建模型实例
vit = ViT()

# 打印模型结构
print(vit)
