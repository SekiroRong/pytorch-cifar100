import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from utils import trunc_normal_


class ViT(nn.Module):

    def __init__(self, features, num_class=100):
        super().__init__()
        self.embedding_layer = PositionEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            channels=in_channels,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            position_embedding_dropout=dropout_rate,
            cls_head=cls_head,
        )
        self.transformer
        self.post_transformer_ln
        self.cls_layer


    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.transformer(x)
        x = self.post_transformer_ln(x)
        x = self.cls_layer(x)
        return x



class PositionEmbedding(nn.Module):
    def __init__(
        self,
        image_size=(224, 224),
        patch_size=(16, 16),
        channels=3,
        embedding_dim=768,
        hidden_dims=None,
        position_embedding_dropout=None,
        cls_head=True,
    ):
        super(PositionEmbedding, self).__init__()

        image_height, image_width = image_size
        patch_height, patch_width = patch_size

        self.grid_size = (
            image_height // patch_height,
            image_width // patch_width,
        )
        num_patches = self.grid_size[0] * self.grid_size[1]

        if cls_head:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
            num_patches += 1

        # positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embedding_dim)
        )
        self.pos_drop = nn.Dropout(p=position_embedding_dropout)

        self.projection = nn.Sequential(
            nn.Conv2d(
                channels,
                embedding_dim,
                kernel_size=patch_size,
                stride=patch_size,
            ),
        )

        self.cls_head = cls_head

        self._init_weights()

    def _init_weights(self):
        # trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x):
        # paths for cls_token / position embedding
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)

        if self.cls_head:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        return self.pos_drop(x + self.pos_embed)


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_ratio=4.0,
        attn_dropout=0.0,
        dropout=0.0,
        qkv_bias=True,
        revised=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        assert isinstance(
            mlp_ratio, float
        ), "MLP ratio should be an integer for valid "
        mlp_dim = int(mlp_ratio * dim)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                num_heads=heads,
                                qkv_bias=qkv_bias,
                                attn_drop=attn_dropout,
                                proj_drop=dropout,
                            ),
                        ),
                        PreNorm(
                            dim,
                            FeedForward(dim, mlp_dim, dropout_rate=dropout,),
                        )
                        if not revised
                        else FeedForward(
                            dim, mlp_dim, dropout_rate=dropout, revised=True,
                        ),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return 