import torch
import torch.nn as nn
import torch.onnx 
import onnx 
import onnxruntime
import numpy as np

class ViT(nn.Module):

    def __init__(self, 
        image_size=(32, 32),
        patch_size=(4, 4),
        in_channels=3,
        embedding_dim=192,
        num_layers=12,
        num_heads=12,
        qkv_bias=True,
        mlp_ratio=4.0,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        hidden_dims=None,
        cls_head=False,
        num_classes=100,):
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
        # transformer
        self.transformer = Transformer(
            dim=embedding_dim,
            depth=num_layers,
            heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout_rate,
            dropout=dropout_rate,
            qkv_bias=qkv_bias,
        )
        self.post_transformer_ln = nn.LayerNorm(embedding_dim)

        # output layer
        self.cls_layer = OutputLayer(
            embedding_dim,
            num_classes=num_classes,
            cls_head=cls_head,
        )


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
                        ),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x



class Attention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0
    ):
        super(Attention, self).__init__()

        assert (
            dim % num_heads == 0
        ), "Embedding dimension should be divisible by number of heads"

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape # N querys(16*16)
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, torch.div(C, self.num_heads, rounding_mode='trunc'))
            .permute(2, 0, 3, 1, 4)
        )
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FeedForward(nn.Module):
    """
    Implementation of MLP for transformer
    """

    def __init__(self, dim, hidden_dim, dropout_rate=0.0):
        super(FeedForward, self).__init__()
        """
        Original: https://arxiv.org/pdf/2010.11929.pdf
        """
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
        )
        self._init_weights()

    def _init_weights(self):
        for name, module in self.net.named_children():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.bias, std=1e-6)

    def forward(self, x):
        x = self.net(x)

        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class OutputLayer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_classes=100,
        cls_head=False,
    ):
        super(OutputLayer, self).__init__()

        self.num_classes = num_classes
        modules = []
        modules.append(nn.Linear(embedding_dim, num_classes))

        self.net = nn.Sequential(*modules)

        if cls_head:
            self.to_cls_token = nn.Identity()

        self.cls_head = cls_head
        self.num_classes = num_classes
        self._init_weights()

    def _init_weights(self):
        for name, module in self.net.named_children():
            if isinstance(module, nn.Linear):
                if module.weight.shape[0] == self.num_classes:
                    nn.init.zeros_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        if self.cls_head:
            x = self.to_cls_token(x[:, 0])
        else:
            """
            Scaling Vision Transformer: https://arxiv.org/abs/2106.04560
            """
            x = torch.mean(x, dim=1)

        return self.net(x)






if __name__ == '__main__':
    weights = '/home/sekiro/pytorch-cifar100/checkpoint/vit/Wednesday_15_March_2023_00h_22m_43s/vit-30-regular.pth'
    model = ViT()
    model.load_state_dict(torch.load(weights))
    dummy_input = torch.randn(1, 3, 32, 32) 
    model(dummy_input)
    with torch.no_grad(): 
        torch.onnx.export( 
            model, 
            dummy_input, 
            "vit.onnx", 
            opset_version=14, 
            input_names=['input'], 
            output_names=['output'])
    
    onnx_model = onnx.load("vit.onnx") 
    try: 
        onnx.checker.check_model(onnx_model) 
    except Exception: 
        print("Model incorrect") 
    else: 
        print("Model correct")

    dummy_input_np = np.random.rand(1, 3, 32, 32).astype(np.float32)
    ort_session = onnxruntime.InferenceSession("vit.onnx")
    ort_inputs = {'input': dummy_input_np}
    ort_output = ort_session.run(['output'], ort_inputs)[0]
    print(ort_output)