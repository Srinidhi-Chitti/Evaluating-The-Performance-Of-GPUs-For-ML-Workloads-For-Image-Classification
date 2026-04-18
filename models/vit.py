import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
import math
from jax import random


class Patches(nn.Module):
    patch_size: int
    embed_dim: int

    def setup(self):
        self.conv = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding='VALID'
        )

    def __call__(self, images):
        patches = self.conv(images)
        b, h, w, c = patches.shape
        patches = jnp.reshape(patches, (b, h * w, c))
        return patches


class PatchEncoder(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        n, seq_len, _ = x.shape

        x = nn.Dense(self.hidden_dim)(x)

        cls = self.param('cls_token', nn.initializers.zeros, (1, 1, self.hidden_dim))
        cls = jnp.tile(cls, (n, 1, 1))
        x = jnp.concatenate([cls, x], axis=1)

        pos_embed = self.param(
            'pos_embedding',
            nn.initializers.normal(stddev=0.02),
            (1, seq_len + 1, self.hidden_dim)
        )

        return x + pos_embed


class MLP(nn.Module):
    mlp_dim: int
    drop_p: float
    out_dim: int

    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Dense(self.mlp_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.drop_p, deterministic=not train)(x)
        x = nn.Dense(self.out_dim)(x)
        x = nn.Dropout(self.drop_p, deterministic=not train)(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    hidden_dim: int
    n_heads: int
    drop_p: float

    def setup(self):
        self.qkv = nn.Dense(self.hidden_dim * 3)
        self.proj = nn.Dense(self.hidden_dim)
        self.dropout = nn.Dropout(self.drop_p)

    def __call__(self, x, train=True):
        B, T, C = x.shape
        head_dim = C // self.n_heads

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, T, 3, self.n_heads, head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = jnp.matmul(q, k.transpose(0,1,3,2)) / math.sqrt(head_dim)
        attn = nn.softmax(attn, axis=-1)

        x = jnp.matmul(attn, v)
        x = x.transpose(0,2,1,3).reshape(B, T, C)

        x = self.proj(x)
        x = self.dropout(x, deterministic=not train)

        return x


class TransformerEncoder(nn.Module):
    hidden_dim: int
    n_heads: int
    mlp_dim: int
    drop_p: float

    def setup(self):
        self.norm1 = nn.LayerNorm()
        self.attn = MultiHeadSelfAttention(self.hidden_dim, self.n_heads, self.drop_p)
        self.norm2 = nn.LayerNorm()
        self.mlp = MLP(self.mlp_dim, self.drop_p, self.hidden_dim)

    def __call__(self, x, train=True):
        x = x + self.attn(self.norm1(x), train)
        x = x + self.mlp(self.norm2(x), train)
        return x


class ViT(nn.Module):
    patch_size: int
    embed_dim: int
    hidden_dim: int
    n_heads: int
    mlp_dim: int
    num_layers: int
    drop_p: float
    num_classes: int

    def setup(self):
        self.patch = Patches(self.patch_size, self.embed_dim)
        self.encoder = PatchEncoder(self.hidden_dim)

        self.blocks = [
            TransformerEncoder(self.hidden_dim, self.n_heads, self.mlp_dim, self.drop_p)
            for _ in range(self.num_layers)
        ]

        self.norm = nn.LayerNorm()
        self.head = nn.Dense(self.num_classes)

    def __call__(self, x, train=True):
        x = self.patch(x)
        x = self.encoder(x)

        for block in self.blocks:
            x = block(x, train)

        x = self.norm(x)

        cls_token = x[:,0]

        return self.head(cls_token)


def initialize_model():
    model = ViT(
        patch_size=7,
        embed_dim=64,
        hidden_dim=64,
        n_heads=4,
        mlp_dim=128,
        num_layers=4,
        drop_p=0.1,
        num_classes=10
    )

    rng = random.PRNGKey(0)
    dummy = jnp.ones((1,28,28,1))

    params = model.init(rng, dummy, train=False)['params']

    return model, params, rng


model, params, rng = initialize_model()
print("Model initialized successfully!")
