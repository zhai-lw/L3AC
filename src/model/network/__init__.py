from l3ac import en_codec
from .discriminator import Discriminator

NetworkConfig = en_codec.ModelConfig
Network = en_codec.EnCodec
# Network = lambda config: refactor_codec(en_codec.EnCodec(config))


def refactor_codec(codec: en_codec.EnCodec):
    from x_transformers.x_transformers import AttentionLayers
    from l3ac.local_trans import LocalTrans
    from l3ac.tconv.base import TrendPool
    for name, layer in codec.named_modules():
        if isinstance(layer, LocalTrans):
            dim, depth, heads = codec.mc.feature_dim, len(layer.layers), 6
            new_attn = AttentionLayers(causal=True, dim=dim, depth=depth, heads=heads,
                                       dynamic_pos_bias=not layer.use_rotary_pos_emb,
                                       rotary_pos_emb=layer.use_rotary_pos_emb)
            codec.set_submodule(name, new_attn)
        if isinstance(layer, TrendPool):
            layer.kernel_size = 1

    return codec
