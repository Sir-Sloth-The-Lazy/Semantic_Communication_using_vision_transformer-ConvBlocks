import tensorflow as tf
import tensorflow_compression as tfc
from models.channellayer import RayleighChannel, AWGNChannel, RicianChannel
from models.vitblock import VitBlock

class SemViT(tf.keras.Model):
    def __init__(self, block_types, filters, num_blocks, has_gdn=True,
                 num_symbols=512, snrdB=25, channel='AWGN'):
        super().__init__()
        if has_gdn:
            gdn_func = tfc.layers.GDN()
            igdn_func = tfc.layers.GDN(inverse=True)
        else:
            gdn_func = tf.keras.layers.Lambda(lambda x: x)
            igdn_func = tf.keras.layers.Lambda(lambda x: x)

        assert len(block_types) == len(filters) == len(num_blocks) == 6

        self.encoder = SemViT_Encoder(
            block_types[:3],
            filters[:3],
            num_blocks[:3],
            num_symbols,
            gdn_func=gdn_func
        )

        if channel == 'Rayleigh':
            self.channel = RayleighChannel(snrdB)
        elif channel == 'AWGN':
            self.channel = AWGNChannel(snrdB)
        elif channel == 'Rician':
            self.channel = RicianChannel(snrdB, k=2)
        else:
            self.channel = tf.identity

        self.decoder = SemViT_Decoder(
            block_types[3:],
            filters[3:],
            num_blocks[3:],
            gdn_func=igdn_func
        )

    def call(self, x):
        x = self.encoder(x)  # Output: (batch, H, W, C)
        
        # Reshape for channel layer: (batch, H, W, C) -> (batch, H*W*C//2, 2)
        batch_size = tf.shape(x)[0]
        h, w, c = x.shape[1], x.shape[2], x.shape[3]
        
        # Ensure channels are even for I/Q split
        if c % 2 != 0:
            raise ValueError(f"Channel dimension must be even for I/Q split, got {c}")
        
        # Reshape to (batch, spatial_features, 2) for channel
        x = tf.reshape(x, [batch_size, h * w * c // 2, 2])
        x = self.channel(x)
        
        # Reshape back to spatial format for decoder
        x = tf.reshape(x, [batch_size, h, w, c])
        x = self.decoder(x)
        return x


class SemViT_Encoder(tf.keras.layers.Layer):
    def __init__(self, block_types, filters, num_blocks,
                 num_symbols, gdn_func=None):
        super().__init__()
        self.layers = tf.keras.Sequential([
            build_blocks(0, block_types, num_blocks, filters, 512, kernel_size=9, stride=2, gdn_func=gdn_func),
            build_blocks(1, block_types, num_blocks, filters, 256, kernel_size=5, stride=2, gdn_func=gdn_func),
            build_blocks(2, block_types, num_blocks, filters, 128, kernel_size=5, stride=2, gdn_func=gdn_func),
            tf.keras.layers.Conv2D(
                filters=num_symbols // 64 // 64 * 2,
                kernel_size=1
            )
        ])

    def call(self, x):
        return self.layers(x)


class SemViT_Decoder(tf.keras.layers.Layer):
    def __init__(self, block_types, filters, num_blocks, gdn_func=None):
        super().__init__()
        self.layers = tf.keras.Sequential([
            build_blocks(0, block_types, num_blocks, filters, 64, kernel_size=5, gdn_func=gdn_func),
            tf.keras.layers.Resizing(128, 128),
            build_blocks(1, block_types, num_blocks, filters, 128, kernel_size=5, gdn_func=gdn_func),
            tf.keras.layers.Resizing(256, 256),
            build_blocks(2, block_types, num_blocks, filters, 256, kernel_size=9, gdn_func=gdn_func),
            tf.keras.layers.Resizing(512, 512),
            tf.keras.layers.Conv2D(
                filters=3,
                kernel_size=1,
                activation='sigmoid'
            )
        ])

    def call(self, x):
        return self.layers(x)


def build_blocks(layer_idx, block_types, num_blocks, filters, spatial_size, kernel_size=5, stride=1, gdn_func=None):
    if block_types[layer_idx] == 'C':
        return build_conv(
            repetition=num_blocks[layer_idx],
            filter_size=filters[layer_idx],
            kernel_size=kernel_size,
            stride=stride,
            gdn_func=gdn_func
        )
    else:
        return build_vitblocks(
            repetition=num_blocks[layer_idx],
            num_heads=filters[layer_idx] // 32,
            head_size=32,
            spatial_size=spatial_size,
            stride=stride,
            gdn_func=gdn_func
        )


def build_conv(repetition, filter_size, kernel_size=5, stride=1, gdn_func=None):
    x = tf.keras.Sequential()
    for i in range(repetition):
        s = stride if i == 0 else 1
        x.add(tfc.SignalConv2D(
            filter_size,
            kernel_size,
            corr=True,
            strides_down=s,
            padding="same_zeros",
            use_bias=True
        ))
        if gdn_func:
            x.add(gdn_func)
        x.add(tf.keras.layers.PReLU(shared_axes=[1, 2]))
    return x


def build_vitblocks(repetition, num_heads, head_size, spatial_size, stride=1, gdn_func=None):
    x = tf.keras.Sequential()
    for i in range(repetition):
        embed_dim = num_heads * head_size
        
        # Determine appropriate patch size based on spatial dimensions
        if spatial_size >= 128:
            patch_size = 16
        elif spatial_size >= 64:
            patch_size = 8
        elif spatial_size >= 32:
            patch_size = 4
        else:
            patch_size = 2
            
        x.add(VitBlock(embed_dim=embed_dim, num_heads=num_heads, patch_size=patch_size))
        if gdn_func:
            x.add(gdn_func)
    return x


class SemViT_Encoder_Only(tf.keras.Model):
    def __init__(self, block_types, filters, num_blocks, has_gdn=True, num_symbols=512):
        super().__init__()
        gdn_func = tfc.layers.GDN() if has_gdn else tf.keras.layers.Lambda(lambda x: x)

        self.encoder = SemViT_Encoder(
            block_types[:3],
            filters[:3],
            num_blocks[:3],
            num_symbols,
            gdn_func=gdn_func
        )

    def call(self, x):
        return self.encoder(x)


class SemViT_Decoder_Only(tf.keras.Model):
    def __init__(self, block_types, filters, num_blocks, has_gdn=True):
        super().__init__()
        igdn_func = tfc.layers.GDN(inverse=True) if has_gdn else tf.keras.layers.Lambda(lambda x: x)

        self.decoder = SemViT_Decoder(
            block_types[3:],
            filters[3:],
            num_blocks[3:],
            gdn_func=igdn_func
        )

    def call(self, x):
        return self.decoder(x)