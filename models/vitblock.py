import tensorflow as tf

class Patchify(tf.keras.layers.Layer):
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size

    def call(self, x):
        batch_size = tf.shape(x)[0]
        patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patch_dim = patches.shape[-1]
        num_patches_h = tf.shape(patches)[1]
        num_patches_w = tf.shape(patches)[2]
        patches = tf.reshape(patches, [batch_size, num_patches_h * num_patches_w, patch_dim])
        return patches, num_patches_h, num_patches_w


class MLP(tf.keras.layers.Layer):
    def __init__(self, hidden_units, dropout_rate=0.1):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_units, activation=tf.nn.gelu)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dense2 = tf.keras.layers.Dense(hidden_units)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return x


class VitBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, patch_size=None, dropout_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_size = patch_size  # Will be determined dynamically if None
        self.dropout_rate = dropout_rate
        
        # Initialize layers that don't depend on input shape
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.mlp = MLP(embed_dim, dropout_rate)

    def build(self, input_shape):
        # Determine optimal patch size based on input spatial dimensions
        if self.patch_size is None:
            height, width = input_shape[1], input_shape[2]
            # Use a patch size that divides evenly into the spatial dimensions
            # and results in a reasonable number of patches
            if height >= 64:
                self.patch_size = 16
            elif height >= 32:
                self.patch_size = 8
            elif height >= 16:
                self.patch_size = 4
            else:
                self.patch_size = max(1, min(height, width))
        
        # Initialize patchify with determined patch size
        self.patchify = Patchify(self.patch_size)
        
        # Calculate patch dimension based on input
        channels = input_shape[-1]
        patch_dim = self.patch_size * self.patch_size * channels
        
        # Create projection layer if patch_dim doesn't match embed_dim
        if patch_dim != self.embed_dim:
            self.patch_projection = tf.keras.layers.Dense(self.embed_dim)
        
        super().build(input_shape)

    def call(self, x):
        patches, h, w = self.patchify(x)  # shape: [B, N, patch_dim]
        
        # Apply projection if it exists
        if hasattr(self, 'patch_projection'):
            patches = self.patch_projection(patches)
        
        # Transformer block
        x = self.norm1(patches)
        attn_out = self.attn(x, x)
        x = patches + self.dropout(attn_out)  # Residual connection
        x = x + self.mlp(self.norm2(x))  # MLP with residual connection

        # Reshape back to spatial format
        batch_size = tf.shape(x)[0]
        embed_dim = tf.shape(x)[-1]
        
        # Reshape patches back to spatial grid
        x = tf.reshape(x, (batch_size, h, w, embed_dim))
        
        # Resize to original input spatial dimensions
        input_height = h * self.patch_size
        input_width = w * self.patch_size
        
        x = tf.image.resize(x, [input_height, input_width], method='nearest')
        return x