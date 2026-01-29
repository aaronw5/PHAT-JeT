import tensorflow as tf
from tensorflow.keras import layers, Model
import math

# Check for Flash Attention availability
try:
    from tensorflow.keras.layers import MultiHeadAttention

    # TensorFlow 2.11+ has built-in flash attention support via enable_flash_attention
    FLASH_ATTENTION_AVAILABLE = hasattr(MultiHeadAttention, "__init__")
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("Warning: Flash Attention not available in this TensorFlow version.")

# ========== Core Components ==========


class GeometricCPE(layers.Layer):
    """
    Convolutional Position Encoding that respects jet geometry.
    Uses 2D convolution on (eta, phi) grid.
    """

    def __init__(self, channels, kernel_size=3, grid_size=0.05, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.kernel_size = kernel_size
        self.grid_size = grid_size

        # Depthwise conv via groups=channels (Keras supports this)
        self.conv2d = layers.Conv2D(
            channels,
            kernel_size=kernel_size,
            padding="same",
            groups=channels,
            use_bias=True,
        )
        self.pointwise = layers.Dense(channels)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, eta, phi):
        """
        Args:
            x: Features [B, N, C]
            eta: Particle eta [B, N]
            phi: Particle phi [B, N]
        """
        B = tf.shape(x)[0]
        N = tf.shape(x)[1]
        C = self.channels
        residual = x

        # Quantize to grid per batch element (min-shifted)
        eta_min = tf.reduce_min(eta, axis=1, keepdims=True)
        phi_min = tf.reduce_min(phi, axis=1, keepdims=True)
        grid_eta = tf.cast((eta - eta_min) / self.grid_size, tf.int32)
        grid_phi = tf.cast((phi - phi_min) / self.grid_size, tf.int32)

        # Global (over batch) grid dims (safe, may over-allocate slightly)
        H = tf.reduce_max(grid_eta) + 1
        W = tf.reduce_max(grid_phi) + 1

        batch_idx = tf.range(B)[:, None]
        batch_idx = tf.tile(batch_idx, [1, N])

        indices = tf.stack([batch_idx, grid_eta, grid_phi], axis=-1)  # [B, N, 3]
        flat_indices = tf.reshape(indices, [-1, 3])
        flat_features = tf.reshape(x, [-1, C])

        grid = tf.scatter_nd(flat_indices, flat_features, [B, H, W, C])
        grid = self.conv2d(grid)

        # Gather back to particles
        out = tf.gather_nd(grid, tf.reshape(flat_indices, [B, N, 3]))

        out = self.pointwise(out)
        out = self.norm(out)
        return residual + out


# =========================
# Local Patched Attention
# =========================


class PatchedAttention(layers.Layer):
    """Local attention with patching and optional Flash Attention support."""

    def __init__(
        self,
        d_model,
        num_heads,
        patch_size,
        dropout=0.0,
        use_flash_attention=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.patch_size = patch_size
        self.use_flash_attention = use_flash_attention and FLASH_ATTENTION_AVAILABLE

        if self.use_flash_attention:
            # Use TensorFlow's built-in MultiHeadAttention with Flash Attention
            # Note: Flash Attention is automatically enabled for compatible GPUs in TF 2.11+
            self.mha = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=self.d_head, dropout=dropout, use_bias=True
            )
        else:
            # Use custom implementation
            self.wq = layers.Dense(d_model, use_bias=True)
            self.wk = layers.Dense(d_model, use_bias=True)
            self.wv = layers.Dense(d_model, use_bias=True)
            self.wo = layers.Dense(d_model, use_bias=True)
            self.dropout = layers.Dropout(dropout)

    def _split_heads(self, x, num_heads):
        b, t, d = tf.unstack(tf.shape(x)[:3])
        x = tf.reshape(x, [b, t, num_heads, d // num_heads])
        return tf.transpose(x, [0, 2, 1, 3])

    def _merge_heads(self, x):
        b, h, t, dh = tf.unstack(tf.shape(x))
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, [b, t, h * dh])

    def call(self, x, coords, training=False):
        B, T, D = tf.unstack(tf.shape(x))
        P = self.patch_size

        # Pad if necessary
        pad_len = (P - T % P) % P
        if pad_len > 0:
            x = tf.pad(x, [[0, 0], [0, pad_len], [0, 0]])
            coords = tf.pad(coords, [[0, 0], [0, pad_len], [0, 0]])

        T_padded = T + pad_len
        num_patches = T_padded // P

        # Reshape to patches
        x_patched = tf.reshape(x, [B, num_patches, P, D])
        x_patched = tf.reshape(x_patched, [B * num_patches, P, D])
        coords_patched = tf.reshape(coords, [B, num_patches, P, 2])
        coords_patched = tf.reshape(coords_patched, [B * num_patches, P, 2])

        if self.use_flash_attention:
            # Use Flash Attention via MultiHeadAttention
            # Flash Attention is automatically used on compatible hardware
            out = self.mha(
                query=x_patched,
                value=x_patched,
                key=x_patched,
                training=training,
                return_attention_scores=False,
            )
        else:
            # Use custom implementation
            # Attention within patches
            q = self._split_heads(self.wq(x_patched), self.num_heads)
            k = self._split_heads(self.wk(x_patched), self.num_heads)
            v = self._split_heads(self.wv(x_patched), self.num_heads)

            dk = tf.cast(self.d_head, x.dtype)
            scores = tf.einsum("bhtd,bhTd->bhtT", q, k) / tf.math.sqrt(dk)

            weights = tf.nn.softmax(scores, axis=-1)
            weights = self.dropout(weights, training=training)

            out = tf.einsum("bhtT,bhTd->bhtd", weights, v)
            out = self._merge_heads(out)
            # Re-introduce static last-dim so Dense can build
            out = tf.ensure_shape(out, [None, None, self.d_model])
            out = self.wo(out)

        # Reshape back
        out = tf.reshape(out, [B, num_patches, P, self.d_model])
        out = tf.reshape(out, [B, T_padded, self.d_model])

        # Remove padding
        if pad_len > 0:
            out = out[:, :-pad_len, :]

        return out


# =========================
# Patch Tokenization options
# =========================


class PatchTokenizer(layers.Layer):
    """
    Turn each patch [P, D] into a patch token [D] (or [D_out]) without TÃ—T.

    modes:
      - "mean": mean over tokens
      - "max": max over tokens
      - "flatten_dense": flatten P*D -> Dense(D)
      - "learned_pool": weights per token via Dense(1) then softmax over P
    """

    def __init__(self, d_model, patch_size, mode="mean", **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.patch_size = patch_size
        self.mode = mode

        self.flat_dense = None
        self.pool_logits = None

    def build(self, input_shape):
        if self.mode == "flatten_dense":
            # input per patch is [P, D] -> flatten -> Dense(D)
            self.flat_dense = layers.Dense(
                self.d_model, use_bias=True, name=f"{self.name}_flat_dense"
            )
        elif self.mode == "learned_pool":
            self.pool_logits = layers.Dense(
                1, use_bias=True, name=f"{self.name}_pool_logits"
            )
        elif self.mode in ("mean", "max"):
            pass
        else:
            raise ValueError(
                "PatchTokenizer mode must be one of: mean, max, flatten_dense, learned_pool"
            )

        super().build(input_shape)

    def call(self, x_patch):
        """
        x_patch: [B, NP, P, D]
        returns p: [B, NP, D]
        """
        if self.mode == "mean":
            return tf.reduce_mean(x_patch, axis=2)
        if self.mode == "max":
            return tf.reduce_max(x_patch, axis=2)
        if self.mode == "flatten_dense":
            B = tf.shape(x_patch)[0]
            NP = tf.shape(x_patch)[1]
            P = tf.shape(x_patch)[2]
            D = tf.shape(x_patch)[3]
            flat = tf.reshape(x_patch, [B, NP, P * D])
            return self.flat_dense(flat)
        # learned_pool
        logits = self.pool_logits(x_patch)  # [B, NP, P, 1]
        weights = tf.nn.softmax(logits, axis=2)  # normalize within patch
        return tf.reduce_sum(weights * x_patch, axis=2)


# =========================
# Patch-to-Patch Attention
# =========================


class PatchAttention(layers.Layer):
    """
    MHSA over patch tokens only: [B, NP, D] -> [B, NP, D]
    """

    def __init__(self, d_model, num_heads, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.dropout = layers.Dropout(dropout)

        self.wq = layers.Dense(d_model, use_bias=True)
        self.wk = layers.Dense(d_model, use_bias=True)
        self.wv = layers.Dense(d_model, use_bias=True)
        self.wo = layers.Dense(d_model, use_bias=True)

    def _split_heads(self, x):
        b = tf.shape(x)[0]
        t = tf.shape(x)[1]
        x = tf.reshape(x, [b, t, self.num_heads, self.d_head])
        return tf.transpose(x, [0, 2, 1, 3])  # [B, H, T, Dh]

    def _merge_heads(self, x):
        b = tf.shape(x)[0]
        h = tf.shape(x)[1]
        t = tf.shape(x)[2]
        x = tf.transpose(x, [0, 2, 1, 3])  # [B, T, H, Dh]
        return tf.reshape(x, [b, t, h * self.d_head])

    def call(self, p, pcoords, training=False):
        """
        p:      [B, NP, D]
        pcoords:[B, NP, 2]
        returns [B, NP, D]
        """
        q = self._split_heads(self.wq(p))
        k = self._split_heads(self.wk(p))
        v = self._split_heads(self.wv(p))

        dk = tf.cast(self.d_head, p.dtype)
        scores = tf.einsum("bhtd,bhTd->bhtT", q, k) / tf.math.sqrt(dk)  # [B,H,NP,NP]

        w = tf.nn.softmax(scores, axis=-1)
        w = self.dropout(w, training=training)

        out = tf.einsum("bhtT,bhTd->bhtd", w, v)
        out = self._merge_heads(out)
        out = tf.ensure_shape(out, [None, None, self.d_model])
        return self.wo(out)


# =========================
# Patch Message Broadcast
# =========================


class PatchMessageBroadcast(layers.Layer):
    """
    Broadcast patch-level messages back to tokens in the corresponding patch only.

    Inputs:
      x:      [B, T, D]
      coords: [B, T, 2]
    Mechanism:
      - reshape into [B, NP, P, D]
      - make patch tokens via PatchTokenizer
      - run PatchAttention on patch tokens
      - broadcast patch outputs to [B, NP, P, D] then reshape to [B, T, D]
      - add to x (optionally with projection + gate)
    """

    def __init__(
        self,
        d_model,
        num_heads,
        patch_size,
        tokenizer_mode="mean",
        dropout=0.0,
        message_proj=True,
        gated=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.tokenizer_mode = tokenizer_mode
        self.dropout = layers.Dropout(dropout)
        self.message_proj = message_proj
        self.gated = gated

        self.tokenizer = PatchTokenizer(
            d_model=d_model,
            patch_size=patch_size,
            mode=tokenizer_mode,
            name="patch_tokenizer",
        )
        self.patch_attn = PatchAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="patch_attention",
        )

        self.proj = (
            layers.Dense(d_model, use_bias=True, name="patch_msg_proj")
            if message_proj
            else None
        )
        self.gate_dense = (
            layers.Dense(d_model, use_bias=True, name="patch_msg_gate")
            if gated
            else None
        )

    def call(self, x, coords, training=False):
        """
        x: [B, T, D], coords: [B, T, 2]
        return msg_tokens: [B, T, D] (message to be added)
        """
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        D = self.d_model
        P = self.patch_size

        pad_len = (P - (T % P)) % P
        if pad_len > 0:
            x = tf.pad(x, [[0, 0], [0, pad_len], [0, 0]])
            coords = tf.pad(coords, [[0, 0], [0, pad_len], [0, 0]])

        T_pad = T + pad_len
        NP = T_pad // P

        x_patch = tf.reshape(x, [B, NP, P, D])
        c_patch = tf.reshape(coords, [B, NP, P, 2])

        # patch coords (mean coordinates per patch)
        pcoords = tf.reduce_mean(c_patch, axis=2)  # [B, NP, 2]

        # patch tokens
        p = self.tokenizer(x_patch)  # [B, NP, D]

        # patch attention output
        p_out = self.patch_attn(p, pcoords, training=training)  # [B, NP, D]
        p_out = self.dropout(p_out, training=training)

        if self.proj is not None:
            p_out = self.proj(p_out)

        # broadcast to tokens in same patch only
        p_out = tf.expand_dims(p_out, axis=2)  # [B, NP, 1, D]
        p_out = tf.tile(p_out, [1, 1, P, 1])  # [B, NP, P, D]
        msg = tf.reshape(p_out, [B, T_pad, D])  # [B, T_pad, D]

        if pad_len > 0:
            msg = msg[:, :T, :]

        if self.gated:
            # token-wise gate depends on current token features
            gate = tf.nn.sigmoid(self.gate_dense(x[:, :T, :]))
            msg = msg * gate

        return msg


# =========================
# Geometric pooling
# =========================


class GeometricPooling(layers.Layer):
    """Pools based on spatial proximity (sort by eta)."""

    def __init__(self, out_dim, stride=2, **kwargs):
        super().__init__(**kwargs)
        self.stride = stride
        self.proj = layers.Dense(out_dim)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x, coords = inputs
        B, N = tf.shape(x)[0], tf.shape(x)[1]
        channels = x.shape[-1]
        eta = coords[..., 0]

        sort_idx = tf.argsort(eta, axis=1)
        batch_idx = tf.range(B)[:, None]
        batch_idx = tf.tile(batch_idx, [1, N])
        gather_idx = tf.stack([batch_idx, sort_idx], axis=-1)

        x_sorted = tf.gather_nd(x, gather_idx)
        coords_sorted = tf.gather_nd(coords, gather_idx)

        N_out = N // self.stride
        remainder = N % self.stride
        if remainder != 0:
            pad_len = self.stride - remainder
            x_sorted = tf.pad(x_sorted, [[0, 0], [0, pad_len], [0, 0]])
            coords_sorted = tf.pad(coords_sorted, [[0, 0], [0, pad_len], [0, 0]])
            N_out = (N + pad_len) // self.stride

        x_grouped = tf.reshape(x_sorted, [B, N_out, self.stride, channels])
        coords_grouped = tf.reshape(coords_sorted, [B, N_out, self.stride, 2])

        x_pooled = tf.reduce_max(x_grouped, axis=2)
        coords_pooled = tf.reduce_mean(coords_grouped, axis=2)

        x_pooled = tf.ensure_shape(x_pooled, [None, None, channels])
        x_pooled = self.proj(x_pooled)
        x_pooled = self.norm(x_pooled)
        return [x_pooled, coords_pooled]


# =========================
# Blocks
# =========================


class PHATBlock(layers.Layer):
    """
    Local patched attention + optional patch-to-patch message passing + FFN.
    No TÃ—T attention is ever built.

    Options:
      - use_cpe: enable/disable GeometricCPE
      - ffn_activation: "relu" or "gelu"
      - use_patch_messages: enable/disable patch-to-patch messages
      - patch_tokenizer_mode: how to build patch tokens ("mean","max","flatten_dense","learned_pool")
      - message_gated: gate patch message per token
    """

    def __init__(
        self,
        d_model,
        d_ff,
        num_heads,
        patch_size,
        cpe_k=8,
        grid_size=0.05,
        dropout=0.0,
        use_cpe=True,
        ffn_activation="gelu",
        use_patch_messages=True,
        patch_tokenizer_mode="mean",
        message_proj=True,
        message_gated=False,
        use_flash_attention=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert ffn_activation in ("relu", "gelu")

        self.use_cpe = use_cpe
        if use_cpe:
            self.cpe = GeometricCPE(d_model, kernel_size=cpe_k, grid_size=grid_size)

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = PatchedAttention(
            d_model,
            num_heads,
            patch_size,
            dropout=dropout,
            use_flash_attention=use_flash_attention,
        )
        self.drop1 = layers.Dropout(dropout)

        self.use_patch_messages = use_patch_messages
        if use_patch_messages:
            self.patch_msg = PatchMessageBroadcast(
                d_model=d_model,
                num_heads=num_heads,
                patch_size=patch_size,
                tokenizer_mode=patch_tokenizer_mode,
                dropout=dropout,
                message_proj=message_proj,
                gated=message_gated,
                name="patch_message",
            )
            self.drop_msg = layers.Dropout(dropout)

        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(d_ff, activation=ffn_activation),
                layers.Dropout(dropout),
                layers.Dense(d_model),
            ]
        )
        self.drop2 = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        x, coords = inputs
        eta, phi = coords[..., 0], coords[..., 1]

        # optional CPE
        if self.use_cpe:
            x = self.cpe(x, eta, phi)

        # local patched attention
        y = self.attn(self.norm1(x), coords, training=training)
        x = x + self.drop1(y, training=training)

        # optional patch-to-patch messages (cross-patch mixing via patch tokens)
        if self.use_patch_messages:
            m = self.patch_msg(self.norm1(x), coords, training=training)
            x = x + self.drop_msg(m, training=training)

        # FFN
        y = self.ffn(self.norm2(x), training=training)
        x = x + self.drop2(y, training=training)

        return [x, coords]


# =========================
# Full Models
# =========================


def build_phat_jet_classifier(
    num_particles=150,
    output_dim=5,
    enc_dims=[64, 128, 256],
    enc_layers=[1, 1, 1],
    enc_heads=[4, 8, 8],
    enc_patch_sizes=[64, 32, 16],
    enc_strides=[2, 2],
    cpe_k=8,
    grid_size=0.05,
    use_cpe=True,
    use_pool=True,
    dropout=0.0,
    aggregation="max",
    ffn_activation="gelu",
    use_patch_messages=True,
    patch_tokenizer_mode="mean",  # "mean","max","flatten_dense","learned_pool"
    message_proj=True,
    message_gated=False,
    use_flash_attention=False,
):
    """Build hierarchical PHAT-JeT jet classifier."""

    # Input: [pt, eta, phi]
    features_input = layers.Input((num_particles, 3), name="features")

    coords = features_input[..., 1:3]  # [eta, phi]
    x = layers.Dense(enc_dims[0], activation="relu")(features_input)

    for i in range(len(enc_dims)):
        for _ in range(enc_layers[i]):
            x, coords = PHATBlock(
                d_model=enc_dims[i],
                d_ff=enc_dims[i] * 4,
                num_heads=enc_heads[i],
                patch_size=enc_patch_sizes[i],
                cpe_k=cpe_k,
                grid_size=grid_size,
                dropout=dropout,
                use_cpe=use_cpe,
                ffn_activation=ffn_activation,
                use_patch_messages=use_patch_messages,
                patch_tokenizer_mode=patch_tokenizer_mode,
                message_proj=message_proj,
                message_gated=message_gated,
                use_flash_attention=use_flash_attention,
            )([x, coords])

        if i < len(enc_dims) - 1:
            if use_pool:
                x, coords = GeometricPooling(
                    out_dim=enc_dims[i + 1], stride=enc_strides[i]
                )([x, coords])
            else:
                x = layers.Dense(enc_dims[i + 1])(x)

    x = tf.reduce_mean(x, axis=1) if aggregation == "mean" else tf.reduce_max(x, axis=1)

    x = layers.Dense(enc_dims[-1], activation="relu")(x)
    x = layers.Dropout(dropout)(x)

    activation = "sigmoid" if output_dim == 1 else "softmax"
    outputs = layers.Dense(output_dim, activation=activation)(x)

    return Model(inputs=features_input, outputs=outputs)


# =========================
# Example usage
# =========================

if __name__ == "__main__":
    # PHAT-JeT model (LOCAL attention + optional PATCH messages)
    model = build_phat_jet_classifier(
        num_particles=150,
        output_dim=5,
        enc_dims=[64, 128, 256],
        enc_layers=[2, 2, 2],
        enc_heads=[4, 8, 8],
        enc_patch_sizes=[50, 25, 25],  # choose patch sizes per stage
        enc_strides=[3, 2],  # 150 -> 50 -> 25
        cpe_k=8,
        grid_size=0.05,
        use_cpe=True,  # toggle CPE
        ffn_activation="gelu",  # "relu" or "gelu"
        use_patch_messages=True,  # toggle patch-to-patch messages
        patch_tokenizer_mode="learned_pool",  # "mean","max","flatten_dense","learned_pool"
        message_proj=True,  # Dense on patch message
        message_gated=False,  # token-wise gate on patch message
        dropout=0.1,
        aggregation="max",
    )
    model.summary()

    # Test forward pass
    batch_size = 4
    dummy_input = tf.random.normal([batch_size, 150, 3])
    out = model(dummy_input, training=False)
    print("\nOutput shape:", out.shape)
