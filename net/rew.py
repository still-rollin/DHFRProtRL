"""
Reference: GGS (https://github.com/kirjner/GGS)
Updated: DHFRBaseCNN modified to match newly trained medium.ckpt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Maske11dConv1d (unchanged)
# ---------------------------
class MaskedConv1d(nn.Conv1d):
    """ A masked 1-dimensional convolution layer. """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, dilation: int = 1, groups: int = 1,
                 bias: bool = True):
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, dilation=dilation, groups=groups,
                         bias=bias, padding=padding)

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


# ---------------------------
# LengthMaxPool1D (used by both models)
# ---------------------------
class LengthMaxPool1D(nn.Module):
    def __init__(self, in_dim, out_dim, linear=False, activation='relu'):
        super().__init__()
        self.linear = linear
        if self.linear:
            self.layer = nn.Linear(in_dim, out_dim)

        if activation == 'swish':
            self.act_fn = lambda x: x * torch.sigmoid(100.0 * x)
        elif activation == 'softplus':
            self.act_fn = nn.Softplus()
        elif activation == 'sigmoid':
            self.act_fn = nn.Sigmoid()
        elif activation == 'leakyrelu':
            self.act_fn = nn.LeakyReLU()
        elif activation == 'relu':
            self.act_fn = lambda x: F.relu(x)
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.linear:
            x = self.act_fn(self.layer(x))
        x = torch.max(x, dim=1)[0]
        return x


# ---------------------------
# BaseCNN (default model used for other datasets)
# ---------------------------
class BaseCNN(nn.Module):
    def __init__(
        self,
        n_tokens: int = 20,
        kernel_size: int = 5,
        input_size: int = 256,
        dropout: float = 0.0,
        make_one_hot=True,
        activation: str = 'relu',
        linear: bool = True,
        **kwargs
    ):
        super(BaseCNN, self).__init__()
        self.encoder = nn.Conv1d(n_tokens, input_size, kernel_size=kernel_size)
        self.embedding = LengthMaxPool1D(
            linear=linear,
            in_dim=input_size,
            out_dim=input_size * 2,
            activation=activation,
        )
        self.decoder = nn.Linear(input_size * 2, 1)
        self.n_tokens = n_tokens
        self.dropout = nn.Dropout(dropout)
        self.input_size = input_size
        self._make_one_hot = make_one_hot

    def get_embed(self, x):
        if self._make_one_hot:
            x = F.one_hot(x.long(), num_classes=self.n_tokens)
        x = x.permute(0, 2, 1).float()
        x = self.encoder(x).permute(0, 2, 1)
        x = self.dropout(x)
        x = self.embedding(x)
        return x

    def forward(self, x, get_embed=False):
        if self._make_one_hot:
            x = F.one_hot(x.long(), num_classes=self.n_tokens)
        x = x.permute(0, 2, 1).float()
        x = self.encoder(x).permute(0, 2, 1)
        x = self.dropout(x)
        x = self.embedding(x)
        output = self.decoder(x).squeeze(1)
        if get_embed:
            return x, output
        return output


# ---------------------------
# DHFRBaseCNN (updated to match medium.ckpt)
# ---------------------------
class DHFRBaseCNN(nn.Module):
    """
    Updated DHFR-specific CNN to match the newly trained model architecture.
    Compatible with medium.ckpt (keys: encoder.*, embedding.fc.*, decoder.*)
    """
    def __init__(self, n_tokens: int = 20, input_size: int = 128, kernel_size: int = 5,
                 dropout: float = 0.2, activation: str = 'relu', make_one_hot=False):
        super().__init__()

        self.encoder = nn.Conv1d(n_tokens, input_size, kernel_size=kernel_size, padding=kernel_size // 2)
        self.embedding = LengthMaxPool1D(
            in_dim=input_size,
            out_dim=input_size * 2,
            linear=True,
            activation=activation
        )
        self.decoder = nn.Linear(input_size * 2, 1)

        self.dropout = nn.Dropout(dropout)
        self.n_tokens = n_tokens
        self._make_one_hot = make_one_hot

    def get_embed(self, x):
        if self._make_one_hot:
            x = F.one_hot(x.long(), num_classes=self.n_tokens)
        x = x.permute(0, 2, 1).float()
        x = self.encoder(x).permute(0, 2, 1)
        x = self.dropout(x)
        x = self.embedding(x)
        return x

    def forward(self, x, get_embed=False):
        if self._make_one_hot:
            x = F.one_hot(x.long(), num_classes=self.n_tokens)
        x = x.permute(0, 2, 1).float()
        x = self.encoder(x).permute(0, 2, 1)
        x = self.dropout(x)
        x = self.embedding(x)
        output = self.decoder(x).squeeze(-1)
        if get_embed:
            return x, output
        return output


# ---------------------------
# DHFROracle (matches dhfr_Oracle repo checkpoint)
# ---------------------------
class DHFROracle(nn.Module):
    """
    DHFR Oracle model matching the working checkpoint from dhfr_Oracle repo.
    Architecture: Conv(20->128, k=3) -> Conv(128->256, k=3) -> AdaptiveAvgPool -> FC(256->64) -> FC(64->1)
    Compatible with checkpoint keys: conv.0.*, conv.2.*, fc.0.*, fc.3.*

    Note: uses AdaptiveAvgPool1d (not MaxPool) to match the original training architecture.
    """
    def __init__(self, n_tokens: int = 20, make_one_hot: bool = True):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(n_tokens, 128, kernel_size=3, padding=1),  # conv.0
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),       # conv.2
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Average pooling to match original architecture
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 64),   # fc.0
            nn.ReLU(),
            nn.Dropout(0.2),  # Match original dropout rate
            nn.Linear(64, 1)      # fc.3
        )

        self.n_tokens = n_tokens
        self._make_one_hot = make_one_hot

    def forward(self, x, get_embed=False):
        """
        Args:
            x: Input tensor of shape (batch, seq_len) containing token indices
        Returns:
            output: Fitness predictions of shape (batch,)
        """
        if self._make_one_hot:
            x = F.one_hot(x.long(), num_classes=self.n_tokens)

        # Ensure we have 3D tensor (batch, seq_len, n_tokens) for permute
        if x.dim() == 2:
            # If already 2D and make_one_hot is False, assume it's (batch, seq_len) and one-hot encode
            x = F.one_hot(x.long(), num_classes=self.n_tokens)

        x = x.permute(0, 2, 1).float()  # (batch, n_tokens, seq_len)

        # Convolutional layers with pooling
        embed = self.conv(x)  # (batch, 256) after AdaptiveAvgPool and Flatten

        # Fully connected layers
        output = self.fc(embed).squeeze(-1)  # (batch,)

        if get_embed:
            return embed, output
        return output

    def get_embed(self, x):
        """Get embeddings before final prediction layer"""
        if self._make_one_hot:
            x = F.one_hot(x.long(), num_classes=self.n_tokens)

        # Ensure we have 3D tensor for permute
        if x.dim() == 2:
            x = F.one_hot(x.long(), num_classes=self.n_tokens)

        x = x.permute(0, 2, 1).float()
        embed = self.conv(x)  # AdaptiveAvgPool and Flatten are in conv Sequential
        return embed
