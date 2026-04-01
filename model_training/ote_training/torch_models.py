from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.left_padding > 0:
            x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TCNResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.conv2 = CausalConv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity() if in_channels == hidden_size else nn.Conv1d(in_channels, hidden_size, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        return self.activation(x + residual)


class TCNClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 4,
        dropout: float = 0.2,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        dilations = [2**index for index in range(max(1, num_layers))]
        blocks = []
        in_channels = input_size
        for dilation in dilations:
            blocks.append(
                TCNResidualBlock(
                    in_channels=in_channels,
                    hidden_size=hidden_size,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = hidden_size
        self.blocks = nn.ModuleList(blocks)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        # Align the classifier head with the current-bar label by reading out the
        # latest causal timestep instead of averaging the full history.
        features = x[:, :, -1]
        features = self.dropout(features)
        return self.head(features).squeeze(-1)


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden_state, _) = self.lstm(x)
        features = self.dropout(hidden_state[-1])
        return self.head(features).squeeze(-1)


def build_torch_model(
    model_type: str,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> nn.Module:
    if model_type == "tcn":
        return TCNClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    if model_type == "lstm":
        return LSTMClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported torch model type: {model_type}")
