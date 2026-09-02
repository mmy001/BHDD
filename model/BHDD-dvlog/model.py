from __future__ import annotations

import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class RegionAttention(nn.Module):
    """Shared multi-head attention for region-level landmark aggregation."""

    def __init__(self, input_dim: int = 32, num_heads: int = 16) -> None:
        super().__init__()
        if input_dim % num_heads != 0:
            raise ValueError("input_dim must be divisible by num_heads.")

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, region_nodes: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps, num_nodes, dim = region_nodes.shape
        reshaped_nodes = region_nodes.reshape(batch_size * timesteps, num_nodes, dim)
        projected = self.feature_proj(reshaped_nodes)
        aggregated, _ = self.multihead_attn(
            query=projected.mean(dim=1, keepdim=True),
            key=projected,
            value=projected,
        )
        return aggregated.reshape(batch_size, timesteps, dim)


def init_global_nodes(lm_data: torch.Tensor) -> torch.Tensor:
    """Append 9 global region nodes to the original 68 facial landmarks.

    Note:
        This function preserves the behavior of the original research code,
        including the on-the-fly creation of the shared attention module.
    """

    batch_size, timesteps, _, dim = lm_data.shape
    device = lm_data.device
    shared_attn = RegionAttention(input_dim=dim).to(device)
    global_nodes = torch.zeros(batch_size, timesteps, 9, dim, device=device)

    regions = [
        (0, 16),
        (17, 21),
        (22, 26),
        (27, 30),
        (31, 35),
        (36, 41),
        (42, 47),
        (48, 59),
        (60, 67),
    ]

    for i, (start, end) in enumerate(regions):
        region_data = lm_data[..., start : end + 1, :]
        aggregated = shared_attn(region_data)
        global_nodes[..., i, :] = aggregated

    return torch.cat([lm_data, global_nodes], dim=2)


class GCNLayer(nn.Module):
    """Single graph convolution layer used for landmark processing."""

    def __init__(self, in_features: int, out_features: int, adj_matrix: torch.Tensor) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.adj = adj_matrix
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(self.adj, x)
        out = self.linear(out)
        return self.relu(out)


class GCN(nn.Module):
    """Multi-layer GCN with residual connections."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        adj_matrix: torch.Tensor,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(in_features, hidden_features, adj_matrix))
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden_features, hidden_features, adj_matrix))
        self.layers.append(GCNLayer(hidden_features, out_features, adj_matrix))
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for i, layer in enumerate(self.layers):
            out = layer(x)
            if i < self.num_layers - 1:
                if residual.shape[-1] == out.shape[-1]:
                    out = out + residual
                residual = out
            x = out
        x = self.norm(x)
        x = self.dropout(x)
        return x


class Conv1dModel(nn.Module):
    """Initial temporal convolution stack for audio features."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(25, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class AstroModel(nn.Module):
    """Dilated temporal convolution module used in the original code."""

    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.conv1_1 = weight_norm(nn.Conv1d(128, 128, 3, padding=2, dilation=2), name="weight")
        self.conv1_2 = weight_norm(nn.Conv1d(128, 128, 3, padding=2, dilation=2), name="weight")
        self.conv2_1 = weight_norm(nn.Conv1d(128, 128, 3, padding=4, dilation=4), name="weight")
        self.conv2_2 = weight_norm(nn.Conv1d(128, 128, 3, padding=4, dilation=4), name="weight")
        self.conv3_1 = weight_norm(nn.Conv1d(128, 128, 3, padding=8, dilation=8), name="weight")
        self.conv3_2 = weight_norm(nn.Conv1d(128, 128, 3, padding=8, dilation=8), name="weight")
        self.conv4_1 = weight_norm(nn.Conv1d(128, 128, 3, padding=16, dilation=16), name="weight")
        self.conv4_2 = weight_norm(nn.Conv1d(128, 128, 3, padding=16, dilation=16), name="weight")
        self.conv5_1 = weight_norm(nn.Conv1d(128, 128, 3, padding=32, dilation=32), name="weight")
        self.conv5_2 = weight_norm(nn.Conv1d(128, 128, 3, padding=32, dilation=32), name="weight")
        self.conv1 = weight_norm(nn.Conv1d(128, 128, 3, padding=1), name="weight")
        self.conv2 = weight_norm(nn.Conv1d(128, 128, 3, padding=1), name="weight")
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(1, 75)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = F.relu(self.conv1_1(x))
        x1 = self.dropout(x1)
        x1 = F.relu(self.conv1_2(x1))
        x1 = self.dropout(x1)
        g1 = self.pooling(x)
        g1 = F.relu(self.conv1(g1))
        g1 = F.relu(self.conv2(g1))
        g1 = self.fc(g1)
        raw1 = 0.7 * (x1 + g1) + x

        x2 = F.relu(self.conv2_1(raw1))
        x2 = self.dropout(x2)
        x2 = F.relu(self.conv2_2(x2))
        x2 = self.dropout(x2)
        g2 = self.pooling(raw1)
        g2 = F.relu(self.conv1(g2))
        g2 = F.relu(self.conv2(g2))
        g2 = self.fc(g2)
        raw2 = 0.8 * (x2 + g2) + raw1

        x3 = F.relu(self.conv3_1(raw2))
        x3 = self.dropout(x3)
        x3 = F.relu(self.conv3_2(x3))
        x3 = self.dropout(x3)
        g3 = self.pooling(raw2)
        g3 = F.relu(self.conv1(g3))
        g3 = F.relu(self.conv2(g3))
        g3 = self.fc(g3)
        raw3 = 0.8 * (x3 + g3) + raw2

        x4 = F.relu(self.conv4_1(raw3))
        x4 = self.dropout(x4)
        x4 = F.relu(self.conv4_2(x4))
        x4 = self.dropout(x4)
        g4 = self.pooling(raw3)
        g4 = F.relu(self.conv1(g4))
        g4 = F.relu(self.conv2(g4))
        g4 = self.fc(g4)
        raw4 = 0.8 * (x4 + g4) + raw3

        x5 = F.relu(self.conv5_1(raw4))
        x5 = self.dropout(x5)
        x5 = F.relu(self.conv5_2(x5))
        x5 = self.dropout(x5)
        g5 = self.pooling(raw4)
        g5 = F.relu(self.conv1(g5))
        g5 = F.relu(self.conv2(g5))
        g5 = self.fc(g5)
        raw5 = 0.8 * (x5 + g5) + raw4

        return raw5


class GTCNModel(nn.Module):
    """Temporal CNN wrapper for audio features."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1d = Conv1dModel()
        self.astro_model = AstroModel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = self.astro_model(x)
        x = x.transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    Note:
        This implementation preserves the original research code behavior.
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)].to(x.device)


class TransformerEncoderLayerWithAttention(nn.Module):
    """Transformer encoder layer that exposes attention weights."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu
        self.attention_weights = None

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src2, attn_weights = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        self.attention_weights = attn_weights
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoderWithAttention(nn.Module):
    """Transformer encoder with explicit first-layer attention access."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayerWithAttention(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.num_layers = num_layers

    def forward_first_layer(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src = self.pos_encoder(src)
        first_layer = self.layers[0]
        output = first_layer(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        attention_map = first_layer.attention_weights
        return output, attention_map

    def forward_remaining_layers(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = src
        for i in range(1, self.num_layers):
            output = self.layers[i](
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
            )
        return output


class TokenSelection(nn.Module):
    """Select top-K tokens using first-layer attention scores."""

    def __init__(self, k: int, num_heads: int) -> None:
        super().__init__()
        self.K = k
        self.num_heads = num_heads
        self.head_weights = nn.Parameter(torch.ones(num_heads))
        self.last_topk_indices = None

    def forward(self, x: torch.Tensor, attention_weights: torch.Tensor) -> torch.Tensor:
        attention_score = attention_weights.mean(dim=1)
        _, topk_indices = torch.topk(attention_score, self.K, dim=1)
        self.last_topk_indices = topk_indices.detach().cpu().numpy()
        _, _, dim = x.size()
        return torch.gather(x, 1, topk_indices.unsqueeze(-1).expand(-1, -1, dim))


class ResidualBlock(nn.Module):
    """Residual MLP block used after token selection."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout2(out)
        out += residual
        out = self.relu(out)
        return out


def weight_init(module: nn.Module) -> None:
    """Weight initialization helper for decoder blocks."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


class CustomFeedForward(nn.Module):
    """Feed-forward network used inside the reconstruction decoder."""

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class CustomTransformerDecoderLayer(nn.Module):
    """Transformer decoder layer used for reconstruction."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ff = CustomFeedForward(d_model, dim_feedforward, dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.ff(tgt)
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)
        return tgt


class AttentionDecoder(nn.Module):
    """Reconstruction decoder used for the auxiliary loss."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.apply(weight_init)
        self.decoder_layers = nn.ModuleList(
            [
                CustomTransformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.tgt_projection_visual = nn.Linear(136, d_model)
        self.linear_visual = nn.Linear(d_model, 136)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tgt = self.tgt_projection_visual(tgt)
        tgt = tgt.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)

        output = tgt
        for decoder_layer in self.decoder_layers:
            output = decoder_layer(tgt=output, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        output = output.permute(1, 0, 2)
        output = self.linear_visual(output)
        return output


class MultiCrossAttention(nn.Module):
    """Multi-head cross-attention module for multimodal fusion."""

    def __init__(self, hidden_size: int, all_head_size: int, head_num: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.all_head_size = all_head_size
        self.num_heads = head_num
        self.head_dim = all_head_size // head_num
        if all_head_size % head_num != 0:
            raise ValueError("all_head_size must be divisible by head_num.")

        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)

        qx = self.linear_q(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        kx = self.linear_k(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        vx = self.linear_v(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        qy = self.linear_q(y).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        ky = self.linear_k(y).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        vy = self.linear_v(y).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights_xy = torch.matmul(qx, ky.transpose(-1, -2)) / sqrt(self.head_dim)
        attn_weights_xy = F.softmax(attn_weights_xy, dim=-1)
        attn_output_xy = torch.matmul(attn_weights_xy, vy)

        attn_weights_yx = torch.matmul(qy, kx.transpose(-1, -2)) / sqrt(self.head_dim)
        attn_weights_yx = F.softmax(attn_weights_yx, dim=-1)
        attn_output_yx = torch.matmul(attn_weights_yx, vx)

        attn_output_xy = attn_output_xy.transpose(1, 2).contiguous().view(batch_size, -1, self.all_head_size)
        attn_output_yx = attn_output_yx.transpose(1, 2).contiguous().view(batch_size, -1, self.all_head_size)

        attn_output_xy = attn_output_xy + x
        attn_output_yx = attn_output_yx + y
        return attn_output_xy, attn_output_yx


class FeedForward(nn.Module):
    """Position-wise feed-forward network used after fusion."""

    def __init__(
        self,
        dim_in: int,
        hidden_dim: int,
        dim_out: int | None = None,
        *,
        dropout: float = 0.1,
        activation=nn.ELU,
    ) -> None:
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            activation(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, dim_out),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Regress(nn.Module):
    """Final classification head."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 64),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 512)
        return self.fc(x)


class Net(nn.Module):
    """BHDD model for the D-Vlog dataset."""

    def __init__(self, adj_matrix: torch.Tensor, k: int) -> None:
        super().__init__()
        self.feature_proj = nn.Linear(2, 32)
        self.gcn = GCN(
            in_features=32,
            hidden_features=64,
            out_features=32,
            adj_matrix=adj_matrix,
            num_layers=5,
        )
        self.linear = nn.Linear(9 * 32, 256)

        d_model = 256
        self.transformer = TransformerEncoderWithAttention(
            d_model=d_model,
            nhead=16,
            num_layers=4,
            dim_feedforward=d_model,
            dropout=0.1,
        )
        self.token_selector = TokenSelection(k=k, num_heads=8)
        self.residual_embed = ResidualBlock(dim=d_model, dropout=0.1)
        self.token_linear = nn.Linear(d_model, 128)
        self.decoder = AttentionDecoder(d_model=128, nhead=8, num_layers=4, dim_feedforward=128, dropout=0.1)
        self.gtcn = GTCNModel()
        self.conv = nn.Conv1d(in_channels=75, out_channels=k, kernel_size=1, padding=0, stride=1)
        self.mhca = MultiCrossAttention(hidden_size=128, all_head_size=128, head_num=8)
        self.mhca1 = MultiCrossAttention(hidden_size=256, all_head_size=256, head_num=8)
        self.ffn = FeedForward(dim_in=512, hidden_dim=1024, dim_out=512)
        self.norm = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.regress = Regress()

    def forward(self, lm: torch.Tensor, audio: torch.Tensor):
        visual = lm
        batch_size = lm.size(0)

        lm = lm.view(batch_size, 600, 68, 2)
        lm = self.feature_proj(lm)
        lm = init_global_nodes(lm)
        lm_gcn = self.gcn(lm)
        lm_gcn = lm_gcn[..., -9:, :]
        lm_gcn = lm_gcn.view(batch_size, 600, -1)
        visual_features = self.linear(lm_gcn)

        transformer_input = visual_features.permute(1, 0, 2)
        output_first_layer, attention_map_first_layer = self.transformer.forward_first_layer(transformer_input)
        output_first_layer_b_t_d = output_first_layer.permute(1, 0, 2)
        selected_tokens = self.token_selector(output_first_layer_b_t_d, attention_map_first_layer)
        selected_tokens_k_b_d = selected_tokens.permute(1, 0, 2)
        final_transformer_output_k_b_d = self.transformer.forward_remaining_layers(selected_tokens_k_b_d)
        final_transformer_output = final_transformer_output_k_b_d.permute(1, 0, 2)

        embedded_tokens = self.residual_embed(final_transformer_output)
        embedded_tokens = self.token_linear(embedded_tokens)
        visual_reconstructed = self.decoder(visual, embedded_tokens)

        audio = self.gtcn(audio)
        audio = self.conv(audio)

        fused_visual, fused_audio = self.mhca(embedded_tokens, audio)
        fused_features1 = torch.cat((fused_visual, fused_audio), dim=2)

        transposed_fused_visual = fused_visual.permute(0, 2, 1).contiguous()
        transposed_fused_audio = fused_audio.permute(0, 2, 1).contiguous()
        v_mask = torch.matmul(embedded_tokens, transposed_fused_visual)
        a_mask = torch.matmul(audio, transposed_fused_audio)
        v_mask = torch.sum(v_mask, dim=-1)
        a_mask = torch.sum(a_mask, dim=-1)
        v_mask = 1 - torch.softmax(v_mask, dim=-1)
        a_mask = 1 - torch.softmax(a_mask, dim=-1)
        v = torch.mul(embedded_tokens, v_mask.unsqueeze(dim=-1))
        a = torch.mul(audio, a_mask.unsqueeze(dim=-1))
        v1, a1 = self.mhca(v, a)
        fused_features2 = torch.cat((v1, a1), dim=2)
        fused_features1, fused_features2 = self.mhca1(fused_features1, fused_features2)
        fused_features = torch.cat((fused_features1, fused_features2), dim=2)
        fused_features = self.ffn(self.norm(fused_features)) + fused_features
        output = self.norm2(fused_features)
        output = self.pooling(output.transpose(1, 2)).reshape(output.shape[0], -1)
        result = self.regress(output)

        return result, visual, visual_reconstructed
