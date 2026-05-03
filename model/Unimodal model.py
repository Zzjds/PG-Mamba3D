"""
easynet_pfdt.py
Potential-Field Delta-t Modulated Mamba Decoder for Anomaly Detection

Key contributions:
1. Scan path: reuses HSCANS (serpentine/Hilbert/ZOrder fixed paths)
   - Fixed paths preserve full spatial locality
2. Potential-field Dt modulation: PotentialFieldScanner generates a
   potential map that directly modulates the SSM Dt parameter
   - High-potential regions (defects) -> large Dt -> absorb more features
   - Low-potential regions (background) -> small Dt -> fast pass-through
   - Fully differentiable, compatible with Mamba math
3. Decoder resolution strategy: 32/64/128 use CNN+Mamba, 256 pure CNN
"""

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    print("Warning: mamba_ssm not found.")
    selective_scan_fn = None

try:
    from hilbert import decode as hilbert_decode
    HAS_HILBERT = True
except ImportError:
    HAS_HILBERT = False
    print("Warning: hilbert not found, hilbert scan will fallback to scan.")

try:
    from pyzorder import ZOrderIndexer
    HAS_ZORDER = True
except ImportError:
    HAS_ZORDER = False
    print("Warning: pyzorder not found, zorder scan will fallback to scan.")


def init_weight(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


class HSCANS(nn.Module):
    """
    Fixed-path spatial scan module.
    Supports: scan (serpentine) / hilbert / zorder / zigzag / sweep
    encode: reorder tokens along scan path
    decode: restore original spatial positions
    """
    def __init__(self, size=16, dim=2, scan_type='scan'):
        super().__init__()
        size = int(size)
        max_num = size ** dim
        indexes = np.arange(max_num)

        if scan_type == 'sweep':
            locs_flat = indexes

        elif scan_type == 'scan':
            indexes = indexes.reshape(size, size)
            for i in np.arange(1, size, step=2):
                indexes[i, :] = indexes[i, :][::-1]
            locs_flat = indexes.reshape(-1)

        elif scan_type == 'zorder':
            if HAS_ZORDER:
                zi = ZOrderIndexer((0, size - 1), (0, size - 1))
                locs_flat = []
                for z in indexes:
                    r, c = zi.rc(int(z))
                    locs_flat.append(c * size + r)
                locs_flat = np.array(locs_flat)
            else:
                indexes = indexes.reshape(size, size)
                for i in np.arange(1, size, step=2):
                    indexes[i, :] = indexes[i, :][::-1]
                locs_flat = indexes.reshape(-1)

        elif scan_type == 'zigzag':
            indexes = indexes.reshape(size, size)
            locs_flat = []
            for i in range(2 * size - 1):
                if i % 2 == 0:
                    start_col = max(0, i - size + 1)
                    end_col   = min(i, size - 1)
                    for j in range(start_col, end_col + 1):
                        locs_flat.append(indexes[i - j, j])
                else:
                    start_row = max(0, i - size + 1)
                    end_row   = min(i, size - 1)
                    for j in range(start_row, end_row + 1):
                        locs_flat.append(indexes[j, i - j])
            locs_flat = np.array(locs_flat)

        elif scan_type == 'hilbert':
            if HAS_HILBERT:
                bit = int(math.log2(size))
                locs = hilbert_decode(indexes, dim, bit)
                locs_flat = self._flat_locs_hilbert(locs, dim, bit)
            else:
                indexes = indexes.reshape(size, size)
                for i in np.arange(1, size, step=2):
                    indexes[i, :] = indexes[i, :][::-1]
                locs_flat = indexes.reshape(-1)
        else:
            raise ValueError(f"Unknown scan_type: {scan_type}")

        locs_flat     = np.array(locs_flat, dtype=np.int64)
        locs_flat_inv = np.argsort(locs_flat).astype(np.int64)

        index_flat     = torch.LongTensor(locs_flat).unsqueeze(0).unsqueeze(1)
        index_flat_inv = torch.LongTensor(locs_flat_inv).unsqueeze(0).unsqueeze(1)
        self.index_flat     = nn.Parameter(index_flat,     requires_grad=False)
        self.index_flat_inv = nn.Parameter(index_flat_inv, requires_grad=False)

    def _flat_locs_hilbert(self, locs, num_dim, num_bit):
        ret = []
        l = 2 ** num_bit
        for loc in locs:
            loc_flat = 0
            for j in range(num_dim):
                loc_flat += loc[j] * (l ** j)
            ret.append(loc_flat)
        return np.array(ret, dtype=np.uint64)

    def encode(self, img):
        return torch.zeros_like(img).scatter_(
            2, self.index_flat_inv.expand(img.shape), img)

    def decode(self, img):
        return torch.zeros_like(img).scatter_(
            2, self.index_flat.expand(img.shape), img)


class PotentialFieldScanner(nn.Module):
    """
    Generates a [B, 1, H, W] potential map in [0, 1] for Dt modulation.
    High potential = important region (defect); low potential = background.
    """
    def __init__(self, d_inner, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.potential_net = nn.Sequential(
            nn.Conv2d(d_inner, d_inner, kernel_size, 1, pad, groups=d_inner),
            nn.BatchNorm2d(d_inner), nn.GELU(),
            nn.Conv2d(d_inner, d_inner // 4, 1),
            nn.BatchNorm2d(d_inner // 4), nn.GELU(),
            nn.Conv2d(d_inner // 4, 1, 1),
            nn.Sigmoid()
        )
        self.dt_modulation_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        potential = self.potential_net(x)
        scale = torch.tanh(self.dt_modulation_scale)
        return potential, scale


class PotentialDtSS2D(nn.Module):
    """
    SS2D with potential-field Dt modulation.
    Scan paths are determined by HSCANS; spatial locality is preserved.
    """
    def __init__(self, d_model, d_state=16, expand=2,
                 num_direction=4, size=8, scan_type='scan',
                 potential_kernel=3, **kwargs):
        super().__init__()
        self.d_model   = d_model
        self.d_state   = d_state
        self.d_inner   = int(expand * d_model)
        self.K         = num_direction
        self.dt_rank   = math.ceil(d_model / 16)

        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv2d   = nn.Conv2d(self.d_inner, self.d_inner, 3, 1, 1,
                                   groups=self.d_inner)
        self.act = nn.SiLU()

        x_proj_list = [
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2,
                      bias=False).weight
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack(x_proj_list, dim=0))

        dt_projs = [self._dt_init(self.dt_rank, self.d_inner)
                    for _ in range(self.K)]
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in dt_projs], dim=0))

        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32),
                   'n -> d n', d=self.d_inner).contiguous()
        A_log = torch.log(A)
        A_log = repeat(A_log, 'd n -> r d n', r=self.K)
        self.A_logs = nn.Parameter(A_log.flatten(0, 1))
        self.A_logs._no_weight_decay = True

        D = repeat(torch.ones(self.d_inner), 'n -> r n', r=self.K)
        self.Ds = nn.Parameter(D.flatten(0, 1))
        self.Ds._no_weight_decay = True

        self.scans = HSCANS(size=size, scan_type=scan_type)

        self.potential_scanner = PotentialFieldScanner(
            d_inner=self.d_inner, kernel_size=potential_kernel)
        self.potential_to_dt = nn.Conv2d(1, self.d_inner, 1, bias=False)
        nn.init.zeros_(self.potential_to_dt.weight)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    @staticmethod
    def _dt_init(dt_rank, d_inner):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank ** -0.5
        nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001))
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    def forward_core(self, x, potential_bias):
        B, C, H, W = x.shape
        L = H * W
        K = self.K

        xs = []
        if K >= 2:
            xs.append(self.scans.encode(x.view(B, -1, L)))
        if K >= 4:
            xs.append(self.scans.encode(
                x.transpose(2, 3).contiguous().view(B, -1, L)))
        if K >= 8:
            x_rot = torch.rot90(x, k=1, dims=(2, 3)).contiguous()
            xs.append(self.scans.encode(x_rot.view(B, -1, L)))
            xs.append(self.scans.encode(
                x_rot.transpose(2, 3).contiguous().view(B, -1, L)))

        xs = torch.stack(xs, dim=1).view(B, K // 2, -1, L)
        xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1)

        x_dbl = torch.einsum('b k d l, k c d -> b k c l',
                              xs.view(B, K, -1, L), self.x_proj_weight)
        dts_raw, Bs, Cs = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum('b k r l, k d r -> b k d l',
                            dts_raw, self.dt_projs_weight)

        pb_flat = potential_bias.view(B, -1, L)
        pb_encoded_h = self.scans.encode(pb_flat)
        pb_list = [pb_encoded_h, torch.flip(pb_encoded_h, dims=[-1])]
        if K >= 4:
            pb_encoded_v = self.scans.encode(
                potential_bias.transpose(2, 3).contiguous().view(B, -1, L))
            pb_list += [pb_encoded_v, torch.flip(pb_encoded_v, dims=[-1])]
        if K >= 8:
            pb_rot = torch.rot90(potential_bias, k=1, dims=(2, 3)).contiguous()
            pb_encoded_r  = self.scans.encode(pb_rot.view(B, -1, L))
            pb_encoded_rt = self.scans.encode(
                pb_rot.transpose(2, 3).contiguous().view(B, -1, L))
            pb_list += [pb_encoded_r,  torch.flip(pb_encoded_r,  dims=[-1]),
                        pb_encoded_rt, torch.flip(pb_encoded_rt, dims=[-1])]

        dts = dts + torch.stack(pb_list, dim=1)

        out_y = selective_scan_fn(
            xs.float().view(B, -1, L),
            dts.contiguous().float().view(B, -1, L),
            -torch.exp(self.A_logs.float()).view(-1, self.d_state),
            Bs.float().view(B, K, -1, L),
            Cs.float().view(B, K, -1, L),
            self.Ds.float().view(-1),
            z=None,
            delta_bias=self.dt_projs_bias.float().view(-1),
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        inv_y = torch.flip(out_y[:, K // 2:K], dims=[-1]).view(B, K // 2, -1, L)
        ys = [self.scans.decode(out_y[:, 0]), self.scans.decode(inv_y[:, 0])]
        if K >= 4:
            ys.append(self.scans.decode(out_y[:, 1]).view(B, -1, W, H)
                      .transpose(2, 3).contiguous().view(B, -1, L))
            ys.append(self.scans.decode(inv_y[:, 1]).view(B, -1, W, H)
                      .transpose(2, 3).contiguous().view(B, -1, L))
        if K >= 8:
            ys.append(torch.rot90(self.scans.decode(out_y[:, 2]).view(B, -1, W, H),
                                  k=3, dims=(2, 3)).contiguous().view(B, -1, L))
            ys.append(torch.rot90(self.scans.decode(inv_y[:, 2]).view(B, -1, W, H),
                                  k=3, dims=(2, 3)).contiguous().view(B, -1, L))
            ys.append(torch.rot90(
                self.scans.decode(out_y[:, 3]).view(B, -1, W, H)
                .transpose(2, 3), k=3, dims=(2, 3)).contiguous().view(B, -1, L))
            ys.append(torch.rot90(
                self.scans.decode(inv_y[:, 3]).view(B, -1, W, H)
                .transpose(2, 3), k=3, dims=(2, 3)).contiguous().view(B, -1, L))
        return sum(ys)

    def forward(self, x):
        B, H, W, C = x.shape
        xz      = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_conv  = self.act(self.conv2d(x_in.permute(0, 3, 1, 2)))

        potential, mod_scale = self.potential_scanner(x_conv)
        dt_bias_map = self.potential_to_dt(potential)
        dt_bias_map = mod_scale * dt_bias_map

        y = self.forward_core(x_conv, dt_bias_map)
        y = y.transpose(1, 2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y) * F.silu(z)
        return self.out_proj(y)


class PotentialDtMixer(nn.Module):
    """
    Drop-in replacement block: [B, C, H, W] -> [B, C, H, W]
    CNN features -> potential-Dt Mamba -> residual add
    mamba_gate controls the contribution strength of Mamba output.
    """
    def __init__(self, dim, size=8, scan_type='scan',
                 num_direction=4, mamba_gate_init=0.5):
        super().__init__()
        self.mamba = PotentialDtSS2D(
            d_model=dim, d_state=16, expand=2,
            num_direction=num_direction,
            size=size, scan_type=scan_type)
        self.mamba_gate = nn.Parameter(torch.tensor(float(mamba_gate_init)))

    def forward(self, x):
        x_bhwc   = x.permute(0, 2, 3, 1).contiguous()
        out_bhwc = self.mamba(x_bhwc)
        out      = out_bhwc.permute(0, 3, 1, 2).contiguous()
        gate = torch.sigmoid(self.mamba_gate)
        return x + gate * out


class EncoderReconstructive(nn.Module):
    def __init__(self, in_channels, base_width):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, 3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, 3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))
        self.mp1 = nn.MaxPool2d(2)
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, 3, padding=1),
            nn.BatchNorm2d(base_width * 2), nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, 3, padding=1),
            nn.BatchNorm2d(base_width * 2), nn.ReLU(inplace=True))
        self.mp2 = nn.MaxPool2d(2)
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, 3, padding=1),
            nn.BatchNorm2d(base_width * 4), nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, 3, padding=1),
            nn.BatchNorm2d(base_width * 4), nn.ReLU(inplace=True))
        self.mp3 = nn.MaxPool2d(2)
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 8, 3, padding=1),
            nn.BatchNorm2d(base_width * 8), nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, 3, padding=1),
            nn.BatchNorm2d(base_width * 8), nn.ReLU(inplace=True))
        self.mp4 = nn.MaxPool2d(2)
        self.block5 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, 3, padding=1),
            nn.BatchNorm2d(base_width * 8), nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, 3, padding=1),
            nn.BatchNorm2d(base_width * 8), nn.ReLU(inplace=True))

    def forward(self, x):
        b1  = self.block1(x)
        mp1 = self.mp1(b1)
        b2  = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3  = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4  = self.block4(mp3)
        mp4 = self.mp4(b4)
        b5  = self.block5(mp4)
        return b1, b2, b5


class PotentialDtDecoder(nn.Module):
    """
    Decoder with resolution-adaptive strategy:
      Stage 1  32x32   CNN + Potential-Dt Mamba  (gate_init=0.55)
      Stage 2  64x64   CNN + Potential-Dt Mamba  (gate_init=0.40)
      Stage 3  128x128 CNN + Potential-Dt Mamba  (gate_init=0.25)
      Stage 4  256x256 Pure CNN
    """
    def __init__(self, base_width, out_channels=1, scan_type='scan', num_direction=4):
        super().__init__()

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 8, base_width * 8, 3, padding=1),
            nn.BatchNorm2d(base_width * 8), nn.ReLU(inplace=True))
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, 3, padding=1),
            nn.BatchNorm2d(base_width * 8), nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 4, 3, padding=1),
            nn.BatchNorm2d(base_width * 4), nn.ReLU(inplace=True))
        self.mamba1 = PotentialDtMixer(
            dim=base_width * 4, size=32, scan_type=scan_type,
            num_direction=num_direction, mamba_gate_init=0.55)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 4, base_width * 4, 3, padding=1),
            nn.BatchNorm2d(base_width * 4), nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 4, 3, padding=1),
            nn.BatchNorm2d(base_width * 4), nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 2, 3, padding=1),
            nn.BatchNorm2d(base_width * 2), nn.ReLU(inplace=True))
        self.mamba2 = PotentialDtMixer(
            dim=base_width * 2, size=64, scan_type=scan_type,
            num_direction=num_direction, mamba_gate_init=0.40)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 2, base_width * 2, 3, padding=1),
            nn.BatchNorm2d(base_width * 2), nn.ReLU(inplace=True))
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 2, 3, padding=1),
            nn.BatchNorm2d(base_width * 2), nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 1, 3, padding=1),
            nn.BatchNorm2d(base_width * 1), nn.ReLU(inplace=True))
        self.mamba3 = PotentialDtMixer(
            dim=base_width * 1, size=128, scan_type=scan_type,
            num_direction=num_direction, mamba_gate_init=0.25)

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width, base_width, 3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width, base_width, 3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, 3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))

        self.fin_out = nn.Conv2d(base_width, out_channels, 3, padding=1)

    def forward(self, b5):
        up1 = self.up1(b5)
        db1 = self.mamba1(self.db1(up1))

        up2 = self.up2(db1)
        db2 = self.mamba2(self.db2(up2))

        up3 = self.up3(db2)
        db3 = self.mamba3(self.db3(up3))

        up4 = self.up4(db3)
        db4 = self.db4(up4)

        out = self.fin_out(db4)
        return out, db3, db4


class ReconstructiveSubNetwork(nn.Module):
    """
    Full single-modal reconstructive network.
    scan_type: 'scan' / 'hilbert' / 'zorder' / 'zigzag' / 'sweep'
    num_direction: 4 (standard) or 8 (stronger but slower)
    """
    def __init__(self, in_channels=3, out_channels=3, base_width=128,
                 img_size=256, train_model=True,
                 scan_type='scan', num_direction=4):
        super().__init__()
        self.encoder = EncoderReconstructive(in_channels, base_width)
        self.decoder = PotentialDtDecoder(
            base_width, out_channels=out_channels,
            scan_type=scan_type, num_direction=num_direction)

        self.hidden_size  = 512
        self.hidden_size1 = 256
        self.hidden_size2 = 2
        self.img_size     = img_size

        self.mlp1 = nn.Sequential(
            nn.Conv2d(base_width * 2, self.hidden_size, 1),
            nn.BatchNorm2d(self.hidden_size), nn.LeakyReLU(0.2))
        self.mlp2 = nn.Sequential(
            nn.Conv2d(base_width * 3, self.hidden_size, 1),
            nn.BatchNorm2d(self.hidden_size), nn.LeakyReLU(0.2))
        self.mlp3 = nn.Sequential(
            nn.Conv2d(self.hidden_size * 2, self.hidden_size, 1),
            nn.BatchNorm2d(self.hidden_size), nn.LeakyReLU(0.2),
            nn.Conv2d(self.hidden_size, self.hidden_size1, 1),
            nn.BatchNorm2d(self.hidden_size1), nn.LeakyReLU(0.2),
            nn.Conv2d(self.hidden_size1, self.hidden_size2, 1),
            nn.BatchNorm2d(self.hidden_size2), nn.LeakyReLU(0.2))

        if train_model:
            self.apply(init_weight)
            nn.init.zeros_(self.decoder.mamba1.mamba.out_proj.weight)
            nn.init.zeros_(self.decoder.mamba2.mamba.out_proj.weight)
            nn.init.zeros_(self.decoder.mamba3.mamba.out_proj.weight)
            nn.init.zeros_(self.decoder.mamba1.mamba.potential_to_dt.weight)
            nn.init.zeros_(self.decoder.mamba2.mamba.potential_to_dt.weight)
            nn.init.zeros_(self.decoder.mamba3.mamba.potential_to_dt.weight)

    def forward(self, x):
        b1, b2, b5       = self.encoder(x)
        output, db3, db4 = self.decoder(b5)

        merge1 = self.mlp1(torch.cat((b1, db4), dim=1))
        merge2 = self.mlp2(torch.cat((b2, db3), dim=1))
        merge2 = F.interpolate(
            merge2, size=(self.img_size, self.img_size),
            mode='bilinear', align_corners=False)
        mask = self.mlp3(torch.cat((merge1, merge2), dim=1))
        return output, mask, torch.cat((merge1, merge2), dim=1)


EasyNet = ReconstructiveSubNetwork