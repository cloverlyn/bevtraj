import torch
import torch.nn.functional as F
from torch import nn, einsum

from unitraj.models.bevtraj.utility import gen_sineembed_for_position
from unitraj.models.bevtraj.linear import MLP


# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

# tensor helpers

def create_grid_like(t, dim = 0):
    h, w, device = *t.shape[-2:], t.device

    grid = torch.stack(torch.meshgrid(
        torch.arange(w, device = device),
        torch.arange(h, device = device),
    indexing = 'xy'), dim = dim)

    grid.requires_grad = False
    grid = grid.type_as(t)
    return grid

def normalize_grid(grid, w, h, dim = 1, out_dim = -1, ref_from_Q=False):
    # normalizes a grid to range from -1 to 1
    grid_w, grid_h = grid.unbind(dim = dim) # grid_w: x, grid_h: y
    
    if ref_from_Q:
        grid_w = 2.0 * grid_w / max(w, 1)
        grid_h = 2.0 * grid_h / max(h, 1)
    else:
        grid_w = 2.0 * grid_w / max(w - 1, 1) - 1.0
        grid_h = 2.0 * grid_h / max(h - 1, 1) - 1.0

    return torch.stack((grid_w, grid_h), dim = out_dim) # grid_w: x, grid_h: y

class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

    
# main class

class BEVDeformCrossAttn(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        num_heads = 8,
        offset_groups = None,
        dropout = 0.,
        offset_scale = 4,
        x_bounds = [-51.2, 51.2],
        y_bounds = [-51.2, 51.2]
    ):
        super().__init__()

        inner_dim = dim_head * num_heads
        self.scale = dim_head ** -0.5
        self.num_heads = num_heads
        self.offset_groups = default(offset_groups, num_heads)

        offset_dims = inner_dim // self.offset_groups

        self.offset_scale = offset_scale
        
        assert torch.isclose(torch.abs(torch.tensor(x_bounds[0])), 
                     torch.abs(torch.tensor(x_bounds[1]))), "x range must be symmetric"
        assert torch.isclose(torch.abs(torch.tensor(y_bounds[0])), 
                     torch.abs(torch.tensor(y_bounds[1]))), "y range must be symmetric"
        self.p_w = x_bounds[1] - x_bounds[0]
        self.p_h = y_bounds[1] - y_bounds[0]
        
        self.to_con_q = nn.Linear(dim, inner_dim//2)
        self.to_con_k = nn.Conv2d(dim, inner_dim//2, 1, bias = False)
        self.to_v = nn.Conv2d(dim, inner_dim//2, 1, bias = False)
        
        self.to_pos_q = MLP(inner_dim//2, inner_dim//2, inner_dim//2, 2)
        self.to_pos_k = MLP(inner_dim//2, inner_dim//2, inner_dim//2, 2)

        self.dropout = nn.Dropout(dropout)
        
        self.to_offsets = nn.Sequential(
            nn.Linear(offset_dims//2, offset_dims),
            nn.GELU(),
            nn.Linear(offset_dims, 2),
            nn.Tanh(),
            Scale(offset_scale)
        )

        self.to_out = nn.Linear(inner_dim//2, dim)
    
    def forward(self, dec_embed, bev_feat, query_scale, ref_points, identity=None, return_vgrid=False):
        """
        B - batch
        K - num_modes
        T - future_timestamps
        H - height
        W - width
        D - dimension
        
        dec_embed: (K, B, T, D) or (K, B, D)
        bev_feat: (B, D, H, W)
        query_scale: (K, B, T, D) or (K, B, D)
        ref_points: (K, B, T, 2) or (K, B, 2)
        """
        
        if identity is None:
            identity = dec_embed
            
        # dimension handling for ITP/ITR: support (K,B,T,D) and (K,B,D)
            
        has_T = True if dec_embed.dim() == 4 else False
        num_heads, offset_groups, K, B = self.num_heads, self.offset_groups, *dec_embed.shape[:2]
        if has_T:
            T = dec_embed.shape[2]
            BS = B * K * T
            permute_pattern = (1, 0, 2, 3)
            reshape_pattern = [(B, K, T, offset_groups, -1), (B, K, T, -1)]
        else:
            BS = B * K
            permute_pattern = (1, 0, 2)
            reshape_pattern = [(B, K, offset_groups, -1), (B, K, -1)]
            
        dec_embed, query_scale, ref_points = map(lambda t: t.permute(*permute_pattern), (dec_embed, query_scale, ref_points))
        
        # get con_q & calculate offsets 

        con_q = self.to_con_q(dec_embed)
        offsets = self.to_offsets(con_q.reshape(*reshape_pattern[0]))

        # calculate grid + offsets

        vgrid = ref_points.unsqueeze(-2) + offsets
        if has_T:
            vgrid = vgrid.reshape(B, K*T, offset_groups, -1)
        vgrid_scaled = normalize_grid(vgrid, w=self.p_w, h=self.p_h, dim=-1, out_dim=-1, ref_from_Q=True)
        vgrid_scaled_fliped = vgrid_scaled.clone()
        vgrid_scaled_fliped[..., -1] = vgrid_scaled_fliped[..., -1] * -1 # Align with F.grid_sample coordinate system
        
        # get con_k / values

        gs = lambda x: F.grid_sample(
            x, vgrid_scaled_fliped,
            mode = 'bilinear', padding_mode = 'zeros', align_corners = False
        )
        con_k = gs(self.to_con_k(bev_feat))
        v = gs(self.to_v(bev_feat))
        
        # get pos_q, pos_k
        
        # ref_scaled = normalize_grid(ref_points, w=self.p_w, h=self.p_h, dim=-1, out_dim=-1, ref_from_Q=True)
        query_sine_embed = gen_sineembed_for_position(ref_points, hidden_dim=256, temperature=10000)
        pos_q = self.to_pos_q(query_sine_embed)
        pos_q = pos_q * query_scale
        
        key_sine_embed = gen_sineembed_for_position(vgrid, hidden_dim=256, temperature=10000)
        pos_k = self.to_pos_k(key_sine_embed)
        
        # split out heads
        
        con_q, pos_q = map(lambda t: t.unsqueeze(-2), (con_q, pos_q))
        con_k, v = map(lambda t: t.permute(0, 2, 3, 1), (con_k, v))
        
        con_q, con_k, pos_q, pos_k, v = [t[0].reshape(BS, t[1], num_heads, -1) \
            for t in ((con_q, 1), (con_k, offset_groups), (pos_q, 1), (pos_k, offset_groups), (v, offset_groups))]
        
        q, k = map(lambda t: torch.cat(t, dim=-1), ([con_q, pos_q], [con_k, pos_k]))
        q, k, v = map(lambda t: t.permute(0, 2, 1, 3), (q, k, v))
        
        # multi-head attention

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim - sim.amax(dim = -1, keepdim = True).detach() # numerical stability

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.permute(0, 2, 1, 3).reshape(*reshape_pattern[1]).permute(*permute_pattern)
        out = self.to_out(out)

        if return_vgrid:
            return identity + out, vgrid

        return identity + out
