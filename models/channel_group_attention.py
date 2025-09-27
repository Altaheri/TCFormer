import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelGroupAttention(nn.Module):
    """
    Implements Channel Group Attention.

    Applies attention mechanism over groups of channels rather than individual channels.

    Args:
        in_channels (int): Number of input channels (C). Must be divisible by num_groups.
        num_groups (int): Number of groups (G) to divide channels into.
    """
    def __init__(self, in_channels, num_groups, reduction=4):
        
        super().__init__()

        assert in_channels % num_groups == 0, "in_channels must be divisible by num_groups"

        self.in_channels = in_channels
        self.num_groups = num_groups
        self.group_size = in_channels // num_groups

        # x → GlobalAvgPool → att_fc → [ReLU?] → [Expand and Sigmoid] → multiply with input
        """
            1. Global Avg Pool          → [B, C, 1, 1] ← SQUEEZE
            2. Grouped Conv (att_fc)    → [B, C/r, 1, 1] grouped 1×1 conv
            3. ReLU (activation)        → [B, C/r, 1, 1]
            4. Grouped Conv (expand)    → [B, G, 1, 1] grouped 1×1 conv
            5. Sigmoid                  → [B, G, 1, 1]  [0..1] scale factors per group
            6. Multiply with input      → [B, C, H, W]
        """
        
        # Global Average Pooling to [B, C, 1, 1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Project to 1 value per group (using grouped conv!)
        # Using Conv2d with kernel size 1 acts like a Linear layer on (B, G, 1, 1) tensors
        # For (B, G, 1, 1) tensors, nn.Conv2d(C, G, kernel_size=1) is equivalent to nn.Linear(C, G)
        self.att_fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, groups=num_groups, bias=False)
        self.att_fc2 = nn.Conv2d(in_channels // reduction, num_groups, kernel_size=1, groups=num_groups, bias=False)
        self.relu = nn.ReLU()           
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W) after applying group attention.
        """
        B, C, H, W = x.shape

        if C != self.in_channels:
             raise ValueError(f"Input channel dimension ({C}) does not match "
                              f"configured in_channels ({self.in_channels})")

        # 1. Global Average Pool
        pooled = self.pool(x)   # [B, C, 1, 1]  ← SQUEEZE

        # 2. Compress to per-group attention: [B, num_groups, 1, 1]
        group_att = self.att_fc1(pooled) # [B, C/r, 1, 1] grouped 1×1 conv
        att_weights = self.att_fc2(self.relu(group_att)) # [B, G, 1, 1] grouped 1×1 conv
          
        # 3. Apply sigmoid to get attention values
        att_weights = self.sigmoid(att_weights)

        # 4. Apply attention
        # Efficient expansion using view (instead of repeat_interleave)
        if att_weights.shape[1] != C:
            # 1. Reshape input to group structure
            x_reshaped = x.view(B, self.num_groups, self.group_size, H, W)

            # 2. Reshape attention weights: [B, G, 1, 1, 1]
            att_weights = att_weights.view(B, self.num_groups, 1, 1, 1)

            # 3. Element-wise multiplication with broadcasting
            out = x_reshaped * att_weights

            # 4. Reshape back to [B, C, H, W]
            out = out.view(B, C, H, W)
        else:
            # Broadcasting handles the H and W dimensions: (B, C, H, W) * (B, C, 1, 1) -> (B, C, H, W)
            out = x * att_weights

        return out



# --- Example Usage ---
if __name__ == '__main__':
    # Parameters
    batch_size = 4
    in_channels = 48
    height, width = 32, 32
    num_groups = 3 # Desired number of groups

    # Create a dummy input tensor
    dummy_input = torch.randn(batch_size, in_channels, height, width)

    # Instantiate the Channel Group Attention module
    group_attention_layer = ChannelGroupAttention(in_channels, num_groups)

    # Apply the attention
    output = group_attention_layer(dummy_input)

    # Check output shape (should be the same as input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)

    # Verify group_size calculation
    print(f"Channels per group: {group_attention_layer.group_size}")

    # (Optional) Look at the group weights for one sample
    # Re-run parts of the forward pass to see intermediate values
    with torch.no_grad():
        pooled = group_attention_layer.pool(dummy_input)
        grouped = pooled.view(batch_size, num_groups, group_attention_layer.group_size, 1, 1)
        descriptors = grouped.mean(dim=2, keepdim=False)
        weights = group_attention_layer.att_fc(descriptors)
        weights = group_attention_layer.relu(weights)
        # weights = group_attention_layer.expand(weights)
        weights = group_attention_layer.fc2(weights)
        weights = group_attention_layer.sigmoid(weights)
        print("\nExample group weights (first sample):")
        print(weights[0].squeeze()) # Should be 3 values between 0 and 1

        expanded = weights.repeat_interleave(group_attention_layer.group_size, dim=1)
        print("\nExample expanded weights (first sample, first 20 channels):")
        print(expanded[0, :20].squeeze()) # Should show the first weight repeated 16 times, then the second

