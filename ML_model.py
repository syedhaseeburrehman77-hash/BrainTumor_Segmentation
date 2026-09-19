"""3D network used by every FeTS client and the centralized baseline."""

from monai.networks.nets import UNet
import torch.nn as nn


def build_model():
    """Return a 4-modal, 4-class 3D U-Net."""
    return UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=("INSTANCE", {"affine": True}),
    )


def instance_norm_state_keys(model=None) -> set[str]:
    """State-dict keys belonging to affine 3D InstanceNorm layers."""
    model = model or build_model()
    keys: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, nn.InstanceNorm3d) and module.affine:
            keys.update({f"{module_name}.weight", f"{module_name}.bias"})
    return keys
