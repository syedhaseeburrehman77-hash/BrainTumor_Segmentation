"""3D network used by every FeTS client and the centralized baseline."""

from monai.networks.nets import UNet


def build_model():
    """Return a 4-modal, 4-class 3D U-Net."""
    return UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
