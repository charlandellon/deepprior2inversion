"""Implementation the deep prior network.

This implementation is based on the original implementation of the deep
priors in https://github.com/DmitryUlyanov/deep-image-prior.

Typical usage example:

g = DeepPrior(shape)
z = torch.randn(g.get_latent_shape())
g(z)
"""

from math import floor, ceil
import numpy as np
import torch
from typing import List, Optional, Sequence, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F


def add_module(self, module: torch.nn.Module):
    """Define the `add_module` method to use module number as module name.

    Args:
        module: A torch.nn.Module object.
    """
    self.add_module(str(len(self) + 1), module)


# Redefine the `add` method to use modules number as module name.
torch.nn.Module.add = add_module


class Concat(torch.nn.Module):
    """A concatenation module (layer).

    This module defines a concatenation "layer" that can be used to concatenate
    multiple activations at a given dimension.

    Attributes:
        dim: An integer indicating the dimension to concatenate the activations
            at.
    """

    def __init__(self, dim: int, *args: torch.nn.Module):
        """Initializes a Concat module.

        Args:
            dim: An integer indicating the dimension to concatenate the
                activations at.
            args: A list of torch.nn.Module objects, containing the modules to
                concatenate their activations.
        """
        super().__init__()
        self.dim = dim
        for module in args:
            self.add(module)

    def __len__(self) -> int:
        """Define the `__len__` method that returns the number of modules.

        Returns:
            An integer, the number of modules.
        """
        return len(self._modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenates the activations.

        This function first evaluates the activations of all the modules
        provided as input during initialization. Next, it concatenates the
        activations at the dimension specified during initialization.

        Args:
            x: A torch.Tensor object, containing the input to the `Concat`
                layer.

        Returns:
            A torch.Tensor containing the concatenated activations.
        """
        inputs = []
        for module in self._modules.values():
            inputs.append(module(x))

        inputs_shapes2 = [x_.shape[2] for x_ in inputs]
        inputs_shapes3 = [x_.shape[3] for x_ in inputs]

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and \
                np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2:diff2 + target_shape2,
                                   diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)


class View(torch.nn.Module):
    """A module to create a view of an existing torch.Tensor (avoid copying).

    Attributes:
        shape: A tuple containing the desired shape of the view.
    """

    def __init__(self, *shape: int):
        """Initializes a Concat module.

        Args:
            shape: A tuple containing the desired shape of the view.
        """
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Creates a view of an input.

        Args:
            x: A torch.Tensor object.

        Returns:
            A torch.Tensor containing the view of the input with given
                dimensions.
        """
        return x.view(*self.shape)


def pad_and_conv(c_in: int,
                 c_out: int,
                 kernel_size: int,
                 stride: Optional[int] = 1,
                 bias: Optional[bool] = True) -> torch.nn.Module:
    """A utility function to define a 2D convolutional layer with padding.

    Args:
        c_in: An integer, the number of input channels.
        c_out: An integer, the number of output channels.
        kernel_size: An integer, the size of the convolutional kernel.
        stride: An optional integer, the stride of the convolutional kernel.
        bias: An optional boolean, indicating whether to use bias.

    Returns:
        A torch.nn.Conv2d object.
    """
    # Amount to pad.
    to_pad = int((kernel_size - 1) / 2)
    # Reflect padding.
    padding_layer = torch.nn.ReflectionPad2d(to_pad)
    # Convolutional layer.
    conv_layer = torch.nn.Conv2d(c_in,
                                 c_out,
                                 kernel_size,
                                 stride,
                                 bias=bias,
                                 padding=0)
    return torch.nn.Sequential(*[padding_layer, conv_layer])


class CustomActivation(torch.nn.Module):
    def __init__(self, threshold):
        super(CustomActivation, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        return torch.where(x < self.threshold[0], torch.full_like(x, self.threshold[0]),
                           torch.where(x > self.threshold[1], torch.full_like(x, self.threshold[1]), x))

class NormalizeActivation(torch.nn.Module):
    def __init__(self, threshold):
        super(NormalizeActivation, self).__init__()
        self.vpmin, self.vpmax = threshold
        # self.activate = torch.nn.Softplus(beta=1, threshold=20)
        # self.activate = torch.nn.Tanh()
        self.activate = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.activate(x)
        return x * (self.vpmax - self.vpmin) + self.vpmin
        # return (x * (self.vpmax - self.vpmin) + (self.vpmin + self.vpmax)) / 2.
        # return (x - x.min()) * (self.threshold[1] - self.threshold[0]) / (x.max() - x.min()) + self.threshold[0]


class Crop(torch.nn.Module):
    """A module to crop in the input tensor.

    This module computes the padding dimensions and cropping amount needed to
    ensure same input and output deep prior network sizes.

    Attributes:
        input_shape: A 4-component list or tuple containing the input tensor
            shape.
        crop_shape: A list containing per dimension cropping shape.
        latent_shape: A tuple containing the latent variable shape.
    """

    def __init__(self, input_shape: Sequence, n_scales: int):
        """Initializes a Crop "layer".

        Args:
            args: An argparse.Namespace, containing command line arguments.
        """
        super().__init__()

        self.input_shape = input_shape
        self.crop_shape, self.latent_shape = self.find_cropping_shape(n_scales)

    def find_cropping_shape(self, n_scales: int) -> Tuple[List, List]:
        """Finds cropping shape.

        Cropping shape such that the deep prior output shape is the same size as
        `input_shape`.

        Args:
            n_scales: An integer, the number of scales.

        Returns:
            crop_shape: A list containing the cropping shape.
            latent_shape: A list containing the latent variable shape.

        """
        # Downsampling factor.
        downsample_factor = 2**n_scales

        # Total cropping shape.
        total_crop_shape = [
            min(
                self.input_shape[i + 2] % downsample_factor,
                downsample_factor -
                self.input_shape[i + 2] % downsample_factor) for i in range(2)
        ]

        # Per dimension cropping shape.
        crop_shape = np.zeros([2, 2], dtype=int)
        crop_shape[0, 0] = int(floor(total_crop_shape[0] / 2.))
        crop_shape[0, 1] = int(ceil(total_crop_shape[0] / 2.))
        crop_shape[1, 0] = int(floor(total_crop_shape[1] / 2.))
        crop_shape[1, 1] = int(ceil(total_crop_shape[1] / 2.))

        # Latent variable shape.
        # latent_shape = [
        #     self.input_shape[i + 2] + total_crop_shape[i] for i in range(2)
        # ]

        latent_shape = [
            self.input_shape[i + 2] + total_crop_shape[i] for i in range(2)
        ]


        return crop_shape, latent_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Crops in the input tensor according to the cropping shape.

        Args:
            x: A torch.Tensor object.

        Returns:
            A torch.Tensor containing the cropped input tensor.
        """
        if (self.input_shape[2] == self.latent_shape[0]
                and self.input_shape[3] == self.latent_shape[1]):
            y = x
        elif self.input_shape[2] == self.latent_shape[0]:
            y = x[:, :, :, self.crop_shape[1, 0]:-self.crop_shape[1, 1]]
        elif self.input_shape[3] == self.latent_shape[1]:
            y = x[:, :, self.crop_shape[0, 0]:-self.crop_shape[0, 1], :]
        else:
            y = x[:, :, self.crop_shape[0, 0]:-self.crop_shape[0, 1],
                  self.crop_shape[1, 0]:-self.crop_shape[1, 1]]
        return y


class Scale(torch.nn.Module):
    """A scaling module (layer).

    This module simply scales the its input tensor with a fixed value (float).

    Attributes:
        scaling_factor: A float, the scaling factor.
    """

    def __init__(self, scaling_factor: float):
        """Initializes a Scale module.

        Args:
            scaling_factor: A float, the scaling factor.
        """
        super().__init__()
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scales the input tensor by `scaling_factor`.

        Args:
            x: A torch.Tensor object.

        Returns:
            A torch.Tensor scaled by `scaling_factor`
        """
        return x * self.scaling_factor


class DeepPrior(torch.nn.Module):
    """Deep prior network implementation.

    The CNN architecture proposed by Lempitsky et al. (2018) is a variation of
    the widely used U-net architecture, as described by Ronneberger et al.
    (2015). The U-net architecture is composed of an encoder and a decoder
    module, where information, i.e., intermediate values, from the encoder
    module is also passed to the decoder through skip connections.

    The following are the major differences between the deep prior architecture
    (Lempitsky et al., 2018) and the U-net architecture. Contrary to U-net, the
    convolutional layers in the decoding module of the deep prior architecture
    do not increase the dimensionality of the intermediate values. Rather,
    dimensionality-preserving convolutional layers, that is, stride-one
    convolutions, are augmented with user-defined interpolation schemes to
    achieve upsampling. This enables the degree of smoothness in the image space
    to be controlled by the interpolation kernel. Another difference worth
    noting is the way intermediate values from the encoding phase are
    incorporated into the decoding module through skip connections. In the deep
    prior architecture, the intermediate values in the encoding phase are passed
    through an additional convolutional layer before being fed into the decoder
    module.

    This implementation is based on the original implementation of the deep
    priors in https://github.com/DmitryUlyanov/deep-image-prior. For
    schematic representation of the architecture with the following default
    parameters see Figure 15 in https://arxiv.org/abs/2110.04825.

    Attributes:
        ch_in: An integer, the number of input channels.
        model: A torch.nn.Sequential object, containing the deep prior network.

    """

    def __init__(self,
                 output_shape: Sequence,
                 Vp, Mask=None,
                 ch_in: Optional[int] = 3,
                 ch_out: Optional[int] = 1,
                 ch_encoding: Optional[Union[int, Tuple, List]] = (16, 32, 64),
                 ch_decoding: Optional[Union[int, Tuple, List]] = (16, 32, 64),
                 ch_skip: Optional[Union[int, Tuple, List]] = (0, 0, 64),
                 kernel_size_encoding: Optional[Union[int, Tuple, List]] = 5,
                 kernel_size_decoding: Optional[Union[int, Tuple, List]] = 5,
                 kernel_size_skip: Optional[int] = 1,
                 bias: Optional[bool] = True,
                 upsample_mode: Optional[str] = 'nearest',
                 upsample_scale: Optional[int] = 1,
                 coarse_output_scale: Optional[int] = 1,
                 activation_fn: Optional[torch.nn.Module] = torch.nn.LeakyReLU(0.1, inplace=True),
                 scaling_factor: Optional[Union[Tuple, List]] = (-1, 1)):

        """Initializes a DeepPrior object.

        Args:
        input_shape: A 4-component list or tuple containing the input tensor
            shape.
        ch_in: An optional integer, the number of input (latent variable)
            channels.
        ch_out: An optional integer, the number of output (image) channels.
        ch_encoding: An optional integer, list, or tuple of the number channels
            in the encoder module.
        ch_decoding: An optional integer, list, or tuple of the number channels
            in the decoding module.
        ch_skip: An optional integer, list, or tuple of the number channels
            in the skip connections.
        kernel_size_encoding: An optional integer, list, or tuple of the
            kernel sizes in the encoder module.
        kernel_size_decoding: An optional integer, list, or tuple of the
            kernel sizes in the decoding module.
        kernel_size_skip: An optional integer for the kernel size in the
            skip connections.
        bias: An optional boolean, whether to use bias in the convolutional
            layers.
        upsample_model: An optional string, the upsampling model to use in the
            decoder.
        activation_fn: An optional torch.nn.Module, the activation function.
        scaling_factor: An optional float, the scaling factor to multiply the
            output of the network with.
        """
        super().__init__()

        if len(ch_encoding) != len(ch_decoding):
            ValueError('`ch_encoding` must be of same length as '
                       '`ch_decoding`')
        if len(ch_encoding) != len(ch_skip):
            ValueError('`ch_encoding` must be of same length as '
                       '`ch_skip`')
        if len(ch_decoding) != len(ch_skip):
            ValueError('`ch_decoding` must be of same length as '
                       '`ch_skip`')

        self.ch_in = ch_in
        n_scales = len(ch_encoding)
        self.output_shape = tuple(output_shape[-2:])
        self.coarse_output_scale = max(1, int(coarse_output_scale))
        self.coarse_output_shape = tuple(
            max(1, int(ceil(dim / self.coarse_output_scale)))
            for dim in self.output_shape
        )

        if upsample_scale <= 3:
            self.upsample_scale = upsample_scale + 1
        else:
            self.upsample_scale = upsample_scale

        if not isinstance(upsample_mode, (list, tuple)):
            upsample_mode = [upsample_mode] * n_scales
        if not isinstance(kernel_size_encoding, (list, tuple)):
            kernel_size_encoding = [kernel_size_encoding] * n_scales
        if not isinstance(kernel_size_decoding, (list, tuple)):
            kernel_size_decoding = [kernel_size_decoding] * n_scales

        self.model = torch.nn.Sequential()
        model = self.model

        for i in range(len(ch_encoding)):
            model_ = torch.nn.Sequential()
            skip = torch.nn.Sequential()

            if ch_skip[i] != 0:
                model.add(Concat(1, skip, model_))
            else:
                model.add(model_)

            model.add(
                torch.nn.BatchNorm2d(ch_skip[i] +
                                     (ch_decoding[i + 1] if i <
                                      (n_scales - 1) else ch_encoding[i])))

            if ch_skip[i] != 0:
                skip.add(
                    pad_and_conv(
                        ch_in,
                        ch_skip[i],
                        kernel_size_skip,
                        bias=bias,
                    ))
                skip.add(torch.nn.BatchNorm2d(ch_skip[i]))
                skip.add(activation_fn)
                # model.add(torch.nn.Dropout(0.1))

            model_.add(
                pad_and_conv(
                    ch_in,
                    ch_encoding[i],
                    kernel_size_encoding[i],
                    2,
                    bias=bias,
                ))
            model_.add(torch.nn.BatchNorm2d(ch_encoding[i]))
            model_.add(activation_fn)
            # model.add(torch.nn.Dropout(0.1))

            model_.add(
                pad_and_conv(
                    ch_encoding[i],
                    ch_encoding[i],
                    kernel_size_encoding[i],
                    bias=bias,
                ))
            model_.add(torch.nn.BatchNorm2d(ch_encoding[i]))
            model_.add(activation_fn)
            # model.add(torch.nn.Dropout(0.1))

            deeper_main = torch.nn.Sequential()
            if i == len(ch_encoding) - 1:
                k = ch_encoding[i]
            else:
                model_.add(deeper_main)
                k = ch_decoding[i + 1]

            if not upsample_mode[i] == 'nearest':
                model_.add(
                    torch.nn.Upsample(scale_factor=self.upsample_scale,
                                      mode=upsample_mode[i],
                                      align_corners=True))


            else:
                model_.add(
                    torch.nn.Upsample(scale_factor=self.upsample_scale, mode=upsample_mode[i]))

            model.add(
                pad_and_conv(
                    ch_skip[i] + k,
                    ch_decoding[i],
                    kernel_size_decoding[i],
                    1,
                    bias=bias,
                ))
            model.add(torch.nn.BatchNorm2d(ch_decoding[i]))
            model.add(activation_fn)
            # model.add(torch.nn.Dropout(0.1))

            model.add(
                pad_and_conv(
                    ch_decoding[i],
                    ch_decoding[i],
                    1,
                    bias=bias,
                ))
            model.add(torch.nn.BatchNorm2d(ch_decoding[i]))
            model.add(activation_fn)
            # model.add(torch.nn.Dropout(0.1))

            ch_in = ch_encoding[i]
            model = deeper_main

        self.model.add(pad_and_conv(
            ch_decoding[0],
            ch_out,
            1,
            bias=bias,
        ))
        # model.add(torch.nn.BatchNorm2d(ch_out))
        # self.model.add(activation_fn)
        # self.model.add(CustomActivation(threshold=(0.04, 0.44)))
        # self.model.add(torch.nn.Softplus(beta=.05, threshold=20))

        # self.model.add(torch.nn.Sigmoid())
        self.model.add(torch.nn.Upsample(size=self.coarse_output_shape, mode=upsample_mode[i]))
        # self.model.add(Crop(output_shape, n_scales))

        self.scale = torch.nn.Parameter(torch.tensor(1./30.))

        self.model.add(Scale(scaling_factor=self.scale))

        if Mask == None:
            self.Mask = torch.ones_like(Vp, dtype=torch.float32)
        else:
            self.Mask = Mask

        self.scaling_factor = scaling_factor

        # self.mmin_raw = nn.Parameter(torch.tensor(0.0))   # trainable scalar


        # self.tanh = nn.Tanh()

        # self.vp_ref_max = Vp.clone().detach().max().item()
        # self.Vp = Vp / self.vp_ref_max #F.interpolate(Vp / self.vp_ref_max, size=(Vp.size()[-2], Vp.size()[-1]), mode='bilinear', align_corners=False)

        Vp_ref = F.interpolate(Vp.clone().detach(), size=(Vp.size()[-2], Vp.size()[-1]), mode='bilinear', align_corners=False)

        p = (Vp_ref - self.scaling_factor[0]) / (self.scaling_factor[1] - self.scaling_factor[0])
        p = p.clamp(1e-6, 1.0 - 1e-6)   # evita inf/NaN no logit
        logit_map = torch.log(p / (1.0 - p))
        self.register_buffer('logit_map', logit_map)
        # Vp_ref = Vp.clone().detach()
        # self.register_buffer('Vp_ref', Vp_ref)


        # self.model.add(NormalizeActivation(threshold=scaling_factor))

        # self.model.add(CustomActivation(threshold=scaling_factor))
        self.count = 0

        print(f'vp: {Vp.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32).min()} {Vp.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32).max()}')
        print(f'vp interpolado: {Vp_ref.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32).min()} {Vp_ref.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32).max()}')


        # import matplotlib.pyplot as plt
        # cmap = 0.44
        # plt.figure(figsize=(10, 5))  # Defina o tamanho da figura (largura, altura)

        # # Subplot 1: Primeira imagem
        # plt.subplot(1, 2, 1)  # 1 linha, 2 colunas, primeiro subplot

        # plt.imshow(Vp.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32).T, cmap='jet', vmin=0, vmax=cmap, aspect='auto')
        # plt.title('vp real')  # Adicione um título opcional
        # # plt.text(10, 10, 'Legenda da Imagem 1', color='white', fontsize=10)

        # plt.subplot(1, 2, 2)  # 1 linha, 2 colunas, primeiro subplot

        # plt.imshow(Vp_ref.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32).T, cmap='jet', vmin=0, vmax=cmap, aspect='auto')
        # plt.title('vp interpolado')  # Adicione um título opcional
        # # plt.text(10, 10, 'Legenda da Imagem 1', color='white', fontsize=10)

        # plt.tight_layout()
        # plt.savefig('teste')



    def get_latent_shape(self) -> List[int]:
        """Get the latent variable shape that results in correct output shape.

        Returns:
            A list of integers containing the latent variable shape.
        """
        return [1, self.ch_in] + self.model[-2].latent_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the DeepPrior network.

        Args:
            x: A torch.Tensor containing the input latent variable.

        Returns:
            A torch.Tensor containing the output tensor (image).
        """
        # x = ((((vmax - vmin) * self.tanh(self.model(x)) + (vmax + vmin)) / 2.0)) * 0.65


        # x = (self.tanh(self.model(x)) + 1) * (self.scaling_factor[1] - self.scaling_factor[0]) / 2.0 + self.scaling_factor[0]

        # print(f'delta_vp: {x.min()} {x.max()}')

        # x = x + self.Vp * self.vp_ref_max

        # x = F.interpolate(x, size=(x.size()[-2], x.size()[-1]), mode='bilinear', align_corners=False)

        # if self.count % 100 == 0:
        #     import matplotlib.pyplot as plt

        #     plt.imshow(x.detach().cpu().numpy()[0, 0, ...].T, cmap='seismic', aspect='auto')
        #     plt.show()

        # print(f'model limits: {self.model(x).min()} -- {self.model(x).max()}')
        # p_min = (self.scaling_factor[0] / self.Vp_ref) - 1
        # p_max = (self.scaling_factor[1] / self.Vp_ref) - 1

        # dx = self.model(x) * (p_max - p_min) + p_min
        # x = self.Vp_ref * (1. + dx)

        # x = self.model(x) + self.Vp_ref
        # # print(f'model limits: {x.min()} -- {x.max()}')
        # if self.Mask is not None:
        #     x = self.Mask * x + (1.0 - self.Mask) * self.Vp_ref #* self.vp_ref_max

        # # x = self.model(x) + self.scaling_factor[0]
        # # x = self.Mask * x + (1.0 - self.Mask) * self.scaling_factor[1] #* self.vp_ref_max

        # # fator = torch.randn_like(x) * 0.001 + self.scaling_factor[0] #np.random.normal(self.scaling_factor[0], 0.005)
        # # fator = F.interpolate(fator.clone().detach(), size=(fator.size()[-2], fator.size()[-1]), mode='bilinear', align_corners=False)
        # # x = torch.where(x < self.scaling_factor[0], fator, x)
        # x[x < self.scaling_factor[0]] = self.scaling_factor[0]
        # x[x > self.scaling_factor[1]] = self.scaling_factor[1]

        # compute logit map from Vp_ref (do this each forward; cheap)
        # normalize p in (0,1)
        # print(f'self.scale: {self.scale}')
        # combine: shift raw output so s=0 -> m == Vp_ref
        # config
        # delta = 0.05 * self.scaling_factor[0]            # allow vmax in [vmax_nominal-delta, vmax_nominal+delta]

        # mmin = self.scaling_factor[0] + delta * torch.tanh(self.mmin_raw)   # shape scalar
        mmin = self.scaling_factor[0]
        # print(f'mmin? {mmin}')
        # p = (self.Vp_ref - mmin) / (self.scaling_factor[1] - mmin)
        # p = p.clamp(1e-6, 1.0 - 1e-6)   # evita inf/NaN no logit
        # logit_map = torch.log(p / (1.0 - p))

        x_scaled = self.model(x)
        if x_scaled.shape[-2:] != self.logit_map.shape[-2:]:
            x_scaled = F.interpolate(
                x_scaled,
                size=(self.logit_map.size()[-2], self.logit_map.size()[-1]),
                mode='bilinear',
                align_corners=False,
            )

        x = mmin + (self.scaling_factor[1] - mmin) * torch.sigmoid(x_scaled + self.logit_map)
        self.count += 1
        return x



class Generator(nn.Module):
    def __init__(self, input_shape, Vp, Mask=None, kernel_size=4, num_layer=5, h0=4, w0=8, Vp_min=None, Vp_max=None):
        super(Generator, self).__init__()

        self.Vp = F.interpolate(Vp.clone().detach(), size=(Vp.size()[-2], Vp.size()[-1]), mode='bilinear', align_corners=False)
        self.vp_min = Vp_min
        self.vp_max = Vp_max
        self.size = self.Vp.size()
        self.vp_ref_max = self.Vp.clone().detach().max().item()
        Vp_ref = self.Vp.clone().detach() #/  self.vp_ref_max #
        self.Vp_mask_pad = self.Vp.clone() #/ self.vp_ref_max
        self.register_buffer('Vp_ref', Vp_ref)

        self.num_layer = num_layer
        self.h0 = h0
        self.w0 = w0
        # self.vmin = vmin
        # self.vmax = vmax
        self.shp1, self.shp2 = Vp.size()[2:]

        self.activation = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()
        self.dense = nn.Linear(in_features=input_shape, out_features=self.h0 * self.w0 * self.num_layer, bias=False)
        # self.custon_activation = CustomActivation(threshold=(self.vmin, self.vmax))
        # self.custon_activation = NormalizeActivation(threshold=(self.vmin, self.vmax))


        # Amount to pad.
        # to_pad = int((kernel_size - 1) / 2)
        # # Reflect padding.
        # self.padding_layer = torch.nn.ReflectionPad2d(to_pad)

        self.conv_layers = nn.ModuleList()
        self.bt_layers = nn.ModuleList()

        self.conv_layers.append(nn.Conv2d(self.num_layer, 2 ** (num_layer + 3), kernel_size=kernel_size, stride=1, padding=0, bias=False))
        self.bt_layers.append(torch.nn.BatchNorm2d(2 ** (num_layer + 3)))
        for i in range(num_layer - 1):
            conv = nn.Conv2d(2 ** (num_layer - i + 3), 2 ** (num_layer - i + 2), kernel_size=kernel_size, stride=1, padding=0, bias=False)
            self.conv_layers.append(conv)
            self.bt_layers.append(torch.nn.BatchNorm2d(2 ** (num_layer - i + 2)))

        self.conv_final = nn.Conv2d(2 ** (4), 1, kernel_size=4, stride=1, padding=0, bias=False)

        if Mask == None:
            self.Mask = torch.ones_like(self.Vp, dtype=torch.float32)
        else:
            self.Mask = Mask

    def forward(self, z, vmax, vmin):

        self.Vp_mask_pad = self.Mask * self.Vp + (1.0 - self.Mask) * self.Vp_ref

        x = self.dense(z)
        x = x.view(-1, self.num_layer, self.h0, self.w0)
        # x = -self.tanh(x)
        # print(f'shape: {x.size()}')


        for i in range(self.num_layer):
            # x = self.padding_layer(x)
            # x = F.pad(x, (0, 1, 0, 1), mode='constant')
            # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = torch.nn.Upsample(scale_factor=2, mode='nearest')(x)

            x = self.conv_layers[i](x)
            x = self.bt_layers[i](x)
            x = self.activation(x)
            # print(f'shape: {x.size()}')
        # print(x)
        x = self.conv_final(x) * 1/100.
        # delta_vp = torch.nn.Upsample(size=(self.shp1, self.shp2), mode='bilinear')(x)

        # x /= x.detach().abs().max()
        # x = torch.clip(x, min=-torch.quantile(x.detach().abs(), 0.99), max=torch.quantile(x.detach().abs(), 0.99))

        # delta_vp = ((((vmax-vmin) * self.tanh(x) + (vmax + vmin)) / 2.0)) #* 0.65

        # print(f'x: {(x).min()} {(x).max()}')

        crop_height = (x.shape[2] - self.shp1) // 2 + 1
        crop_width = (x.shape[3] - self.shp2) // 2 + 1
        delta_vp = x[:, :, crop_height:crop_height + self.shp1, crop_width : crop_width + self.shp2] #* 0.02

        self.Vp_mask_pad += delta_vp

        self.Vp_mask_pad = self.Mask * self.Vp_mask_pad + (1.0 - self.Mask) * self.Vp_ref

        # print(f'min: {self.Vp_mask_pad.min()}  max: {self.Vp_mask_pad.max()}')


        # self.Vp_mask_pad = self.Mask * self.Vp_mask_pad * self.vp_ref_max + (1.0 - self.Mask) * self.Vp


        self.Vp_mask_pad[self.Vp_mask_pad < self.vp_min] = self.vp_min
        self.Vp_mask_pad[self.Vp_mask_pad > self.vp_max] = self.vp_max


        # if self.vmin is not None and self.vmax is not None:
        #     x = self.custon_activation(x)
        #     # x = ((self.vmax - self.vmin) * self.tanh(x) + (self.vmax + self.vmin)) / 2.0

        return self.Vp_mask_pad



def padding(cp, mask, nx, nz, nPad):
    tran_cp2 = F.interpolate(cp, size=(nx, nz), mode='bilinear', align_corners=False)
    # tran_cp3 = F.pad(tran_cp2, pad=(0, 0, 0, (nPad)), mode='replicate')
    # tran_mask = F.pad(mask, pad=(0, 0, 0, (nPad)), mode='replicate')
    # cp_pad = tran_cp3.view(nx, nz+nPad)
    # mask_pad = tran_mask.view(nx, nz+nPad)
    return tran_cp2, mask


class NN_FWI(torch.nn.Module):
    def __init__(self, Vp, Mask=None, Vp_min=None, Vp_max=None, l0=8, h0=5, w0=13):
        super(NN_FWI, self).__init__()

        if Mask == None:
            self.Mask = torch.ones_like(Vp, dtype=torch.float32)
        else:
            self.Mask = Mask


        self.vp_min = Vp_min
        self.vp_max = Vp_max
        self.vp_ref_max = Vp.clone().detach().max().item()
        self.Vp = F.interpolate(Vp / self.vp_ref_max, size=(Vp.size()[-2], Vp.size()[-1]), mode='bilinear', align_corners=False)
        self.size = self.Vp.size()

        Vp_ref = self.Vp.clone().detach()  #F.interpolate(Vp.clone().detach(), size=(self.size[0], self.size[1]), mode='bilinear', align_corners=False)
        self.Vp_mask_pad = self.Vp.clone()
        self.register_buffer('Vp_ref', Vp_ref)
        self.act = nn.Tanh()
        self.act2 = nn.LeakyReLU(0.1, inplace=True)
        # self.padding_layer = torch.nn.ReplicationPad2d(2)
        self.mode = 'bilinear'

        self.h0=h0
        self.w0=w0
        self.l0 = l0

        self.head_vp = nn.Linear(in_features=l0, out_features=self.h0*self.w0*self.l0)
        self.body_vp = nn.Sequential(nn.Upsample(scale_factor=2, mode=self.mode),
                             nn.Conv2d(in_channels=self.l0, out_channels=128, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), bias=False),
                             self.act2)
        self.tail_vp = nn.Sequential(nn.Upsample(scale_factor=2, mode=self.mode),
                             nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2), bias=False),
                             self.act2,
                             nn.Upsample(scale_factor=2, mode=self.mode),
                             nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2), bias=False),
                             self.act2,
                             nn.Upsample(scale_factor=2, mode=self.mode),
                             nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2), bias=False),
                             self.act2,
                             nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2), bias=False),
                             self.act2,
                             nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), bias=False),
                             self.act2,
                             nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(2, 2), stride=(1, 1), padding=(2, 2), bias=False))



    def forward(self, z, vmax, vmin):

        # Vp = F.interpolate(self.Vp / self.Vp.max() , size=self.Vp.size()[2:], mode='bilinear', align_corners=False)
        self.Vp_mask_pad = self.Mask * self.Vp + (1.0 - self.Mask) * self.Vp_ref

        delta_vp = self.head_vp(z)
        delta_vp = torch.reshape(delta_vp, (-1, self.l0, self.h0, self.w0))
        delta_vp = self.act(delta_vp)
        delta_vp = self.body_vp(delta_vp)
        delta_vp = self.tail_vp(delta_vp) / 1e3
        # delta_vp = self.act2(delta_vp) / self.Vp.max()





        # delta_vp[delta_vp < vmin] = vmin
        # delta_vp[delta_vp > vmax] = vmax

        # delta_vp /= self.Vp.max()




        # delta_vp = ((((vmax-vmin) * self.act(delta_vp) + (vmax + vmin)) / 2.0)) * 0.65
        print(f'delta_vp: {(delta_vp).min()} {(delta_vp).max()}')

        # print(f'delta_vp: {self.act(delta_vp).min()} {self.act(delta_vp).max()}')

        print(f'dim delta_vp: {delta_vp.size()}')

        # print(f'dim delta_vp: {delta_vp.size()}')
        # delta_vp = torch.squeeze(delta_vp)


        crop_height = (delta_vp.shape[2] - self.size[2]) // 2 + 1
        crop_width = (delta_vp.shape[3] - self.size[3]) // 2 + 1
        delta_vp = delta_vp[:, :, crop_height:crop_height + self.Vp_mask_pad.shape[2],
                            crop_width : crop_width + self.Vp_mask_pad.shape[3]]





        # import matplotlib.pyplot as plt

        # plt.imshow(delta_vp.detach().cpu().numpy()[..., 0, 0].T, cmap='seismic', aspect='auto')
        # plt.show()


        self.Vp_mask_pad += delta_vp / self.vp_ref_max

        self.Vp_mask_pad = (self.Mask * self.Vp_mask_pad + (1.0 - self.Mask) * self.Vp_ref) * self.vp_ref_max

        # print(f'min: {self.Vp_mask_pad.min()}  max: {self.Vp_mask_pad.max()}')


        # self.Vp_mask_pad = self.Mask * self.Vp_mask_pad * self.vp_max + (1.0 - self.Mask) * self.Vp


        self.Vp_mask_pad[self.Vp_mask_pad < self.vp_min] = self.vp_min
        self.Vp_mask_pad[self.Vp_mask_pad > self.vp_max] = self.vp_max

        return self.Vp_mask_pad



class Generator_(nn.Module):
    def __init__(self, vmin=None, vmax=None):
        super(Generator_, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.network = self.build_network()
        self.vmin = vmin
        self.vmax = vmax
        self.custon_activation = NormalizeActivation(threshold=(self.vmin, self.vmax))


    def build_network(self):
        blocks = []
        blocks += [nn.ConvTranspose3d(20, 512, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(512),  self.activation]
        blocks += [nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(256),  self.activation]
        blocks += [nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128),  self.activation]
        blocks += [nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64),  self.activation]
        blocks += [nn.Conv3d(64, 1, kernel_size=3, stride=1, padding=1), nn.Tanh()]
        return nn.Sequential(*blocks)

    def forward(self, z):
        x = self.network(z)
        x = self.custon_activation(x)
        shp1, shp2 = 336, 112
        crop_height = (x.shape[2] - shp1) // 2
        crop_width = (x.shape[3] - shp2) // 2
        x = x[:, :, crop_height:crop_height + shp1, crop_width : crop_width + shp2]
        return x


class Model_vp(torch.nn.Module):
    def __init__(self, initial, min_vel, max_vel):
        super().__init__()
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.model = torch.nn.Parameter(
            torch.logit((initial - min_vel) /
                        (max_vel - min_vel))
        )

    def forward(self):
        return torch.sigmoid(self.model) * (self.max_vel - self.min_vel) + self.min_vel


class Model_v(torch.nn.Module):
    def __init__(self, initial, min_vel, max_vel):
        super().__init__()
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.CustomActivation = CustomActivation(threshold=(self.min_vel, self.max_vel))
        self.model = torch.nn.Parameter(
            initial
        )

    def forward(self):
        return self.CustomActivation(self.model)
        # self.model.data[self.model.data < self.min_vel] = self.min_vel
        # self.model.data[self.model.data > self.max_vel] = self.max_vel
        # return self.model
