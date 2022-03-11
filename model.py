import jittor as jt
import jittor.nn as nn
import numpy as np
import random
from math import sqrt
from jittor.models import vgg16

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight[0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        delattr(module, name)
        setattr(module, name + '_orig', weight)
        module.register_pre_forward_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)

def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class EqualConv2d(jt.Module):
    def __init__(self, *args, **kwargs):
        conv = jt.nn.Conv2d(*args, **kwargs)
        jt.init.gauss_(conv.weight, 0, 1)
        jt.init.constant_(conv.bias,0)
        
        self.conv = equal_lr(conv)

    def execute(self, input):
        return self.conv(input)

class EqualLinear(jt.Module):
    def __init__(self, in_dim, out_dim):
        linear = jt.nn.Linear(in_dim, out_dim)
        jt.init.gauss_(linear.weight, 0, 1)
        jt.init.constant_(linear.bias, 0)

        self.linear = equal_lr(linear)

    def execute(self, input):
        return self.linear(input)
    
class BlurFunctionBackward(jt.Function):
    def execute(self, grad_output, kernel, kernel_flip):
        self.saved_tensors = kernel, kernel_flip

        grad_input = jt.nn.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    def grad(self, gradgrad_output):
        kernel, kernel_flip = self.saved_tensors

        grad_input = jt.nn.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None

class BlurFunction(jt.Function):
    def execute(self, input, kernel, kernel_flip):
        self.saved_tensors = kernel, kernel_flip

        output = jt.nn.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    def grad(self, grad_output):
        kernel, kernel_flip = self.saved_tensors

        grad_input = BlurFunctionBackward().execute(grad_output, kernel, kernel_flip)

        return grad_input, None, None

blur = BlurFunction().apply

class Blur(jt.Module):
    def __init__(self, channel):
        weight = jt.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype='float32')
        weight = weight.reshape(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = jt.flip(weight, [2, 3])

        self._weight = weight.repeat(channel, 1, 1, 1)
        self._weight_flip = weight_flip.repeat(channel, 1, 1, 1)

    def execute(self, input):
        return blur(input, self._weight, self._weight_flip)
        # return jt.nn.conv2d(input, self.weight, padding=1, groups=input.shape[1])
        
class FusedDownsample(jt.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        self.weight = jt.randn(out_channel, in_channel, kernel_size, kernel_size)
        self.bias = jt.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)
        self.pad = padding

    def execute(self, input):
        weight = jt.nn.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]  +
            weight[:, :, :-1, 1:] +
            weight[:, :, 1:, :-1] +
            weight[:, :, :-1, :-1]
        ) / 4

        out = jt.nn.conv2d(input, weight, self.bias, stride=2, padding=self.pad)
        return out

class FusedUpsample(jt.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        self.weight = jt.randn(in_channel, out_channel, kernel_size, kernel_size)
        self.bias = jt.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)
        self.pad = padding

    def execute(self, input):
        weight = jt.nn.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:] +
            weight[:, :, :-1, 1:] +
            weight[:, :, 1:, :-1] +
            weight[:, :, :-1, :-1]
        ) / 4

        out = jt.nn.conv_transpose2d(input, weight, self.bias, stride=2, padding=self.pad)
        return out

class ConvBlock(jt.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        padding,
        kernel_size2=None,
        padding2=None,
        downsample=False,
        fused=False
    ):
        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        self.conv1 = jt.nn.Sequential(
            EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
            jt.nn.LeakyReLU(0.2),
        )

        if downsample:
            if fused:
                self.conv2 = jt.nn.Sequential(
                    Blur(out_channel),
                    FusedDownsample(out_channel, out_channel, kernel2, padding=pad2),
                    jt.nn.LeakyReLU(0.2),
                )
            else:
                self.conv2 = jt.nn.Sequential(
                    Blur(out_channel),
                    EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                    jt.nn.AvgPool2d(2),
                    jt.nn.LeakyReLU(0.2),
                )
        else:
            self.conv2 = jt.nn.Sequential(
                EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                jt.nn.LeakyReLU(0.2),
            )
    def execute(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        return out

class Discriminator(jt.Module):
    def __init__(self, fused=True, from_rgb_activate=False):
        self.progression = jt.nn.ModuleList(
            [
                ConvBlock( 16,  32, 3, 1, downsample=True, fused=fused),  # 512
                ConvBlock( 32,  64, 3, 1, downsample=True, fused=fused),  # 256
                ConvBlock( 64, 128, 3, 1, downsample=True, fused=fused),  # 128
                ConvBlock(128, 256, 3, 1, downsample=True, fused=fused),  # 64
                ConvBlock(256, 512, 3, 1, downsample=True),  # 32
                ConvBlock(512, 512, 3, 1, downsample=True),  # 16
                ConvBlock(512, 512, 3, 1, downsample=True),  # 8
                ConvBlock(512, 512, 3, 1, downsample=True),  # 4
                ConvBlock(513, 512, 3, 1, 4, 0),
            ]
        )
        
        def make_from_rgb(out_channel):
            if from_rgb_activate:
                return jt.nn.Sequential(EqualConv2d(3, out_channel, 1), jt.nn.LeakyReLU(0.2))

            else:
                return EqualConv2d(3, out_channel, 1)

        self.from_rgb = jt.nn.ModuleList(
            [
                make_from_rgb(16),
                make_from_rgb(32),
                make_from_rgb(64),
                make_from_rgb(128),
                make_from_rgb(256),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
            ]
        )
        self.n_layer = len(self.progression)
        self.linear = EqualLinear(512, 1)
        
    def execute(self, input, step=0, alpha=-1):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            if i == 0:
                out_std = np.std(out.data, axis=0)
                mean_std = jt.array(out_std.mean())
                mean_std = mean_std.expand((out.size(0), 1, 4, 4))
                out = jt.concat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                if i == step and 0 <= alpha < 1:
                    skip_rgb = jt.nn.avg_pool2d(input, 2)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)

                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)
        out = self.linear(out)
        return out

class ConstantInput(jt.Module):
    def __init__(self, channel, size=4):
        self.input = jt.randn(1, channel, size, size)

    def execute(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out

class NoiseInjection(jt.Module):
    def __init__(self, channel):
        self.weight = jt.zeros((1, channel, 1, 1))

    def execute(self, image, noise):
        return image + self.weight * noise

class AdaptiveInstanceNorm(jt.nn.Module):
    def __init__(self, in_channel, style_dim):
        self.norm = jt.nn.InstanceNorm2d(in_channel, affine=False)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def execute(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out
    
class StyledConvBlock(jt.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        style_dim=512,
        initial=False,
        upsample=False,
        fused=False,
    ):
        if initial:
            self.conv1 = ConstantInput(in_channel)
        else:
            if upsample:
                if fused:
                    self.conv1 = jt.nn.Sequential(
                        FusedUpsample(
                            in_channel, out_channel, kernel_size, padding=padding
                        ),
                        Blur(out_channel),
                    )
                else:
                    self.conv1 = jt.nn.Sequential(
                        jt.nn.Upsample(scale_factor=2, mode='nearest'),
                        EqualConv2d(
                            in_channel, out_channel, kernel_size, padding=padding
                        ),
                        Blur(out_channel),
                    )
            else:
                self.conv1 = EqualConv2d(
                    in_channel, out_channel, kernel_size, padding=padding
                )

        self.noise1 = equal_lr(NoiseInjection(out_channel))
        self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu1 = jt.nn.LeakyReLU(0.2)

        self.conv2  = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.noise2 = equal_lr(NoiseInjection(out_channel))
        self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu2 = jt.nn.LeakyReLU(0.2)
    
    def execute(self, input, style, noise):
        out = self.conv1(input)
        out = self.noise1(out, noise)
        out = self.lrelu1(out)
        out = self.adain1(out, style)

        out = self.conv2(out)
        out = self.noise2(out, noise)
        out = self.lrelu2(out)
        out = self.adain2(out, style)

        return out

class PixelNorm(jt.Module):
    def __init__(self):
        pass

    def execute(self, input):
        return input / jt.sqrt(jt.mean(input ** 2, dim=1, keepdims=True) + 1e-8)

class Generator(jt.Module):
    def __init__(self, code_dim, fused=True):
        self.progression = jt.nn.ModuleList(
            [
                StyledConvBlock(512, 512, 3, 1, initial=True),   # 4
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 8
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 16
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 32
                StyledConvBlock(512, 256, 3, 1, upsample=True),  # 64
                StyledConvBlock(256, 128, 3, 1, upsample=True, fused=fused),  # 128
                StyledConvBlock(128,  64, 3, 1, upsample=True, fused=fused),  # 256
                StyledConvBlock( 64,  32, 3, 1, upsample=True, fused=fused),  # 512
                StyledConvBlock( 32,  16, 3, 1, upsample=True, fused=fused),  # 1024
            ]
        )

        self.to_rgb = jt.nn.ModuleList(
            [
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(256, 3, 1),
                EqualConv2d(128, 3, 1),
                EqualConv2d(64, 3, 1),
                EqualConv2d(32, 3, 1),
                EqualConv2d(16, 3, 1),
            ]
        )

    def execute(self, style, noise, step=0, alpha=-1, mixing_range=(-1, -1)):
        out = noise[0]

        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            if i > 0 and step > 0:
                out_prev = out
            out = conv(out, style[i], noise[i])
            if i == step:
                out = to_rgb(out)
                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i - 1](out_prev)
                    skip_rgb = jt.nn.interpolate(skip_rgb, scale_factor=2, mode='nearest')
                    out = (1 - alpha) * skip_rgb + alpha * out
                break

        return out

class StyledGenerator(jt.Module):
    def __init__(self, code_dim=512, n_mlp=8):
        self.generator = Generator(code_dim)

        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(jt.nn.LeakyReLU(0.2))

        self.style = jt.nn.Sequential(*layers)

    def execute(
        self,
        input,
        noise=None,
        step=0,
        alpha=-1,
        mean_style=None,
        style_weight=0,
        mixing_range=(-1, -1),
    ):
        #'''
        styles = []
        if type(input) not in (list, tuple):
            input = [input]

        for i in input:
            styles.append(self.style(i))
        #'''

        batch = input[0].shape[0]

        if noise is None:
            noise = []
            for i in range(step + 1):
                size = 4 * 2 ** i
                noise.append(jt.randn(batch, 1, size, size))

        return self.generator(styles, noise, step, alpha, mixing_range=mixing_range)

    def mean_style(self, input):
        style = self.style(input).mean(0, keepdims=True)

        return style

class Z2WSpace(jt.Module):
    def __init__(self, code_dim=512, n_mlp=8):
        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(jt.nn.LeakyReLU(0.2))
        self.style = jt.nn.Sequential(*layers)
    def execute(self, input):
        return self.style(input)
class ZP2WPSpace(jt.Module):
    def __init__(self, n_layer, code_dim=512, n_mlp=8):
        self.mlps = nn.Sequential()
        for i in range(n_layer):
            self.mlps.append(Z2WSpace(code_dim, n_mlp))
        self.n_layer = n_layer
    def execute(self, input):
        output = nn.Sequential()
        for i in range(self.n_layer):
            output.append(self.mlps[i](input[i]))
        return output
class FeatExtract(jt.Module):
    def __init__(self):
        resnet = jt.models.Resnet50(pretrained=True)
        self.feature_extract = nn.Sequential([
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
        ])
        self.feature_extract_2 = resnet.layer2
        self.feature_extract_3 = resnet.layer3
        self.feature_extract_4 = resnet.layer4
    def execute(self, ipt):
        high = self.feature_extract(ipt)
        #[12,256,32,32,]
        mid = self.feature_extract_2(high)
        #[12,512,16,16,]
        low = self.feature_extract_3(mid)
        #[12,1024,8,8,]
        low = self.feature_extract_4(low)
        #[12,2048,4,4,]
        return high, mid, low
class Map2Style(jt.Module):
    def __init__(self, in_channel, out_channel, in_size):
        self.map2style = nn.Sequential()
        while in_channel != out_channel or in_size != 1:
            next_channel = in_channel
            if next_channel < out_channel:
                next_channel *= 2
            if next_channel > out_channel:
                next_channel /= 2
            self.map2style.append(nn.Conv(int(in_channel), int(next_channel), kernel_size=3, stride=2, padding=1))
            self.map2style.append(nn.BatchNorm(int(next_channel)))
            self.map2style.append(nn.LeakyReLU(0.2))
            in_channel = next_channel
            in_size /= 2
    def execute(self, ipt):
        return self.map2style(ipt)
class VAE(jt.Module):
    def __init__(self, layers):
        self.map2style5 = Map2Style(256, 512, 32)
        self.map2style4 = Map2Style(256, 512, 32)
        self.map2style3 = Map2Style(512, 512, 16)
        self.map2style2 = Map2Style(512, 512, 16)
        self.map2style1 = Map2Style(2048, 512, 4)
        self.map2style0 = Map2Style(2048, 512, 4)
        self.layers = layers
        self.fc_mean = nn.Sequential()
        self.fc_logvar = nn.Sequential()
        for i in range(layers):
            self.fc_mean.append(nn.Sequential([
                nn.Linear(512, 512, bias=False),
                #nn.BatchNorm(512),
                #nn.LeakyReLU(0.1),
            ]))
            self.fc_logvar.append(nn.Sequential([
                nn.Linear(512, 512, bias=False),
                #nn.BatchNorm(512),
                #nn.LeakyReLU(0.1),
            ]))


    def execute(self, high, mid, low):
        mean = []
        logvar = []
        features = [self.map2style0(low), 
                    self.map2style1(low), 
                    self.map2style2(mid), 
                    self.map2style3(mid), 
                    self.map2style4(high), 
                    self.map2style5(high),
                    ]
        for i in range(self.layers):
            features[i] = features[i].reshape((features[i].shape[0], -1))
        for i in range(self.layers):
            mean.append(self.fc_mean[i](features[i]))
            logvar.append(self.fc_logvar[i](features[i]))

        return mean, logvar

class PerceptualLoss(nn.Module):
    def __init__(self, requires_grad=False):
        #self.register_buffer('mean_rgb', mean_rgb)
        #self.register_buffer('std_rgb', std_rgb)

        vgg_pretrained_features = vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def normalize(self, x):
        mean_rgb = jt.float32([0.485, 0.456, 0.406])
        std_rgb = jt.float32([0.229, 0.224, 0.225])
        out = x/2 + 0.5
        out = (out - jt.view(mean_rgb, (1,3,1,1))) / jt.view(std_rgb, (1,3,1,1))
        return out

    def __call__(self, im1, im2, mask=None, conf_sigma=None):
        im = jt.contrib.concat([im1,im2], 0)
        im = self.normalize(im)  # normalize input
        ## compute features
        feats = []
        f = self.slice1(im)
        feats += [jt.misc.chunk(f, 2, dim=0)]
        f = self.slice2(f)
        feats += [jt.misc.chunk(f, 2, dim=0)]
        f = self.slice3(f)
        feats += [jt.misc.chunk(f, 2, dim=0)]
        f = self.slice4(f)
        feats += [jt.misc.chunk(f, 2, dim=0)]
        losses = jt.float32(0)
        l_feats = len(feats)
        for f1, f2 in feats:
            loss = (f1-f2)**2
            loss = loss.mean()
            #loss = (f1 * f2).sum() / ((f1 * f1).sqrt() * (f2 * f2).sqrt())
            '''
            if conf_sigma is not None:
                loss = loss / (2*conf_sigma**2 +EPS) + (conf_sigma +EPS).log()
            if mask is not None:
                b, c, h, w = loss.shape
                _, _, hm, wm = mask.shape
                sh, sw = hm//h, wm//w
                mask0 = nn.avg_pool2d(mask, kernel_size=(sh,sw), stride=(sh,sw)).expand_as(loss)
                loss = (loss * mask0).sum() / mask0.sum()
            '''
            losses += loss / l_feats
        return losses
