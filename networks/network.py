import torch as th
import torch.nn as nn
import torch.nn.functional as F



class encoder(nn.Module):
    def __init__(self, ):
        super(encoder, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3, bias=True)
        self.lrelu =  nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.lrelu(x)
        x = self.conv1_2(x)
        x = self.lrelu(x)
        conv1 = x
        p1 = self.maxpool(x)
        x = self.conv2_1(p1)
        x = self.lrelu(x)
        x = self.conv2_2(x)
        x = self.lrelu(x)
        conv2 = x
        p2 = self.maxpool(x)
        x = self.conv3_1(p2)
        x = self.lrelu(x)
        x = self.conv3_2(x)
        x = self.lrelu(x)
        conv3 = x
        p3 = self.maxpool(x)
        x = self.conv4_1(p3)
        x = self.lrelu(x)
        x = self.conv4_2(x)
        x = self.lrelu(x)
        conv4 = x
        p4 = self.maxpool(x)
        x = self.conv5_1(p4)
        x = self.lrelu(x)
        x = self.conv5_2(x)
        x_en = self.lrelu(x)
        return conv1,conv2,conv3,conv4,p1,p2,p3,p4,x_en


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class GEunit(nn.Module):
    def __init__(self, nIn, nOut):
        super(GEunit, self).__init__()
        self.d1 = nn.Conv2d(nIn, nOut, kernel_size=3, stride=2, padding=1, bias=True)
        self.d2 = nn.Conv2d(nOut, nOut, kernel_size=3, stride=2, padding=1, bias=True)
        self.d3 = nn.Conv2d(nOut, nOut, kernel_size=3, stride=2, padding=1, bias=True)
        self.interp = nn.Upsample(scale_factor=8, mode='bilinear')
        self.lrelu =  nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.act = nn.Sigmoid()
    def forward(self, x):
        y = self.lrelu(self.d1(x))
        y = self.lrelu(self.d2(y))
        y = self.d3(y)
        y = self.interp(y)
        attention_map = self.act(y)
        return attention_map*x

class LCunit(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=2, d=1):
        super(LCunit, self).__init__()
        self.d1 = nn.Conv2d(nIn, nOut, kernel_size=3, stride=1, padding=1, groups=nOut, bias=True, dilation=1)
        self.d2 = nn.Conv2d(nIn, nOut, kernel_size=3, stride=1, padding=5, bias=True, dilation=5)
        self.lrelu =  nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.act = nn.Sigmoid()
    def forward(self, x):
        y1 = self.lrelu(self.d1(x))
        y2 = self.lrelu(self.d2(x))
        df = y1 - y2
        return self.act(df)*x

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class NonLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(NonLocalBlock2D, self).__init__()
        
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)
        
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        batch_size = x.size(0)
        
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        
        g_x = g_x.permute(0,2,1)
        
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        
        theta_x = theta_x.permute(0,2,1)
        
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        
        f = th.matmul(theta_x, phi_x)
       
        f_div_C = F.softmax(f, dim=1)
        
        
        y = th.matmul(f_div_C, g_x)
        
        y = y.permute(0,2,1).contiguous()
         
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

class AttentiveNonLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(AttentiveNonLocalBlock2D, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)        
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)

        self.attention = GEunit(nIn=in_channels,nOut=in_channels)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.attention(x)
        
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0,2,1)
        
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0,2,1)
        
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        
        f = th.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=1)
        
        y = th.matmul(f_div_C, g_x)
        y = y.permute(0,2,1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

class NonLocalAttentionBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(NonLocalAttentionBlock2D, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)        
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)

        self.attention = GEunit(nIn=in_channels,nOut=in_channels)

    def forward(self, x):
        batch_size = x.size(0)

        attention_map = self.attention(x)
        
        g_x = self.g(attention_map).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0,2,1)
        
        theta_x = self.theta(attention_map).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0,2,1)
        
        phi_x = self.phi(attention_map).view(batch_size, self.inter_channels, -1)
        
        f = th.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=1)
        
        y = th.matmul(f_div_C, g_x)
        y = y.permute(0,2,1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + attention_map
        return z*x

class contrast(nn.Module):
    def __init__(self, nIn, nOut, stride=1, d=1):
        super(contrast, self).__init__()
        self.d1 = nn.Conv2d(nIn, nOut, kernel_size=3, stride=1, padding=1, groups=nOut, bias=True, dilation=1)
        self.d2 = nn.Conv2d(nIn, nOut, kernel_size=3, stride=1, padding=5, bias=True, dilation=5)
        self.lrelu =  nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.act = nn.Sigmoid()
    def forward(self, x):
        y1 = self.lrelu(self.d1(x))
        y2 = self.lrelu(self.d2(x))
        df = th.abs(y1 - y2)
        return self.act(df)

class enBlock(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(enBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.contrast = contrast(self.in_channels, self.in_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        batch_size = x.size(0)

        contrast_map = self.contrast(x)
        reverse_contrast_map = 1 - contrast_map  #self.contrast(x) # low contrast are contents

        atten_feature = contrast_map *x
        r_atten_feature = reverse_contrast_map * x
        shrink_reverse_contrast_feature = self.maxpool(r_atten_feature) 
        shrink_reverse_contrast_feature = self.maxpool(shrink_reverse_contrast_feature)
        shrink_reverse_contrast_feature = self.maxpool(shrink_reverse_contrast_feature) # spatial/8

        
        theta_x = self.theta(shrink_reverse_contrast_feature).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0,2,1)
        
        phi_x = self.phi(shrink_reverse_contrast_feature).view(batch_size, self.inter_channels, -1)
        f = th.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=1)

        g_x = self.g(shrink_reverse_contrast_feature).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0,2,1)

        y = th.matmul(f_div_C, g_x)
        y = y.permute(0,2,1).contiguous()
        y = y.view(batch_size, self.inter_channels, *shrink_reverse_contrast_feature.size()[2:])
        y = self.up(y)
        y = self.up(y)
        y = self.up(y)
        W_y = self.W(y)
        z = W_y + x
        return z, contrast_map, atten_feature#reverse_contrast_map, r_atten_feature

class enBlock2(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(enBlock2, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.contrast = contrast(self.in_channels, self.in_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x, amap, fmap):
        batch_size = x.size(0)

        contrast_map = self.contrast(x) + amap  #self.contrast(x) 
        atten_feature = contrast_map * x + self.contrast(fmap)
        shrink_reverse_contrast_feature = self.maxpool(contrast_map * x) 
        shrink_reverse_contrast_feature = self.maxpool(shrink_reverse_contrast_feature)
        shrink_reverse_contrast_feature = self.maxpool(shrink_reverse_contrast_feature) # spatial/8

        
        theta_x = self.theta(shrink_reverse_contrast_feature).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0,2,1)
        
        phi_x = self.phi(shrink_reverse_contrast_feature).view(batch_size, self.inter_channels, -1)
        f = th.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=1)

        g_x = self.g(shrink_reverse_contrast_feature).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0,2,1)

        y = th.matmul(f_div_C, g_x)
        y = y.permute(0,2,1).contiguous()
        y = y.view(batch_size, self.inter_channels, *shrink_reverse_contrast_feature.size()[2:])
        y = self.up(y)
        y = self.up(y)
        y = self.up(y)
        W_y = self.W(y)
        z = W_y + x
        return z #contrast_map, atten_feature

class network(nn.Module):
    def __init__(self, ):
        super(network, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.lrelu =  nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.block1 = enBlock(32,32)
        self.block2 = enBlock2(32,32)
        #self.block2 = enBlock(64/2,64/2)
        #self.ratio = nn.Parameter(th.tensor(250.0), True)

        # self.encoder = encoder()
        # #self.nonlocal = NonLocalBlock2D(512,64)
        # self.atnonlocal1 = AttentiveNonLocalBlock2D(64,32)
        # #self.atnonlocal2 = AttentiveNonLocalBlock2D(64,32)
        # #self.nonlocalattention = NonLocalAttentionBlock2D(512,128)
        # self.lrelu =  nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # decoder part
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2, bias=True)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2, bias=True)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2, bias=True)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2, bias=True)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv10 = nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv10_mul = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv10_add = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0, bias=True)

        self.ca4 = CALayer(512)
        #self.ge4 = GEunit(512,512)
        self.ca3 = CALayer(256)
        #self.ge3 = GEunit(256,256)
        self.ca2 = CALayer(128)
        #self.ge2 = GEunit(128,128)
        self.ca1 = CALayer(64)
        # self.ge1 = GEunit(64,64)
        self.lc4 = contrast(256,256)
        self.lc3 = contrast(128,128)
        self.lc2 = contrast(64,64)
        self.lc1 = contrast(32,32)
        self.sigmoid = nn.Sigmoid()
        self._initialize_weights()
        self.ratio = nn.Parameter(th.tensor(1.0), True)


    def _initialize_weights(self):
        checkpoint = th.load('/home/kevin/Desktop/lowlight/latest')
        cur_dict = self.state_dict()
        pretrained_dict = checkpoint['net']
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
        cur_dict.update(pretrained_dict)
        self.load_state_dict(cur_dict)

    def forward(self, x_in):
        # encoder part
        #x_a = x_in * self.ratio
        x = self.conv1_1(x_in)
        x = self.lrelu(x)
        x, x_r, x_f = self.block1(x)
        x = self.conv1_2(x)
        x = self.lrelu(x)
        conv1 = x
        p1 = self.maxpool(x)
        x = self.conv2_1(p1)
        x = self.lrelu(x)
        x = self.conv2_2(x)
        x = self.lrelu(x)
        conv2 = x
        p2 = self.maxpool(x)
        x = self.conv3_1(p2)
        x = self.lrelu(x)
        x = self.conv3_2(x)
        x = self.lrelu(x)
        conv3 = x
        p3 = self.maxpool(x)
        x = self.conv4_1(p3)
        x = self.lrelu(x)
        x = self.conv4_2(x)
        x = self.lrelu(x)
        conv4 = x
        p4 = self.maxpool(x)
        x = self.conv5_1(p4)
        x = self.lrelu(x)
        x = self.conv5_2(x)
        x_en = self.lrelu(x)

        # stage 1, low contrast -> mul
        conv1 = (1-self.lc1(conv1)) * conv1
        conv2 = (1-self.lc2(conv2)) * conv2
        conv3 = (1-self.lc3(conv3)) * conv3
        conv4 = (1-self.lc4(conv4)) * conv4
        x = self.up6(x_en)
        x = th.cat((x, conv4), 1)
        x = self.ca4(x)
        x = self.conv6_1(x)
        x = self.lrelu(x)
        x = self.conv6_2(x)
        x = self.lrelu(x)
        x = self.up7(x)
        x = th.cat((x, conv3), 1)
        x = self.ca3(x)
        x = self.conv7_1(x)
        x = self.lrelu(x)
        x = self.conv7_2(x)
        x = self.lrelu(x)
        x = self.up8(x)
        x = th.cat((x, conv2), 1)
        x = self.ca2(x)
        x = self.conv8_1(x)
        x = self.lrelu(x)
        x = self.conv8_2(x)
        x = self.lrelu(x)
        x = self.up9(x)
        x = th.cat((x, conv1), 1)
        x = self.ca1(x)
        x = self.conv9_1(x)
        x = self.lrelu(x)
        x = self.conv9_2(x)
        x = self.lrelu(x)
        xc = self.conv10(x)
        mul = th.tanh(self.conv10_mul(xc))
############################################################## stage 2
        x_mul = xc * mul * self.ratio
        x = self.conv1_1(x_mul)
        x = self.lrelu(x)
        #print type(x)
        #print type(x_r)
        x = self.block2(x,x_r,x_f)#ACE_map, atten_feature
        #print type(x)
        x = self.conv1_2(x)
        x = self.lrelu(x)
        conv1 = x
        p1 = self.maxpool(x)
        x = self.conv2_1(p1)
        x = self.lrelu(x)
        x = self.conv2_2(x)
        x = self.lrelu(x)
        conv2 = x
        p2 = self.maxpool(x)
        x = self.conv3_1(p2)
        x = self.lrelu(x)
        x = self.conv3_2(x)
        x = self.lrelu(x)
        conv3 = x
        p3 = self.maxpool(x)
        x = self.conv4_1(p3)
        x = self.lrelu(x)
        x = self.conv4_2(x)
        x = self.lrelu(x)
        conv4 = x
        p4 = self.maxpool(x)
        x = self.conv5_1(p4)
        x = self.lrelu(x)
        x = self.conv5_2(x)
        x_en = self.lrelu(x)

        conv1 = (self.lc1(conv1)) * conv1
        conv2 = (self.lc2(conv2)) * conv2
        conv3 = (self.lc3(conv3)) * conv3
        conv4 = (self.lc4(conv4)) * conv4
        x = self.up6(x_en)
        x = th.cat((x, conv4), 1)
        x = self.ca4(x)
        x = self.conv6_1(x)
        x = self.lrelu(x)
        x = self.conv6_2(x)
        x = self.lrelu(x)
        x = self.up7(x)
        x = th.cat((x, conv3), 1)
        x = self.ca3(x)
        x = self.conv7_1(x)
        x = self.lrelu(x)
        x = self.conv7_2(x)
        x = self.lrelu(x)
        x = self.up8(x)
        x = th.cat((x, conv2), 1)
        x = self.ca2(x)
        x = self.conv8_1(x)
        x = self.lrelu(x)
        x = self.conv8_2(x)
        x = self.lrelu(x)
        x = self.up9(x)
        x = th.cat((x, conv1), 1)
        x = self.ca1(x)
        x = self.conv9_1(x)
        x = self.lrelu(x)
        x = self.conv9_2(x)
        x = self.lrelu(x)
        x = self.conv10(x)
        add = self.conv10_add(x)
        res = add

        res = th.tanh(res)
        #print ACE_r_map.shape
        #print ACE_map.shape
        return xc, x_mul, res
        #for testing
        #return xc, mul, x_mul, add, res, ACE_r_map, ACE_map, r_atten_feature, atten_feature
