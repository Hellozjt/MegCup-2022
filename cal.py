import megengine as mge
import megengine.module as M
import megengine.functional as F
import time
import numpy as np


class SCConv(M.Module):
    def __init__(self, planes, pooling_r):
        super(SCConv, self).__init__()
        self.k2 = M.Sequential(
            M.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            M.Conv2d(planes, planes, 3, 1, 1),
        )
        self.k3 = M.Sequential(
            M.Conv2d(planes, planes, 3, 1, 1),
        )
        self.k4 = M.Sequential(
            M.Conv2d(planes, planes, 3, 1, 1),
            M.LeakyReLU(0.2),
        )

    def forward(self, x):
        identity = x

        out = F.nn.sigmoid(identity+ F.nn.interpolate(self.k2(x), identity.shape[2:], mode='BILINEAR', align_corners=False))  # sigmoid(identity + k2)
        out = self.k3(x)* out  # k3 * sigmoid(identity + k2)
        out = self.k4(out)  # k4

        return out

class SCBottleneck(M.Module):
    # expansion = 4
    pooling_r = 4  # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, in_planes, planes):
        super(SCBottleneck, self).__init__()
        planes = int(planes / 2)

        self.conv1_a = M.Conv2d(in_planes, planes, 1, 1)
        self.k1 = M.Sequential(
            M.Conv2d(planes, planes, 3, 1, 1),
            M.LeakyReLU(0.2),
        )

        self.conv1_b = M.Conv2d(in_planes, planes, 1, 1)

        self.scconv = SCConv(planes, self.pooling_r)

        self.conv3 = M.Conv2d(planes * 2, planes * 2, 1, 1)
        self.relu = M.LeakyReLU(0.2)

    def forward(self, x):
        residual = x

        out_a = self.conv1_a(x)
        out_a = self.relu(out_a)

        out_a = self.k1(out_a)

        out_b = self.conv1_b(x)
        out_b = self.relu(out_b)

        out_b = self.scconv(out_b)

        out = self.conv3(F.concat([out_a, out_b], axis=1))

        out += residual


        return out

class SepConv(M.Module):
    def __init__(self,in_channels,kernel_size=3,stride=1):
        super(SepConv, self).__init__()
        self.sepconv = M.Sequential(M.Conv2d(in_channels,in_channels,kernel_size=kernel_size,padding=(kernel_size-1)//2,stride=stride,groups=in_channels),
                                    M.Conv2d(in_channels,in_channels,kernel_size=1),
                                    )
    def forward(self, inputs):
        return self.sepconv(inputs)
class ResSepConv(M.Module):
    def __init__(self,n_feat,kernel_size=3):
        super(ResSepConv, self).__init__()
        self.conv = M.Sequential(
            SepConv(n_feat,kernel_size),
            M.LeakyReLU(negative_slope=0.2),
            SepConv(n_feat,kernel_size)
        )
    def forward(self, inputs):
        return inputs+self.conv(inputs)
class SKFF(M.Module):
    def __init__(self, in_channels, height=3, reduction=4, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = M.AdaptiveAvgPool2d(1)
        self.conv_du = M.Sequential(M.Conv2d(in_channels, d, 1, padding=0, bias=bias), M.PReLU())

        self.fcs = M.Sequential(*[M.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias) for _ in range(height)])

        self.softmax = M.Softmax(axis=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = F.concat(inp_feats, axis=1)
        inp_feats = inp_feats.reshape(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = F.sum(inp_feats, axis=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = F.concat(attention_vectors,axis=1)
        attention_vectors = attention_vectors.reshape(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)

        feats_V = F.sum(inp_feats * attention_vectors, axis=1)

        return feats_V

class ContextBlock(M.Module):
    def __init__(self,inplanes,ratio=0.25):
        super(ContextBlock, self).__init__()

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)

        self.conv_mask = M.Conv2d(inplanes, 1, kernel_size=1)
        self.softmax = M.Softmax(axis=2)

        self.channel_add_conv = M.Sequential(
                M.Conv2d(self.inplanes, self.planes, kernel_size=1),
                M.LayerNorm([self.planes, 1, 1]),
                M.ReLU(),  # yapf: disable
                M.Conv2d(self.planes, self.inplanes, kernel_size=1))

    #     self.reset_parameters()
    #
    # def reset_parameters(self):
    #     last_zero_init(self.channel_add_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.shape
        input_x = x
        # [N, C, H * W]
        input_x = input_x.reshape(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = F.expand_dims(input_x,1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.reshape(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = F.expand_dims(context_mask,-1)
        # [N, 1, C, 1]
        context = F.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.reshape(batch, channel, 1, 1)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        out = out + channel_add_term

        return out


class SWRCAB(M.Module):
    def __init__(self, f):
        super(SWRCAB, self).__init__()

        self.convs = M.Sequential(M.Conv2d(f, 1, kernel_size=1, bias=True),M.Sigmoid())
        self.conv1 = M.Sequential(M.Conv2d(f, f, stride=1, kernel_size=3, padding=1, bias=True, groups=f),M.Conv2d(f, f, kernel_size=1, bias=True))
        self.conv2 = M.Sequential(M.Conv2d(f, f, stride=1, kernel_size=3, padding=1, bias=True, groups=f),M.Conv2d(f, f, kernel_size=1, bias=True))

        self.sig = M.Sigmoid()
        # self.fc = M.Conv2d(f, f, kernel_size=1)
        self.fc = M.Sequential(
            M.Conv2d(f, f//4, kernel_size=1),
            M.LayerNorm([f//4, 1, 1]),
            M.ReLU(),  # yapf: disable
            M.Conv2d(f//4, f, kernel_size=1))

        self.relu = M.LeakyReLU(negative_slope=0.2)
        self.gap = M.AdaptiveAvgPool2d(1)

    def forward(self, x):

        out1 = self.relu(self.conv1(x))
        out2 = self.conv2(out1)

        c1 = self.convs(out1)
        c1 = self.gap(c1*out2)
        c1 = self.sig(self.fc(c1))

        return x+c1*out2
class SA(M.Module):
    def __init__(self, n_feats):
        super(SA, self).__init__()
        f = n_feats // 4
        self.conv1 = M.Conv2d(n_feats, f, kernel_size=1)
        self.conv_f = M.Conv2d(f, f, kernel_size=1)
        self.conv_max = M.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv2 = M.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = M.Conv2d(f, f, kernel_size=3, padding=1,groups=f)
        self.conv3_ = M.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = M.Conv2d(f, n_feats, kernel_size=1)
        self.sigmoid = M.Sigmoid()
        self.relu = M.ReLU()
        self.pooling = M.MaxPool2d(kernel_size=7, stride=3)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = self.pooling(c1)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.nn.interpolate(c3, (int(x.shape[2]), int(x.shape[3])), mode='BILINEAR', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m
class U(M.Module):
    def __init__(self,n_feat):
        super(U, self).__init__()
        self.n_feat = n_feat
        self.fe = M.Sequential(
            M.Conv2d(4, self.n_feat, 3, padding=1, bias=True),
            M.LeakyReLU(negative_slope=0.2),
            # ContextBlock(self.n_feat),
            # SA(self.n_feat),
        )
        self.down1 = M.Sequential(
            # M.Conv2d(self.n_feat, self.n_feat, 3, padding=1, bias=True),
            SWRCAB(self.n_feat), M.LeakyReLU(negative_slope=0.2),
            # SWRCAB(self.n_feat),M.LeakyReLU(negative_slope=0.2),

            # ContextBlock(self.n_feat),
            M.MaxPool2d(2, 2),
        )
        self.down2 = M.Sequential(
            # M.Conv2d(self.n_feat, self.n_feat, 3, padding=1, bias=True),
            SWRCAB(self.n_feat), M.LeakyReLU(negative_slope=0.2),
            # SWRCAB(self.n_feat), M.LeakyReLU(negative_slope=0.2),

            # ContextBlock(self.n_feat),
            M.MaxPool2d(2, 2),
        )
        self.down3 = M.Sequential(
            # M.Conv2d(self.n_feat, self.n_feat, 3, padding=1, bias=True),
            SWRCAB(self.n_feat), M.LeakyReLU(negative_slope=0.2),
            # SWRCAB(self.n_feat),M.LeakyReLU(negative_slope=0.2),

            # ContextBlock(self.n_feat),
            M.MaxPool2d(2, 2),
        )

        # self.gc3 = ContextBlock(self.n_feat)
        # self.gc2 = ContextBlock(self.n_feat)
        # # self.gc1 = ContextBlock(self.n_feat)
        # self.skff3 = SKFF(self.n_feat)
        # self.skff2 = SKFF(self.n_feat)
        # self.skff1 = SKFF(self.n_feat)

        self.up3 = M.Sequential(
            M.Conv2d(self.n_feat * 2, self.n_feat, 1),
            M.LeakyReLU(negative_slope=0.2),
            # M.Conv2d(self.n_feat, self.n_feat, 3, padding=1, bias=True),
            # SCBottleneck(self.n_feat,self.n_feat),
            SWRCAB(self.n_feat), M.LeakyReLU(negative_slope=0.2),SWRCAB(self.n_feat), M.LeakyReLU(negative_slope=0.2),
            SWRCAB(self.n_feat), M.LeakyReLU(negative_slope=0.2),
            # SWRCAB(self.n_feat), M.LeakyReLU(negative_slope=0.2),
            SA(self.n_feat),
        )
        self.up2 = M.Sequential(
            M.Conv2d(self.n_feat * 2, self.n_feat, 1),
            M.LeakyReLU(negative_slope=0.2),
            # M.Conv2d(self.n_feat, self.n_feat, 3, padding=1, bias=True),
            # SCBottleneck(self.n_feat, self.n_feat),
            SWRCAB(self.n_feat), M.LeakyReLU(negative_slope=0.2),SWRCAB(self.n_feat), M.LeakyReLU(negative_slope=0.2),
            SWRCAB(self.n_feat), M.LeakyReLU(negative_slope=0.2),
            # SWRCAB(self.n_feat), M.LeakyReLU(negative_slope=0.2),
            SA(self.n_feat),
        )
        self.up1 = M.Sequential(
            M.Conv2d(self.n_feat * 2, self.n_feat, 1),
            M.LeakyReLU(negative_slope=0.2),
            # M.Conv2d(self.n_feat, self.n_feat, 3, padding=1, bias=True),
            # SCBottleneck(self.n_feat, self.n_feat),
            SWRCAB(self.n_feat), M.LeakyReLU(negative_slope=0.2),SWRCAB(self.n_feat), M.LeakyReLU(negative_slope=0.2),
            SWRCAB(self.n_feat), M.LeakyReLU(negative_slope=0.2),
            # SWRCAB(self.n_feat), M.LeakyReLU(negative_slope=0.2),
            SA(self.n_feat),
        )
        self.tail = M.Sequential(
            M.Conv2d(self.n_feat, 4, 3, padding=1, bias=True),
            M.LeakyReLU(negative_slope=0.2),
        )


    def forward(self, x):
        n, c, h, w = x.shape
        res = x
        x = x.reshape((n, c, h // 2, 2, w // 2, 2)).transpose((0, 1, 3, 5, 2, 4)).reshape((n, c * 4, h // 2, w // 2))
        
        x_1 = self.fe(x)
        x_2 = self.down1(x_1)
        x_3 = self.down2(x_2)
        x_4 = self.down3(x_3)


        x_4 = F.nn.interpolate(x_4, (int(x_3.shape[2]), int(x_3.shape[3])), mode='BILINEAR', align_corners=False)
        x_3 = self.up3(F.concat([x_4, x_3], 1))
        x_3 = F.nn.interpolate(x_3, (int(x_2.shape[2]), int(x_2.shape[3])), mode='BILINEAR', align_corners=False)
        x_2 = self.up2(F.concat([x_3, x_2], 1))
        x_2 = F.nn.interpolate(x_2, (int(x_1.shape[2]), int(x_1.shape[3])), mode='BILINEAR', align_corners=False)
        x_1 = self.up1(F.concat([x_2, x_1], 1))
        x = self.tail(x_1)
        x = x.reshape((n, c, 2, 2, h // 2, w // 2)).transpose((0, 1, 4, 2, 5, 3)).reshape((n, c, h, w))
        return x+res

class Predictor(M.Module):

    def __init__(self):
        super(Predictor,self).__init__()

        self.n_feat = 30
        # self.fe = M.Sequential(
        #     M.Conv2d(4, self.n_feat, 3, padding=1, bias=True),
        #     M.LeakyReLU(negative_slope=0.2),
        #     # ContextBlock(self.n_feat),
        #     # SA(self.n_feat),
        # )

        # self.relu =  M.LeakyReLU(0.2)
        self.u1 = U(self.n_feat)
        self.u2 = U(self.n_feat)
        # self.conv = M.Sequential(M.Conv2d(self.n_feat, self.n_feat, 3, padding = 1, bias = True),
        #     M.LeakyReLU(negative_slope = 0.2))
        # self.tail = M.Sequential(
        #     M.Conv2d(self.n_feat, 4, 3, padding = 1, bias = True),
        #     M.LeakyReLU(negative_slope = 0.2),
        # )

    def forward(self, x):
        x1 = self.u1(x)
        x = self.u2(x1)
        return x






def augment(l):
    hflip = random.random()<0.5
    vflip = random.random()<0.5
    rot90 = random.random()<0.5
    def _augment(img):
        if hflip:img = img[::-1,:]
        if vflip:img = img[:,::-1]
        if rot90:img = img.transpose(1,0)
        return img

    # if np.random.randint(2, size=1)[0] == 1:  # random flip
    #     input_patch = np.flip(input_patch, axis=1)
    #     gt_patch = np.flip(gt_patch, axis=1)
    # if np.random.randint(2, size=1)[0] == 1:
    #     input_patch = np.flip(input_patch, axis=2)
    #     gt_patch = np.flip(gt_patch, axis=2)
    # if np.random.randint(2, size=1)[0] == 1:  # random transpose
    #     input_patch = np.transpose(input_patch, (0, 2, 1, 3))
    #     gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))
    return [_augment(_l) for _l in l]
#training -------------------------------------------------------------------------

losses = []



print('training')

loss_log = mge.Tensor([])
score_log = mge.Tensor([])

#net.load_state_dict(mge.load('./model'))
train_start = time.time()
lr = 1e-3
idx_ = np.arange(8000)
best = 0


train_batch = 8000
total_batch = 8192
batchsz = 1 
patchsz = 256
def forward(net):
    samples_pred = np.zeros((192,patchsz,patchsz),dtype='float32')
    for bn in range(train_batch,total_batch):
        batch_inp_np = np.zeros((batchsz, 1, patchsz, patchsz), dtype='float32')
        batch_out_np = np.zeros((batchsz, 1, patchsz, patchsz), dtype='float32')
        for i in range(batchsz):
            batch_inp_np[i, 0, :, :] = np.float32(samples_ref[bn*batchsz+i, :, :]) * np.float32(1 / 65536)
            batch_out_np[i, 0, :, :] = np.float32(samples_gt[bn*batchsz+i, :, :]) * np.float32(1 / 65536)
        batch_inp = mge.tensor(batch_inp_np)
        batch_out = mge.tensor(batch_out_np)
        res = mge.tensor(batch_inp_np)

        #batch_inp = preprocess(batch_inp)
        pred = net(batch_inp)
        #pred = postprocess(pred)

        pred = (pred.numpy()[:, 0, :, :] * 65536).clip(0, 65535).astype('uint16')##.tobytes()
        samples_pred[(bn-train_batch)*batchsz:(bn-train_batch+1)*batchsz,:,:] = pred
    gt = np.float32(samples_gt[batchsz*train_batch:,:,:])
    means = gt.mean(axis=(1, 2))
    weight = (1 / means) ** 0.5
    diff = np.abs(samples_pred - gt).mean(axis=(1, 2))
    diff = diff * weight
    score = diff.mean()
    score = np.log10(100 / score) * 5
    print('score', score)
import tqdm
    
#forward(net1)
#forward(net2)
net = Predictor()
#alpha = 0.8
#net.load_state_dict({
#    k: alpha * v1 + (1 - alpha) * v2 
#    for (k, v1), (_, v2) in zip(net1.state_dict().items(), net2.state_dict().items())
#})
net.load_state_dict(mge.load('./model_f'))
print('ensemble')
#forward(net)
#fout = open('./model_f', 'wb')
#mge.save(net.state_dict(), fout)
#exit(0)
def rotate(x,pred0,net):
    x  = x.transpose((0, 1, 3, 2))
    x_np = np.zeros((batchsz, 1, patchsz, 256), dtype='float32')
    
    x_np[i, 0,  0:254,:] = x[:,:,1:255,:]
    x_np[i, 0, 254,:] = x[:,:,255,:]
    x_np[i, 0,  255,:] = x[:,:,0,:]

    x = mge.tensor(x_np)
    
    pred,x1 = net(x)
    pred = pred[:,:,1:253,:]
    pred = pred.transpose((0,1,3,2))
    pred0[:,:,:,2:254] = (pred0[:,:,:,2:254]+pred)/2
    
    return pred0


def VerticalFlip(x,pred0,net):
    n = x.shape[0]
    x = x[:,:,::-1,:]
    x_np = np.zeros((batchsz, 1, patchsz, 256), dtype='float32')
    
    x_np[i, 0,  0:254,:] = x[:,:,1:255,:]
    x_np[i, 0, 254,:] = x[:,:,255,:]
    x_np[i, 0,  255,:] = x[:,:,0,:]

    x = mge.tensor(x_np)
    
    pred = net(x)
    pred = pred[:,:,1:253,:]
    pred = pred[:,:,::-1,:]
    pred0[:,:,4:252,:] = 0.6*pred0[:,:,4:252,:]+pred[:,:,2:250,:]*0.4#(pred0[:,:,4:252,:]+pred[:,:,2:250,:])/2
    #pred0[:,:,2:254,:] = (pred0[:,:,2:254,:]+pred)/2
    return pred0

def VerticalFli2(x,pred0,net):
    n = x.shape[0]
    x = x[:,:,::-1,:]
    x_np = np.zeros((batchsz, 1, patchsz, 256), dtype='float32')
    
    x_np[i, 0,  0:254,:] = x[:,:,1:255,:]
    x_np[i, 0, 254,:] = x[:,:,255,:]
    x_np[i, 0,  255,:] = x[:,:,0,:]

    x = mge.tensor(x_np)
    
    pred = net(x)
    pred = pred[:,:,1:253,:]
    pred = pred[:,:,::-1,:]
    pred0[:,:,4:252,:] = 0.6*pred0[:,:,4:252,:]+pred[:,:,2:250,:]*0.4#(pred0[:,:,4:252,:]+pred[:,:,2:250,:])/2
    #pred0[:,:,2:254,:] = (pred0[:,:,2:254,:]+pred)/2
    return pred0

def cropFuse(x,pred0,net):
    
    x1 = x[:,:,0:128,0:128]
    x2 = x[:,:,0:128,128:]
    x3 = x[:,:,128:,0:128]
    x4 = x[:,:,128:,128:]
    pred1,xx = net(x1)
    pred2,xx = net(x2)
    pred3,xx = net(x3)
    pred4,xx = net(x4)
       
    pred0[:,:,1:127,1:127] = pred0[:,:,1:127,1:127]*0.8+pred1[:,:,1:127,1:127]*0.2
    
    pred0[:,:,1:127,129:255] = pred0[:,:,1:127,129:255]*0.8+pred2[:,:,1:127,1:127]*0.2
    
    #pred0[:,:,129:255,1:127] = pred0[:,:,129:255,1:127]*0.6+pred3[:,:,1:127,1:127]*0.4
    #pred0[:,:,129:255,129:255] = pred0[:,:,129:255,129:255]*0.6 + pred4[:,:,1:127,1:127]*0.4
    
    return pred0

def padTest(x,pred0,net):
    x_np = np.zeros((batchsz, 1, 260, 260), dtype='float32')
    x_np[:,:,2:258,2:258] = x
    x_np[:,:,0:2,2:258] = x[:,:,0:2,:]
    x_np[:,:,2:258,258:] = x[:,:,:,254:]
    x_np[:,:,2:258,0:2] = x[:,:,:,0:2]
    x_np[:,:,258:,2:258] = x[:,:,254:,:]
    x = mge.tensor(x_np)
    
    pred = net(x)
    pred1 = 0.5*pred0+pred[:,:,2:258,2:258]*0.5
    
    
    return pred1

print('new6')

content = open('../../dataset/burst_raw/competition_test_input.0.2.bin', 'rb').read()
samples_ref = np.frombuffer(content, dtype = 'uint16').reshape((-1,256,256))
fout = open('./result_f.bin', 'wb')
samples_pred = np.zeros((192,patchsz,256),dtype='float32')
for bn in tqdm.tqdm(range(0,1024)):
    batch_inp_np = np.zeros((batchsz, 1, patchsz, 256), dtype='float32')
    batch_inp_np2 = np.zeros((batchsz, 1, patchsz, 256), dtype='float32')
    for i in range(batchsz):
        batch_inp = samples_ref[bn*batchsz+i, :, :]
        #batch_inp = F1.flip(batch_inp,1)
        batch_inp_np2[i, 0, :, :] = np.float32(samples_ref[bn*batchsz+i, :, :]) * np.float32(1 / 65536)
        #batch_inp = batch_inp[:,1:241,:]
        batch_inp = batch_inp[:,::-1]
        batch_inp_np[i, 0, :, 0:254] = np.float32(batch_inp[:,1:255]) * np.float32(1 / 65536)
        batch_inp_np[i, 0, :, 254] = np.float32(batch_inp[:,255]) * np.float32(1 / 65536)
        batch_inp_np[i, 0, :, 255] = np.float32(batch_inp[:,0]) * np.float32(1 / 65536)

    batch_inp = mge.tensor(batch_inp_np)
    batch_inp2 = mge.tensor(batch_inp_np2)
  
    #batch_inp = preprocess(batch_inp)
 
    pred = net(batch_inp)
    pred2 = net(batch_inp2)
    pred4 = pred2
    res = pred2
  
    #pred4[:,:,:,2:254] = pred3[:,:,:,0:252]
    '''
    
 
    pred2 = pred2[:,:,2:254,2:254]
    #pred = postprocess(pred)
    pred1 = pred[:,:,2:254,1:253]
    pred1 = pred1[:,:,:,::-1]
    pred3 = (pred1+pred2)/2
    pred4[:,:,2:254,2:254] = pred3
    
    '''
   

   

    
    
    pred2 = VerticalFlip(batch_inp_np2,pred4,net)
    
    pred22 = res[:,:,:,2:254]
    #pred = postprocess(pred)
    pred1 = pred[:,:,:,1:253]
    pred1 = pred1[:,:,:,::-1]
    pred3 = pred1*0.6+pred22*0.4
    pred4[:,:,:,4:252] = pred3[:,:,:,2:250]*0.6+pred2[:,:,:,4:252]*0.4#(pred3[:,:,:,2:250]+pred2[:,:,:,4:252])/2
#    pred4 = (res+pred4)/2
    

    #pred4 = cropFuse(batch_inp2,pred4,net)
    pred_p = padTest(batch_inp_np2,res,net)
    #pred4[:,:,2:254,2:254] = (pred4[:,:,2:254,2:254]+pred_p[:,:,2:254,2:254])/2
    
    pred4 = (pred4+pred_p)/2
    #pred4 = pred4*1+pred_both*0
    #predm= mix([pred4,pred_both,res,pred_p])
    #predm[:,:,2:254.2:254] = pred4[:,:,2:254.2:254] 
    #pred4 = predm
    pred4 = (pred4.numpy()[:, 0, :, :] * 65536).clip(0, 65535).astype('uint16')##.tobytes()
    

    samples_pred[(bn-8000)*batchsz:(bn-8000+1)*batchsz,:,:] = pred4
   
    fout.write(pred4.tobytes())
fout.close()
#gg = samples_gt[batchsz*8000:,:,:].reshape((256,256,-1))
#gg = gg[:,1:241,:].reshape((-1,256,240))
#gt = np.float32(gg)
exit(0)

