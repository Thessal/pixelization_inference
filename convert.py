# +
# # !wget "https://drive.google.com/u/3/uc?id=1VRYKQOsNlE1w1LXje3yTRU5THN2MGdMM&export=download&confirm=yes" -O ./pixelart_vgg19.pth
# # !wget "https://drive.google.com/u/3/uc?id=17f2rKnZOpnO9ATwRXgqLz5u5AZsyDvq_&export=download&confirm=yes" -O ./alias_net.pth
# # !wget "https://drive.google.com/u/3/uc?id=1i_8xL3stbLWNF4kdQJ50ZhnRFhSDh3Az&export=download&confirm=yes" -O ./160_net_G_A.pth

# torch.__version__ = '1.13.1+cu117'
# numpy.__version__ = '1.24.1'
# PIL.__version__ = '9.1.0'
# onnxruntime.__version__ = '1.13.1'
# -

import torch
import pixelization
import numpy as np
from PIL import Image
import onnxruntime as ort
device='cuda'
m = pixelization.Model(device=device)
m.device=torch.device(device)
m.load()

max_size = 1000

# +
in_img = Image.fromarray(np.zeros((max_size,max_size,3),dtype=np.int32), mode="RGB")
in_t = pixelization.process(in_img).to(device)

ref_img = Image.open("reference.png").convert('L')
ref_t = pixelization.process(pixelization.greyscale(ref_img)).to(device)

dummy_input = (in_t, ref_t)
dummy_input_2 = m.G_A_net(in_t, ref_t)

# +
siz = max_size//4
alias_net = m.alias_net
G_A_net = m.G_A_net
torch.onnx.export(alias_net.RGBEnc, dummy_input_2, "alias_RGBEnc.onnx", opset_version=10, verbose=True, input_names=['x'], output_names=['x'])
torch.onnx.export(alias_net.RGBDec, alias_net.RGBEnc(dummy_input_2), "alias_RGBDec.onnx", opset_version=10, verbose=True, input_names=['x'], output_names=['x'])              
torch.onnx.export(G_A_net.RGBEnc, dummy_input[0], "g_a_RGBEnc.onnx", opset_version=10, verbose=True, input_names=['clipart'], output_names=["feature"])
torch.onnx.export(G_A_net.PBEnc, dummy_input[1], "g_a_PBEnc.onnx", opset_version=10, verbose=True, input_names=['pixelart'],  output_names=["code"])
torch.onnx.export(G_A_net.MLP, G_A_net.PBEnc(dummy_input[1]), "g_a_MLP.onnx", opset_version=10, verbose=True, input_names=["style0"], output_names=["adain_params"])
torch.onnx.export(
    G_A_net.RGBDec.mod_conv_1,
    (torch.tensor(np.ones((1,256,siz,siz)),dtype=torch.float).cuda(),torch.tensor(np.ones((1,256)),dtype=torch.float).cuda()),
    "g_a_RGBDec_mod_conv_1.onnx",
    verbose=True, opset_version=10, 
    input_names=['x1', 'x2'],
#     dynamic_axes={'x1':[2,3]}
)
torch.onnx.export(G_A_net.RGBDec.mod_conv_2,(torch.tensor(np.ones((1,256,siz,siz)),dtype=torch.float).cuda(),torch.tensor(np.ones((1,256)),dtype=torch.float).cuda()),
    "g_a_RGBDec_mod_conv_2.onnx",verbose=True, opset_version=10,input_names=['x1', 'x2'],)
torch.onnx.export(G_A_net.RGBDec.mod_conv_3,(torch.tensor(np.ones((1,256,siz,siz)),dtype=torch.float).cuda(),torch.tensor(np.ones((1,256)),dtype=torch.float).cuda()),
    "g_a_RGBDec_mod_conv_3.onnx",verbose=True, opset_version=10,input_names=['x1', 'x2'],)
torch.onnx.export(G_A_net.RGBDec.mod_conv_4,(torch.tensor(np.ones((1,256,siz,siz)),dtype=torch.float).cuda(),torch.tensor(np.ones((1,256)),dtype=torch.float).cuda()),
    "g_a_RGBDec_mod_conv_4.onnx",verbose=True, opset_version=10,input_names=['x1', 'x2'],)
torch.onnx.export(G_A_net.RGBDec.mod_conv_5,(torch.tensor(np.ones((1,256,siz,siz)),dtype=torch.float).cuda(),torch.tensor(np.ones((1,256)),dtype=torch.float).cuda()),
    "g_a_RGBDec_mod_conv_5.onnx",verbose=True, opset_version=10,input_names=['x1', 'x2'],)
torch.onnx.export(G_A_net.RGBDec.mod_conv_6,(torch.tensor(np.ones((1,256,siz,siz)),dtype=torch.float).cuda(),torch.tensor(np.ones((1,256)),dtype=torch.float).cuda()),
    "g_a_RGBDec_mod_conv_6.onnx",verbose=True, opset_version=10,input_names=['x1', 'x2'],)
torch.onnx.export(G_A_net.RGBDec.mod_conv_7,(torch.tensor(np.ones((1,256,siz,siz)),dtype=torch.float).cuda(),torch.tensor(np.ones((1,256)),dtype=torch.float).cuda()),
    "g_a_RGBDec_mod_conv_7.onnx",verbose=True, opset_version=10,input_names=['x1', 'x2'],)
torch.onnx.export(G_A_net.RGBDec.mod_conv_8,(torch.tensor(np.ones((1,256,siz,siz)),dtype=torch.float).cuda(),torch.tensor(np.ones((1,256)),dtype=torch.float).cuda()),
    "g_a_RGBDec_mod_conv_8.onnx",verbose=True, opset_version=10,input_names=['x1', 'x2'],)
torch.onnx.export(
    G_A_net.RGBDec.upsample_block1,
    (torch.tensor(np.ones((1,256,siz,siz)), dtype=torch.float).cuda()),
    "g_a_RGBDec_upsample_block1.onnx",
    verbose=True, opset_version=11,input_names=['x1'],)
torch.onnx.export(
    G_A_net.RGBDec.upsample_block2,
    (torch.tensor(np.ones((1,128,siz*2,siz*2)), dtype=torch.float).cuda()),
    "g_a_RGBDec_upsample_block2.onnx",
    verbose=True, opset_version=11,input_names=['x1'],)

torch.onnx.export(
    G_A_net.RGBDec.conv_1,
    (torch.tensor(np.ones((1,256, siz*2, siz*2)), dtype=torch.float).cuda()),
    "g_a_RGBDec_conv_1.onnx",
    verbose=True, opset_version=11,input_names=['x1'],)
torch.onnx.export(
    G_A_net.RGBDec.conv_2,
    (torch.tensor(np.ones((1,128, siz*4, siz*4)), dtype=torch.float).cuda()),
    "g_a_RGBDec_conv_2.onnx",
    verbose=True, opset_version=11,input_names=['x1'],)
torch.onnx.export(
    G_A_net.RGBDec.conv_3,
    (torch.tensor(np.ones((1,64,  siz*4, siz*4)), dtype=torch.float).cuda()),
    "g_a_RGBDec_conv_3.onnx",
    verbose=True, opset_version=11,input_names=['x1'],)


# +
def greyscale(img):
    gray = np.array(img.convert('L'))
    tmp = np.expand_dims(gray, axis=2)
    tmp = np.concatenate((tmp, tmp, tmp), axis=-1)
    return Image.fromarray(tmp)


def process(img):
    ow,oh = img.size

    nw = int(round(ow / 4) * 4)
    nh = int(round(oh / 4) * 4)

    left = (ow - nw)//2
    top = (oh - nh)//2
    right = left + nw
    bottom = top + nh

    img = img.crop((left, top, right, bottom))
    # Normalize
    img = np.array( (img - np.mean(img, axis=(0,1))) / np.std(img, axis=(0,1)), dtype=np.float32)
    img = np.swapaxes(img, 0, -1)
    img = np.swapaxes(img, 1, -1)
    return img[np.newaxis, :, :, :]

def pixelize(in_img):
    # data load
    in_t = process(in_img)
    ref_img = Image.open("reference.png").convert('L')
    ref_t = process(greyscale(ref_img))

    # model load
    alias_RGBEnc = ort.InferenceSession("alias_RGBEnc.onnx")
    alias_RGBDec = ort.InferenceSession("alias_RGBDec.onnx")
    g_a_RGBEnc = ort.InferenceSession("g_a_RGBEnc.onnx")
    g_a_PBEnc = ort.InferenceSession("g_a_PBEnc.onnx")
    g_a_MLP = ort.InferenceSession("g_a_MLP.onnx")
    g_a_RGBDec_mod_conv_1 = ort.InferenceSession("g_a_RGBDec_mod_conv_1.onnx")
    g_a_RGBDec_mod_conv_2 = ort.InferenceSession("g_a_RGBDec_mod_conv_2.onnx")
    g_a_RGBDec_mod_conv_3 = ort.InferenceSession("g_a_RGBDec_mod_conv_3.onnx")
    g_a_RGBDec_mod_conv_4 = ort.InferenceSession("g_a_RGBDec_mod_conv_4.onnx")
    g_a_RGBDec_mod_conv_5 = ort.InferenceSession("g_a_RGBDec_mod_conv_5.onnx")
    g_a_RGBDec_mod_conv_6 = ort.InferenceSession("g_a_RGBDec_mod_conv_6.onnx")
    g_a_RGBDec_mod_conv_7 = ort.InferenceSession("g_a_RGBDec_mod_conv_7.onnx")
    g_a_RGBDec_mod_conv_8 = ort.InferenceSession("g_a_RGBDec_mod_conv_8.onnx")
    g_a_RGBDec_upsample_block1 = ort.InferenceSession("g_a_RGBDec_upsample_block1.onnx")
    g_a_RGBDec_upsample_block2 = ort.InferenceSession("g_a_RGBDec_upsample_block2.onnx")
    g_a_RGBDec_conv_1 = ort.InferenceSession("g_a_RGBDec_conv_1.onnx")
    g_a_RGBDec_conv_2 = ort.InferenceSession("g_a_RGBDec_conv_2.onnx")
    g_a_RGBDec_conv_3 = ort.InferenceSession("g_a_RGBDec_conv_3.onnx")
    
    def g_a_RGBDec(x, code):
            residual = x
            x = g_a_RGBDec_mod_conv_1.run(None,{"x1":x, "x2":code[:,:256]})[0]
            x = g_a_RGBDec_mod_conv_2.run(None,{"x1":x, "x2":code[:, 256*1:256*2]})[0]
            x += residual
            residual = x
            x = g_a_RGBDec_mod_conv_2.run(None,{"x1":x, "x2":code[:, 256*2:256*3]})[0]
            x = g_a_RGBDec_mod_conv_2.run(None,{"x1":x, "x2":code[:, 256*3:256*4]})[0]
            x += residual
            residual =x
            x = g_a_RGBDec_mod_conv_2.run(None,{"x1":x, "x2":code[:, 256*4:256*5]})[0]
            x = g_a_RGBDec_mod_conv_2.run(None,{"x1":x, "x2":code[:, 256*5:256*6]})[0]
            x += residual
            residual = x
            x = g_a_RGBDec_mod_conv_2.run(None,{"x1":x, "x2":code[:, 256*6:256*7]})[0]
            x = g_a_RGBDec_mod_conv_2.run(None,{"x1":x, "x2":code[:, 256*7:256*8]})[0]
            x += residual
            x = g_a_RGBDec_upsample_block1.run(None,{"x1":x})[0]
            x = g_a_RGBDec_conv_1.run(None,{"x1":x})[0]
            x = g_a_RGBDec_upsample_block2.run(None,{"x1":x})[0]
            x = g_a_RGBDec_conv_2.run(None,{"x1":x})[0]
            x = g_a_RGBDec_conv_3.run(None,{"x1":x})[0]
            return x

    def g_a(clipart, pixelart):
        feature = g_a_RGBEnc.run(None,{"clipart":clipart})[0]
        code = g_a_PBEnc.run(None,{"pixelart":pixelart})[0]
        adain_params = g_a_MLP.run(None,{"style0":code})[0]
        images = g_a_RGBDec(feature, adain_params)
        return images, adain_params
    
    images, _ = g_a(in_t, ref_t)
    
    images = alias_RGBEnc.run(None,{"x.1":images})[0]
    images = alias_RGBDec.run(None,{"x.1":images})[0]
    
    img = images[0]
    img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img = img.resize((img.size[0]//4, img.size[1]//4), resample=Image.Resampling.NEAREST)
    img = img.resize((img.size[0]*4, img.size[1]*4), resample=Image.Resampling.NEAREST)
    return img


pixelize(Image.open("input_file.jpg").convert('RGB').resize((1000,1000)))
# -


