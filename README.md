## Inference script for [Pixelization](https://github.com/WuZongWei6/Pixelization) in Rust
All credit to those guys, I just conveted the model to make it simple to use.

## Usage
```
git clone https://github.com/arenatemp/pixelization_inference
pip install pillow torch torchvision numpy
```
Download the pretrained models into the pixelization_inference folder:
[pixelart_vgg19.pth](https://drive.google.com/file/d/1VRYKQOsNlE1w1LXje3yTRU5THN2MGdMM/view?usp=sharing)
[alias_net.pth](https://drive.google.com/file/d/17f2rKnZOpnO9ATwRXgqLz5u5AZsyDvq_/view?usp=sharing)
[160_net_G_A.pth](https://drive.google.com/file/d/1i_8xL3stbLWNF4kdQJ50ZhnRFhSDh3Az/view?usp=sharing)
```
python convert.py
cd pixelize-rust
cargo run [your-image-file]
```

## Windows build
Download pixelize-rust.zip from above.
or download from [ipfs](https://cloudflare-ipfs.com/ipfs/QmXYtF75yWRKXXy6jibDTvT8QXnEtPpqd6y9KXmyF9wuhM?__cf_chl_tk=iPJZMXZgU51bp7POumrfbNPsAI.P8X.aqpedIJt54QY-1675280415-0-gaNycGzNByU)
```
unzip microsoft.ml.onnxruntime.1.8.1.nupkg and copy onnxruntime.dll
install vc_redist.x64.exe
cargo run --target x86_64-pc-windows-msvc [your-image-file]
```
## Limiation
Max image size = 1000 x 1000 px
