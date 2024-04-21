# genefacepp
### 记录学习geneface++所遇到的各种问题
<br>
欢迎大家互相交流学习沟通 作者邮箱417074596@qq.com
<br>
数据预处理部分，裁剪视频，附上了代码在cropvideo.py中，可以将视频裁剪为512*512大小，音频采样为16000hz
<br>

### ffmpeg的一些参数调整

```
cmd =[
    "ffmpeg",
    "-i", frame_pattern,
    "-i", f"{audio_output}/{name}_audio.wav",
    "-c:v", "libx264",
    "-crf", "23",  # 添加CRF值
    "-preset","veryslow",
    "-framerate", str(targetFps),
    '-pix_fmt', 'yuv444p',
    "-y", f"{outputFolder}/{name}_face_crop.mp4"
]
```
### 参数解释
<br>
1.CRF (恒定速率因子): 通过 -crf 参数设置，用于控制输出视频的质量。CRF 值越低，质量越高，文件大小越大。对于 libx264 和 libx265 编码器，CRF 值通常在 18 到 28 之间，其中 18 被认为是视觉上无损的。
<br>
2.预设 (Preset): -preset 参数用于平衡编码速度和压缩率（从而影响质量和文件大小）。预设值越慢，编码过程越长，但可以获得更好的压缩效率和质量。常见的预设包括 ultrafast, superfast, veryfast, faster, fast, medium（默认值）, slow, slower, 和 veryslow。
<br>
3.比特率 (Bitrate): -b:v 参数用于指定视频的比特率，即每秒传输的数据量。增加比特率可以提高视频质量，但也会增加文件大小。适用于需要固定比特率输出的场景。
<br>
4.最大比特率和缓冲区大小: 使用 -maxrate 和 -bufsize 参数可以在使用 VBR（可变比特率）时限制最高比特率，这有助于控制视频质量和文件大小。
<br>
5.像素格式 (Pixel Format): -pix_fmt 参数用于指定像素格式，如 yuv420p（大多数情况下的默认值）, yuv422p, yuv444p 等。使用高质量的像素格式可以提高视频质量，但可能会增加文件大小。
‘-pix_fmt’:
yuv420p: 这是最常用的像素格式之一，特别是对于H.264编码。它使用4:2:0色度子采样，与大多数设备和播放器兼容。
yuv422p: 相比于 yuv420p，它提供了更好的色度分辨率，使用4:2:2色度子采样。
yuv444p: 提供未经子采样的色度信息，使用4:4:4色度子采样，保留了更多的色彩信息，但文件大小会更大。
rgb24: 一个基于RGB的像素格式，每个颜色通道（红、绿、蓝）各占8位，没有色度子采样。
rgba: 类似于 rgb24，但增加了一个8位的透明度通道。
总的来说，如果您需要最高的色彩保真度且不需要透明通道，yuv444p是一个非常好的选择。如果您的视频需要透明效果，那么rgba将是必要的选择。在不需要透明度的情况下，使用rgba可能会导致不必要的文件大小增加，而没有实质性的质量提升。
<br>
6.帧率 (Frame Rate): -r 参数用于设置视频的帧率。较高的帧率可以使视频播放更加平滑，但会增加文件大小。
<br>
7.分辨率 (Resolution): 通过调整输出视频的分辨率，可以直接影响视频的清晰度和文件大小。使用 -s 参数或通过过滤器来调整分辨率。

### pr.1
<br>
运行python data_gen/utils/process_video/extract_segment_imgs.py --ds_name=nerf --vid_dir=data/raw/videos/${VIDEO_ID}.mp4 # extract image, segmap, and background
<br>
发现处理的很慢，根本就无法提取背景，一直显示0%。后来debug下去发现是多进程的时候，会出现队列为空的现象。由于本人菜鸡，不知道为什么，但是可以绕开这个问题，就取消多进程。处理的方法就是在extract_segment_imgs.py这个文件中，将multiprogrocess_enable 设置为false，即可运行。

### pr.2
<br>
还有一点，在计算pid的时候，可能会计算失败，所以直接将cuda_id设置为0或其他。
<br>

### pr.3
<br>
在训练head时也报错
<br>
```
size mismatch for blink_encoder.1.weight: copying a param with shape torch.Size(82, 32]) from checkpoint, the shape in current model is torch.Size([4, 32]). size mismatch for blink_encoder.1.bias: copying a param with shape torch.Size([2]) from checkpoint, the shape in current model is torch.Size([8]). Terminated
```
这个时候调整load_strict=False失败了，理由是匹配上了，但是大小不相同无法用这个解决。解决办法第一个是删除不匹配的层，或者直接跳过，我是采用跳过的
可以参考博客
修改/utils/commons/trainer.py 

```
for k, v in checkpoint['state_dict'].items():
            if k in ['blink_encoder.1.weight', 'blink_encoder.1.bias']:
                print(f"| Skipping loading of {k} due to size mismatch.")
                continue  # Skip loading this parameter
            if hasattr(task_ref, k):
                getattr(task_ref, k).load_state_dict(v, strict=False)
            else:
                print(f"| the checkpoint has unmatched keys {k}")
                
```
<br>
可以参考链接：
[网页链接]（https://blog.csdn.net/hxxjxw/article/details/119491163）

### pr.4
在训练torso的时候报错
```
RuntimeError: Error(s) in loading state_dict for RADNeRFwithSR: size mismatch for blink_encoder.1.weight: copying a param with shape torch.Size([2, 32]) from checkpoint, the shape in current model is torch.Size([4, 32]). size mismatch for blink_encoder.1.bias: copying a param with shape torch.Size([2]) from checkpoint, the shape in current model is torch.Size([4]). Terminated
```
修改/tasks/radnerfs/radnerf_torso_sr.py 下面的RADNeRFTorsoTask类中的load_ckpt(head_model, hparams['head_model_dir'], strict=False) 将strict改为False



### 推理是的ui界面的几个参数

<br>
blinkmode 是否眨眼
<br>
lle-percent:根据论文第二部分所说的lle参数，值越大说明越趋于大大量训练数据，若设置为0则可能出现异常值，抖动之类的
<br>
mouth amplitude：越大嘴巴张开越大
<br>
tempearature: 这个本人菜菜，猜测应该是autio转换以后的词向量维度相关的参数
<br>
ray marching end-threshold:这个就是加速了

### 再次训练自己的调参记录
