GitHub源代码地址：https://github.com/fregu856/deeplabv3

由于文件上传的限制，文件夹**pretrained_models**中的预训练好的模型传至百度网盘，百度网盘链接为

链接：https://pan.baidu.com/s/1_rwuTskB9rkVXB82A1dCKA?pwd=dp8q 
提取码：dp8q

源代码使用的是pytorch0.4，我使用的是pytorch1.10，会报某个函数版本警告问题，但是不会影响最后的测试过程。

由于在Windows系统下运行代码，对源代码进行了一定的修改才能在我本地的电脑上运行

具体修改内容如下：

#####  修改model文件夹下的deeplabv3.py的第9、10行

```python
from model.resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8`

from model.aspp import ASPP, ASPP_Bottleneck
```

在resnet和aspp前加上model.

##### 修改model文件夹下的resnet.py第179和189行

```python
resnet.load_state_dict(torch.load("D:/Code/python/deeplabv3/pretrained_models/resnet/resnet18-5c106cde.pth"))
```

文件路径改为绝对路径。

测试使用文件为**visualization**文件夹下的run\_on\_seq.py。使用前需要修改以下内容：

在第30行加入以下内容，并注意后面的代码需要缩进

```python
if __name__ == "__main__":
    batch_size = 2
```

并且sys.path.append的路径全部改成绝对路径，在之后的34，35行做如下修改：

```python
network = DeepLabV3("eval_seq", project_dir="D:/Code/python/deeplabv3").cuda()
network.load_state_dict(torch.load("D:/Code/python/deeplabv3/pretrained_models/model_13_2_2_2_epoch_580.pth"))
```

即全部改成绝对路径。

第40行做如下修改：

```python
val_dataset = DatasetSeq(cityscapes_data_path="D:/Code/python/deeplabv3/data/cityscapes",
                         cityscapes_meta_path="D:/Code/python/deeplabv3/data/cityscapes",
                         sequence=sequence)
```

即也修改为接下来要放入测试序列图片的绝对路径。

接下来在上述绝对路径下新建文件夹**leftImg8bit**，再在这个文件夹下新建文件夹**demoVideo**，再在**demoVideo**文件夹下新建三个文件夹**stuttgart_00**、**stuttgart_01**和**stuttgart_02**。最后将要测试的序列图片放入**stuttgart_00**文件夹即可。

之后就可以运行run_on_seq.py测试了，运行结果保存在**deeplabv3\training_logs\model_eval_seq**文件夹下。

本实验测试用例来自YouTube视频https://www.youtube.com/watch?v=fkps18H3SXY&ab_channel=4KUrbanLife，并且用PR软件截取了部分片段并转换成了帧图片序列进行测试。

最后产生的测试结果视频上传至了百度网盘：

链接：https://pan.baidu.com/s/1iFicpKryWPrIDO4RrNIbmg?pwd=7ikk 
提取码：7ikk

