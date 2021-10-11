1. RePOSE: Fast 6D Object Pose Refinement via Deep Texture Rendering
2. DONet: Learning Category-Level 6D Object Pose and Size Estimation from Depth Observation
3. Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation
4. Model Based Training, Detection and Pose Estimation of Texture-Less 3D Objects in Heavily Cluttered Scenes
5. GDR-Net: Geometry-Guided Direct Regression Network for Monocular 6D Object Pose Estimation

## 论文一

### 1. 论文信息

1. 题目：RePOSE: Fast 6D Object Pose Refinement via Deep Texture Rendering

2. 数据集主页：https://paperswithcode.com/dataset/ycb-video

3. 数据集下载：https://rse-lab.cs.washington.edu/projects/posecnn/

   - 代码和数据集

     [姿势CNN（github）](https://github.com/yuxng/PoseCNN)

     [YCB 视频数据集 ~ 265G](https://drive.google.com/file/d/1if4VoEXNx9W3XCn0Y7Fp15B4GpcYbyYi/view?usp=sharing)

     [YCB-Video 3D 模型 ~ 367M](https://drive.google.com/file/d/1gmcDD-5bkJfcMKLZb3zGgH_HUFbulQWu/view?usp=sharing)

     [YCB 视频数据集工具箱 (github)](https://github.com/yuxng/YCB_Video_toolbox)

4. PoseCNN：用于在杂乱场景中进行 6D 对象姿态估计的卷积神经网络（PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes）

### 2. 论文摘要

摘要

我们提出了RePOSE，一种快速迭代细化方法用于6D物体姿态估计。之前的方法是通过输入放大的输入数据和渲染的RGB数据来进行细化的通过将放大的输入和渲染的RGB图像输入到CNN中，并直接回归到一个更新的细化的姿势。由于CNN的计算成本，它们的运行时间慢。他们的运行时间很慢，因为CNN的计算成本很高，这在多物体姿势精炼中尤为突出。多物体姿势精炼中尤为突出。为了克服这个问题。RePOSE利用图像渲染进行快速特征提取使用一个具有可学习纹理的三维模型。我们称之为这就是深度纹理渲染，它使用一个浅层多层感受器来直接回归一个物体的视图不变量图像。物体的代表。此外，我们利用可微分的Levenberg-Marquardt (LM)优化，通过最小化姿势，快速准确地完善通过最小化输入和呈现的图像表示之间的距离，快速而准确地完善姿势。输入和渲染的图像表示之间的距离最小，而不需要放大。这些图像表征训练，使可微调的LM优化在几次迭代中收敛。因此，RePOSE的运行速度为以92 FPS的速度运行，并在闭塞线数据集上实现了最先进的51:6%的准确率。在Occlusion LineMOD数据集上实现了最先进的51:6%的准确率--与现有技术相比有4:1%的绝对改进。与现有技术相比，在YCB-Video数据集上取得了类似的结果。在YCB-Video数据集上的结果也相当，而且运行时间更快。该代码是可在https://github.com/sh8/repose。



### 3. 摘要PPT

主题：论文提出了一种用于6D目标位姿估计的快速迭代求精方法- RePOSE。

研究现状：之前的方法是通过将放大的输入和呈现的RGB图像输入到CNN中，并直接回归到一个”经过优化的姿态的更新“来进行优化。由于CNN的计算成本，它们的运行时间较慢，这在多目标姿态求解中问题尤为突出。

创新点：为了克服这个问题，RePOSE利用图像渲染，使用带有可学习纹理的3D模型快速提取特征。我们称之为深度纹理渲染，它使用一个浅层的多层感知器来直接回归对象的视图不变图像表示。

优化：此外，该论文利用可微分Levenberg-Marquardt (LM)优化来优化在不需要放大的情况下，最小化输入和渲染图像之间的距离，快速而准确地实现姿态。这些图像表示经过训练，使得可微分LM优化在几次迭代内收敛。

实验结果：因此，在the Occlusion LineMOD数据集上，RePOSE以92 FPS的速度运行，达到了51.6%的最先进的精度——比现有技术绝对提高了4.1%，并且在YCB-Video数据集上的运行速度更快。代码可在https://github.com/sh8/repose上获得。

## 论文二：

### 1. 论文信息

1. 题目：DONet: Learning Category-Level 6D Object Pose and Size Estimation from Depth Observation
2. 数据集：
   1. NOCS dataset（He Wang, Srinath Sridhar, Jingwei Huang, Julien Valentin,Shuran Song, and Leonidas J Guibas. Normalized object coordinate space for category-level 6d object pose and size estimation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2642–2651, 2019. 1,2, 3, 5, 6, 7）
   2. NOCS-REAL275 datasets
   3. LineMOD（Stefan Hinterstoisser, Vincent Lepetit, Slobodan Ilic, Stefan Holzer, Gary Bradski, Kurt Konolige, and Nassir Navab. Model based training, detection and pose estimation of texture-less 3d objects in heavily cluttered scenes. In Asianconference on computer vision, pages 548–562. Springer,2, 7）

### 2. 论文摘要

我们提出了一种基于单一深度图像的类别级6D目标姿态和尺寸估计(COPSE)方法，不需要外部姿态标注的真实世界训练数据。以往的工作[43,39,4]利用RGB(D)图像中的视觉线索，而我们的方法仅基于深度通道中物体丰富的几何信息进行推理。从本质上讲，我们的框架通过学习统一的3D方向一致（Orientation-Consistent ）表示(3D- ocr)模块来探索这些几何信息，并进一步通过几何约束反射对称(GeoReS)模块的性质来加强。最后通过镜像成对尺寸估计(MPDE)模块估计出目标尺寸和中心点的大小信息。在类别级别的NOCS基准上的大量实验表明，我们的框架可以与需要有标签的真实世界图像的最先进的方法竞争。我们还将我们的方法应用到一个物理Baxter机器人上，在看不见但类别已知的实例上执行操作任务，结果进一步验证了我们提出的模型的有效性。

### 3. 摘要PPT

主题：该论文提出了一种基于单一深度图像的类别级6D目标姿态和尺寸估计(COPSE)方法，不需要外部姿态标注的真实世界训练数据。

研究现状：以往的工作利用RGB(D)图像中的视觉线索，而我们的方法仅基于深度通道中物体丰富的几何信息进行推理。

创新点：从本质上讲，我们的框架通过学习统一的3D方向一致表示(3D-OCR)模块来探索这些几何信息，并进一步通过几何约束反射对称(GeoReS)模块的性质来加强。最后通过镜像成对尺寸估计(MPDE)模块估计出目标尺寸和中心点的大小信息。

实验结果：在类别级别的NOCS基准上的大量实验表明，我们的框架可以与需要有标签的真实世界图像的最先进的方法竞争。我们还将我们的方法应用到一个物理Baxter机器人上，在看不见但类别已知的实例上执行操作任务，结果进一步验证了我们提出的模型的有效性。

## 论文三

### 1. 论文信息

1. 题目：Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation

2. 发表：[CVPR 2019](http://cvpr2019.thecvf.com/)

3. GitHub地址：https://github.com/hughw19/NOCS_CVPR2019

4. 项目主页：https://geometry.stanford.edu/projects/NOCS_CVPR2019/

5. 数据集：

   1. CAMERA 数据集：[训练](http://download.cs.stanford.edu/orion/nocs/camera_train.zip)/[测试](http://download.cs.stanford.edu/orion/nocs/camera_val25K.zip)/ [IKEA_backgrounds](http://download.cs.stanford.edu/orion/nocs/ikea_data.zip) / [Composed_depths](http://download.cs.stanford.edu/orion/nocs/camera_composed_depth.zip)

   ```
   +合成深度图像包含所有训练和验证数据的前景合成对象和背景真实场景的深度
   ```

   - 真实数据集：[训练](http://download.cs.stanford.edu/orion/nocs/real_train.zip)/[测试](http://download.cs.stanford.edu/orion/nocs/real_test.zip)
   - 真实姿势注释（为了更容易评估）：[Val&Real_test](http://download.cs.stanford.edu/orion/nocs/gts.zip)
   - [对象网格](http://download.cs.stanford.edu/orion/nocs/obj_models.zip)

### 2. 论文摘要

本文的目标是估计RGB-D图像中不可见物体实例的6D位姿和维数。与“实例级”6D提出的估计任务相反，我们的问题假设在训练或测试期间没有确切的对象CAD模型可用。为了处理给定类别中不同且不可见的对象实例，我们引入了规范化对象坐标空间(NOCS)——一个类别中所有可能的对象实例的共享规范表示。然后，我们的基于区域的神经网络被训练来直接推断从观察像素到共享对象表示(NOCS)的对应关系，以及其他对象信息，如类标签和实例掩码。这些预测可以结合深度图来联合估计一个杂乱场景中多个物体的6D姿态和尺寸。为了训练我们的网络，我们提出了一种新的上下文感知技术来生成大量完全注释的混合现实数据。为了进一步改进我们的模型并评估其在真实数据上的性能，我们还提供了一个具有大环境和实例变化的完整注释的真实数据集。大量的实验表明，该方法能够稳健地估计真实环境中不可见物体实例的位姿和大小，同时在标准的6D位姿估计基准上也取得了最先进的性能。

### 3. 摘要PPT

主题：本文的目标是估计RGB-D图像中不可见物体实例的6D位姿和维数。

对比：与“实例级”6D提出的估计任务相反，我们的问题假设在训练或测试期间没有确切的对象CAD模型可用。

创新点：为了处理给定类别中不同且不可见的对象实例，我们引入了规范化对象坐标空间(NOCS)——一个类别中所有可能的对象实例的共享规范表示。然后，我们的基于区域的神经网络被训练来直接推断从观察像素到共享对象表示(NOCS)的对应关系，以及其他对象信息，如类标签和实例掩码。这些预测可以结合深度图来联合估计一个杂乱场景中多个物体的6D姿态和尺寸。

数据：为了训练我们的网络，我们提出了一种新的上下文感知技术来生成大量完全注释的混合现实数据。为了进一步改进我们的模型并评估其在真实数据上的性能，我们还提供了一个具有大环境和实例变化的完整注释的真实数据集。

实验结果：大量的实验表明，该方法能够稳健地估计真实环境中不可见物体实例的位姿和大小，同时在标准的6D位姿估计基准上也取得了最先进的性能。

## 论文四

### 1. 论文信息

1. 题目：Model Based Training, Detection and Pose Estimation of Texture-Less 3D Objects in Heavily Cluttered Scenes
2. 发表：2013年，经典
   1. 数据集：The datasets is public available
      at http://campar.in.tum.de/twiki/pub/Main/StefanHinterstoisser.

### 2. 论文摘要

摘要我们提出了一个使用Kinect自动建模、检测和跟踪3D物体的框架。检测部分主要基于最近的基于模板的LINEMOD方法[1]进行目标检测。我们展示了如何从3D模型自动建立模板，以及如何准确和实时估计6个自由度的姿态。姿态估计和颜色信息使我们能够检查检测假设，相对于原始LINEMOD提高了13%的正确检测率。这些改进使我们的框架适合于机器人应用程序中的对象操作。此外，我们提出了一个新的数据集，该数据集由15个不同对象的15个1100多帧视频序列组成，用于评估未来的竞争方法

### 3. 摘要PPT

主题：该论文提出了一个使用Kinect自动建模、检测和跟踪3D物体的框架。

创新点：检测部分主要基于最近的基于模板的LINEMOD方法进行目标检测。我们展示了如何从3D模型自动建立模板，以及如何准确和实时估计6个自由度的姿态。

结果：姿态估计和颜色信息使我们能够检查检测假设，相对于原始LINEMOD提高了13%的正确检测率。这些改进使我们的框架适合于机器人应用程序中的对象操作。

数据：此外，我们提出了一个新的数据集，该数据集由15个不同对象的15个1100多帧视频序列组成，用于评估未来的竞争方法

## 论文五

### 1. 论文信息

1. 题目：GDR-Net: Geometry-Guided Direct Regression Network for Monocular 6D Object Pose Estimation
2. 数据集：LM, LM-O and YCB-V datasets.
   1. 从[BOP 网站](https://bop.felk.cvut.cz/datasets/)和 [VOC 2012](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)下载 6D 姿势数据集（LM、LM-O、YCB-V） 作为背景图像。也请从这里下载`image_sets`和`test_bboxes`（[百度网盘](https://pan.baidu.com/s/1gGoZGkuMYxhU9LBKxuSz0g)，[OneDrive](https://1drv.ms/u/s!Ah83ZdJvIaBnnjqVy9Eyn0yxDb8i?e=0Q3qRU)，密码：qjfk）。
3. Code is available at https://git.io/GDR-Net.

### 2. 论文摘要

单幅RGB图像的6D位姿估计是计算机视觉的基础任务。目前性能最好的基于深度学习的方法依赖于一种间接策略，即首先建立图像平面坐标与目标坐标系之间的2D-3D对应关系，然后应用PnP/RANSAC算法的变体。然而，这种两阶段的管道不是端到端可训练的，因此很难用于许多需要可微分姿态的任务。另一方面，目前基于直接回归的方法不如基于几何的方法。在本文中，我们对直接和间接方法进行了深入的研究，并提出了一个简单而有效的几何引导直接回归网络(GDR-Net)，以端到端方式从密集的基于对应的中间几何表示学习6D位姿。大量的实验表明，我们的方法在LM、LM- o和YCB-V数据集上的性能显著优于目前最先进的方法。代码可在https://git.io/GDR-Net上找到。

### 3. 摘要PPT

研究现状：单幅RGB图像的6D位姿估计是计算机视觉的基础任务。目前性能最好的基于深度学习的方法依赖于一种间接策略，即首先建立图像平面坐标与目标坐标系之间的2D-3D对应关系，然后应用PnP/RANSAC算法的变体。然而，这种两阶段的管道不是端到端可训练的，因此很难用于许多需要可微分姿态的任务。另一方面，目前基于直接回归的方法不如基于几何的方法。

创新点：在本文中，我们对直接和间接方法进行了深入的研究，并提出了一个简单而有效的几何引导直接回归网络(GDR-Net)，以端到端方式从密集的基于对应的中间几何表示学习6D位姿。

实验结果：大量的实验表明，我们的方法在LM、LM- o和YCB-V数据集上的性能显著优于目前最先进的方法。代码可在https://git.io/GDR-Net上找到。

## 论文六

### 1. 论文信息

### 2. 论文摘要

### 3. 摘要PPT

## 论文七

### 1. 论文信息

### 2. 论文摘要

### 3. 摘要PPT

## 论文八

### 1. 论文信息

### 2. 论文摘要

### 3. 摘要PPT

## 论文九

### 1. 论文信息

### 2. 论文摘要

### 3. 摘要PPT

## 论文十

### 1. 论文信息

### 2. 论文摘要

### 3. 摘要PPT
