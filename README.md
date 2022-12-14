### 赛题背景
![image](https://github.com/WHJX272/Ocr_For_ID/blob/master/study_and_makeError/1.png)
![image](https://github.com/WHJX272/Ocr_For_ID/blob/master/study_and_makeError/2.png)
![image](https://github.com/WHJX272/Ocr_For_ID/blob/master/study_and_makeError/3.png)
### 前期试错
我在这里给出一些我前期的部分试错步骤。
思路是先不管赛题，做出一个能识别身份证的模型。
这里在study_and_makeError文件夹里提供了早期的模型。
![image](https://github.com/WHJX272/Ocr_For_ID/blob/master/study_and_makeError/4.png)
![image](https://github.com/WHJX272/Ocr_For_ID/blob/master/study_and_makeError/5.png)
![image](https://github.com/WHJX272/Ocr_For_ID/blob/master/study_and_makeError/6.png)
![image](https://github.com/WHJX272/Ocr_For_ID/blob/master/study_and_makeError/7.png)
不过这么弄下来倒还是让我理解了之后要做的工作。

### 环境配置
仅提供 Anaconda 的环境配置
1. 前往 [清华镜像](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/) 下载最新版本的 anaconda
2. 打开 Anaconda Prompt
3. 输入 `conda create -n <your_env_name> python=3.7` 创建虚拟环境
4. 输入 `activate <your_env_name>` 进入虚拟环境(此时命令行前的环境名应为你创建的虚拟环境)
5. 输入 `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch` 安装 pytorch
6. 更换 pycharm 的 python 解释器
   1. 选择 `文件 > 设置 > 项目 > python 解释器
   2. 点击 python 解释器 > 全部显示 > + 号 > conda 环境 > 现有环境
   3. 一直确认即可
   4. 等待 pycharm 进行包扫描(每次安装新的包都需要扫描)
7. 对于其他的依赖的安装(Anaconda Prompt)
**_不要使用其他包管理器, 使用 conda 进行安装_**
   1. 使用 `conda search <package>` 搜索依赖
   2. 使用 `conda install <package>=<version>` 安装依赖
   3. 尽量安装新版本的依赖(最好不要让 conda 更改原有依赖)
8. 一般来说，此项目按前面步骤配好环境后只需安装cv2即可，请前往Anaconda Navigator中的Environment搜索下载
