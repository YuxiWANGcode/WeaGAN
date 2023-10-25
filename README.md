# WeaGAN: Weather-Aware Graph Attention Network for Traffic Prediction

<div align=center><img width="500" src="https://github.com/YuxiWANGcode/WeaGAN/blob/main/figures/model.PNG"/></div>


This is a TensorFlow implementation our ECAI2023 paper: Wang Yuxi and Luo yuan.[WeaGAN: Weather-Aware Graph Attention Network for Traffic Prediction](https://www.researchgate.net/publication/374311830_WeaGAN_Weather-Aware_Graph_Attention_Network_for_Traffic_Prediction)

# Data 
The traffic datasets are available at [Google Drive](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g#list/path=%2F), provided by [DCRNN](https://github.com/liyaguang/DCRNN), and should be put into the corresponding `data/` folder.

The weather datasets are available at [OpenWeather](https://openweathermap.org/) by use the API to get the data you need. Here is a dataset of wind speed to demonstrate the format of multi-weather datasets. The dataset are available at [Google Drive](https://drive.google.com/file/d/1q75QGcMZFw6P9Ke7HuzbPQ7Xpg1MvMkL/view?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/1vvPSbt-gZqIxcCPrwpFaYw?pwd=azcn).

# Requirements
Dependency can be installed using the following command:
```
pip install -r denpendency.txt
```
# Results
<div align=center><img hight="500" src="https://github.com/YuxiWANGcode/WeaGAN/blob/main/figures/result.PNG"/></div>
