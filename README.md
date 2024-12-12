# CS2_Autoaim
视觉，~~挂~~！目前功能只有按键辅助瞄准，无UI。本仓库提供完整开发流程

当前实战效果：

![](2024-05-07 23-28-12_1.mp4)

---

## 数据采集说明(./1getGraphData)

### ./screen_shot

使用rust开发，间隔时间截屏，配置好rust环境后可直接打开cmd，切换到该文件夹运行

```rust
cargo run
```

截图会存放在Scr_images文件夹

#### ScreenShot.exe使用

- 打开scrennshot.exe截图，语音控制，小键盘5开关
- 截取的为屏幕正中心范围
- 高保真无损截图，截取尺寸较大
- 减小图片尺寸可以原尺寸用cut.exe裁剪，也可以输入更小的尺寸
- 均以32倍数最佳
- 截图保存在ScreenShot文件夹中

### ./label

该部分含有labelimg.exe，可直接使用

## 部署说明(./2yolo/yolov5)
- 除了下载源码，你还需要下载environment.yml文件来配置conda环境，注意适配自己的开发环境
- 在使用environment.yml前，确保cuda和torch已经按照电脑配置正确安装
- 配置方法自行检索，使用时从anaconda的命令行激活环境后在```2yolo/yolov5```命令行python启动```detect.py```

## 训练说明(./2yolo/yolov5-7.0-full4train)

- 完整的yolo工具，用于训练各种数据