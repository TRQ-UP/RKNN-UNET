# 1.文件目录

- 3rdparty:第三方库
  
- build:编译的文件
  
- CMakeLists.txt: cmakelist文件
  
- img:检测图片
  
- librknn_api: rknn使用的库
  
- model: rknn模型
  
- result.png: 推理结果图片
  
- src:源文件
  

# 2.使用平台

- RK3588

# 3.使用说明

- step1 cmake -S . -B build
  
- step2 cmake --build build
  
- step3 ./build/unet (可能没有权限，需要添加root权限，使用sudo)
