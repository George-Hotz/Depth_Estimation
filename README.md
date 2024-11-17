# 单目深度估计的模型部署(C++ TensorRT)
- 轻松实现Depth Estimation的TensorRT高性能推理
- 没有复杂的包装，没有耦合!


# 结果展示
![](assets/result.gif)


# 流程
### Step1 cmake编译工程
`cmake -S . -B build`

### Step2 build工程
`cmake --build build`

### Step3: 转化模型引擎 
`./build/build --onnx_file=vits.onnx`

### Step4: Yolov8-seg 推理部署
`./build/main --thread_num 4 --video_path input/videos/road.mp4 --model_path weights/vits_fp16.engine`

# 参考
- [💡Tutorial : C++ TensorRT High-performance deployments（恩培计算机视觉）](https://enpeicv.com/)

