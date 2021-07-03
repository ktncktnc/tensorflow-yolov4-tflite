# tensorflow-yolov4-tflite with flask deployment 
Simple yolo-v4 tiny deployment to web with Flask.
### Prerequisites
* Tensorflow 2.3.0rc0

### Setup

```bash
cd tensorflow-yolov4-tflite
#For CPU
pip install -r requirements.txt
#For GPU
pip install -r requirements-gpu.txt
```
### Run
``` bash
python3 app.py
```
And then access http://127.0.0.1:5000/

### Performance
<p align="center"><img src="data/performance.png" width="640"\></p>

### References

  * YOLOv4: Optimal Speed and Accuracy of Object Detection [YOLOv4](https://arxiv.org/abs/2004.10934).
  * [darknet](https://github.com/AlexeyAB/darknet)
  
   My project is inspired by these previous fantastic YOLOv3 implementations:
  * [Yolov3 tensorflow](https://github.com/YunYang1994/tensorflow-yolov3)
  * [Yolov3 tf2](https://github.com/zzh8829/yolov3-tf2)

