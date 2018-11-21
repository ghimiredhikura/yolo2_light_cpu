# yolo2_light
Light version of convolutional neural network Yolo v2 for objects detection with a minimum of dependencies (INT8-inference, BIT1-XNOR-inference)

This repository supports:

* Windows - cpu only

How to compile:
* Build `yolo_cpu.sln`
    
How to use **INT8**-inference:
* Use flag `-quantized` at the end of command, for example, `yolo_cpu.exe detector test tiny-yolov2/voc.names tiny-yolov2/yolov2-tiny-voc.cfg tiny-yolov2/yolov2-tiny-voc.weights -thresh 0.10 person.jpg -quantized`
* For the custom dataset, you should use `input_calibration=` parameter in your cfg-file, from the correspon cfg-file: [`yolov2-tiny-voc.cfg`](https://github.com/ghimiredhikura/yolo2_light_cpu/blob/f8b8937759c2636bca44887275e6b740d17170db/bin/tiny-yolov2/yolov2-tiny-voc.cfg#L24), ...

How to use **BIT1-XNOR**-inference:
* You should base your cfg-file on [`tiny-yolo-obj_xnor.cfg`](https://github.com/AlexeyAB/yolo2_light/blob/master/bin/tiny-yolo-obj_xnor.cfg) and train it by using this repository as usual https://github.com/AlexeyAB/darknet

* Then use it for Detection-test:
    * `yolo_cpu.exe detector test tiny-yolo-voc_xnor/voc.names tiny-yolo-voc_xnor/tiny_yolo_xnor.cfg tiny-yolo-voc_xnor/tiny_yolo_xnor.weights -thresh 0.21 person.jpg`
	* Note: here we provide `tiny_yolo_xnor.weights` which was trained in voc dataset with [tiny_yolo_xnor-train.cfg](bin/tiny-yolo-voc_xnor/tiny_yolo_xnor-train.cfg)