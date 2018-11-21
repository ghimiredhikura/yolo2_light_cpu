# yolo2_light
Light version of convolutional neural network Yolo v3 & v2 for objects detection with a minimum of dependencies (INT8-inference, BIT1-XNOR-inference)

This repository supports:

* Windows - cpu only

How to compile:
* Build `yolo_cpu.sln`
    
How to use **INT8**-inference:
* Use flag `-quantized` at the end of command, for example, [`tiny-yolo-int8.sh`](https://github.com/AlexeyAB/yolo2_light/blob/master/bin/tiny-yolo-int8.sh) or [`yolo_cpu_int8.cmd`](https://github.com/AlexeyAB/yolo2_light/blob/master/bin/yolo_cpu_int8.cmd)
* For the custom dataset, you should use `input_calibration=` parameter in your cfg-file, from the correspon cfg-file: [`yolov3-tiny.cfg`](https://github.com/AlexeyAB/yolo2_light/blob/29905072f194ee86fdeed6ff2d12fed818712411/bin/yolov3-tiny.cfg#L25) or [`yolov3.cfg`](https://github.com/AlexeyAB/yolo2_light/blob/29905072f194ee86fdeed6ff2d12fed818712411/bin/yolov3.cfg#L25), ...

How to use **BIT1-XNOR**-inference:
* You should base your cfg-file on [`tiny-yolo-obj_xnor.cfg`](https://github.com/AlexeyAB/yolo2_light/blob/master/bin/tiny-yolo-obj_xnor.cfg) and train it by using this repository as usual https://github.com/AlexeyAB/darknet

* Then use it for Detection-test:
    * `yolo_cpu.exe detector test tiny-yolo-voc_xnor/voc.names tiny-yolo-voc_xnor/tiny_yolo_xnor.cfg tiny-yolo-voc_xnor/tiny_yolo_xnor.weights -thresh 0.21 person.jpg`
	* Note: here we provide `tiny_yolo_xnor.weights` which was trained in voc dataset with [tiny_yolo_xnor-train.cfg](bin/tiny-yolo-voc_xnor/tiny_yolo_xnor-train.cfg)