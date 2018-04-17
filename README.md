# SIN
Structure Inference Net: Object Detection Using Scene-level Context and Instance-level Relationships. In CVPR 2018.(http://vipl.ict.ac.cn/uploadfile/upload/2018041318013480.pdf)

### Requirements: software

1. Requirements for Tensorflow 1.3.0 (see: [Tensorflow](https://www.tensorflow.org/))

2. Python packages you might not have: `cython`, `python-opencv`, `easydict`

### Installation (sufficient for the demo)

1. Clone the SIN repository
  
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive https://github.com/choasUp/SIN.git
  ```

2. Build the Cython modules
  ```Shell
  cd $SIN_ROOT/lib
  make
  ```

### Demo

*After successfully completing [basic installation](#installation-sufficient-for-the-demo)*, you'll be ready to run the demo.

Wait ...

### Training Model
1. Download the training, validation, test data and VOCdevkit

	```Shell
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
	```

2. Extract all of these tars into one directory named `VOCdevkit`

	```Shell
	tar xvf VOCtrainval_06-Nov-2007.tar
	tar xvf VOCtest_06-Nov-2007.tar
	tar xvf VOCdevkit_08-Jun-2007.tar
	```

3. It should have this basic structure

	```Shell
  	$VOCdevkit/                           # development kit
  	$VOCdevkit/VOCcode/                   # VOC utility code
  	$VOCdevkit/VOC2007                    # image sets, annotations, etc.
  	# ... and several other directories ...
  	```

4. Create symlinks for the PASCAL VOC dataset

    ```Shell
    cd $SIN_ROOT/data
    ln -s $VOCdevkit VOCdevkit
    ```
    
5. Download the pre-trained ImageNet models [[Google Drive]](https://drive.google.com/open?id=0ByuDEGFYmWsbNVF5eExySUtMZmM) [[Dropbox]](https://www.dropbox.com/s/po2kzdhdgl4ix55/VGG_imagenet.npy?dl=0)
   
   ```Shell
    mv VGG_imagenet.npy $SIN_ROOT/data/pretrain_model/VGG_imagenet.npy
    ```

6. [optional] Set learning rate and max iter

	```Shell
   	vim experiments/scripts/faster_rcnn_end2end.sh 		# ITERS
	vim lib/fast/config.py 					# LR
	cd lib	 						# if you edit the code, make best
	make
	```

7. Set your GPU id, then run script to train and test model
	```Shell
	cd $SIN_ROOT
	export CUDA_VISIBLE_DEVICSE=0
	./train.sh
	```

8. Test your dataset
	```Shell
	./test_all.sh
	```

### The result of testing on PASCAL VOC 2007 (VGG net)
```
AP for aeroplane = 0.7853
AP for bicycle = 0.8045
AP for bird = 0.7456
AP for boat = 0.6657
AP for bottle = 0.6144
AP for bus = 0.8424
AP for car = 0.8663
AP for cat = 0.8894
AP for chair = 0.5803
AP for cow = 0.8466
AP for diningtable = 0.7171
AP for dog = 0.8578
AP for horse = 0.8626
AP for motorbike = 0.7802
AP for person = 0.7857
AP for pottedplant = 0.4869
AP for sheep = 0.7599
AP for sofa = 0.7351
AP for train = 0.8199
AP for tvmonitor = 0.7683
Mean AP = 0.7607
```

### References
[Faster R-CNN caffe version](https://github.com/rbgirshick/py-faster-rcnn)

[Faster R-CNN tf version](https://github.com/smallcorgi/Faster-RCNN_TF)

### Citation
