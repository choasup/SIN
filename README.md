# SIN
Structure Inference Net: Object Detection Using Scene-level Context and Instance-level Relationships. In CVPR 2018.

### Requirements: software

1. Requirements for Tensorflow (see: [Tensorflow](https://www.tensorflow.org/))

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
    ln -s $VOCdevkit VOCdevkit2007
    ```
    
5. Download the pre-trained ImageNet models [[Google Drive]](https://drive.google.com/open?id=0ByuDEGFYmWsbNVF5eExySUtMZmM) [[Dropbox]](https://www.dropbox.com/s/po2kzdhdgl4ix55/VGG_imagenet.npy?dl=0)
   
   ```Shell
    mv VGG_imagenet.npy $SIN_ROOT/data/pretrain_model/VGG_imagenet.npy
    ```

6. Run script to train and test model
	```Shell
	cd $SIN_ROOT
	./train.sh
	```
  DEVICE is either cpu/gpu

### The result of testing on PASCAL VOC 2007 

### References
[Faster R-CNN caffe version](https://github.com/rbgirshick/py-faster-rcnn)

[Faster R-CNN tf version](https://github.com/smallcorgi/Faster-RCNN_TF)

### Citation
