# Set up Win10 Step

## Introduction
1.install tensorflow_GPU  
2.set up object detection directory structure and Anaconda Virtuai Environment  
3.gather and label picture  
```
use labeimg
```
4.generate training data  
5.create label map and configure training  
6.train object detector  
7.export inference graph  
8.test it  

### Step(python3.6)

- 1.install tensorflow_GPU  

  <p>
  Follow 
  <a href="https://www.youtube.com/watch?v=RplXYjxgZbw" title="Title">
  this video</a> inline link.
  </p>
  installing Anaconda CUDA version 9.0 and cuDNN version 7.0.5 do not need install tensorflow

- 2.set up object detection directory structure and Anaconda Virtuai Environment  

  1.create dir in C:// named tensorflow1  

  2.get model  
  <p>
  Download
  <a href="https://github.com/tensorflow/models" title="Title">
  this model</a> inline link.
  </p>
  and named it models then throw it to tensorflow1  

  3.get faster_rcnn_inception_v2  
  <p>
  Download faster_rcnn_inception_v2 
  <a href="http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz" title="Title">
  this model</a> inline link.
  </p>

  4.extract faster_rcnn_inception_v2 to models/reseach/object_detection  

  5.download TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10  
  <p>
  Download 
  <a href="https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10" title="Title">
  this repos </a> inline link.
  </p>

  6.extract TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10 to models/reseach/object_detection  
  readme.md can be replaced  

  7. go to training folder and delete all in this folder  
  go to inference_graph folder delete all in this folder  
  go to images folder delete test_labels and train_labels  
  go to images/test delete all in this folder  
  go to images/train delete all in this folder  

  8.install tensorflow  
  use Anaconda Prompt run as administrator  
```
conda create -n tensorflow1 pip
```
```
activate tensorflow1
```
```
pip install --ignore-installed --upgrade tensorflow-gpu
```
```
conda install -c anaconda protobuf
```
```
pip install pillow
```
```
pip install lxml
```
```
pip install jupyter
```
```
pip install matplotlib
```
```
pip install pandas
```
```
pip install opencv-python
```
```
set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
```
```
set PATH=%PATH%;PYTHONPATH
```
```
cd C:\tensorflow1\models\research
```
```
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto .\object_detection\protos\flexible_grid_anchor_generator.proto
```
```
python setup.py build
```
```
python setup.py install
```
  9.test it  
```
cd C:\tensorflow1\models\research\object_detection
```
```
jupyter notebook object_detection_tutorial.ipynb
```
  press run to the end you will see picture if it does not have problems  

- 3.gather and label picture  
  take picture around 90up  
  and labeled each picture(use labelimg)  
  press Create RectBox  

- 4.generate training data  
  besure in the tensorflow1 env and cd C:\tensorflow1\models\research\object_detection  
```
python xml_to_csv.py
```
  edit generate_tfrecord use idle replace the labelmap  
```
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
```
```
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
```

- 5.create label map and configure training  
  create labelmap.pbtxt use any editor and save as all types in the object_detection/training  
```
example
item{
  id:1
  name:'name1'
}
```
  copy faster_rcnn_inception_v2_pets.config at sample/configs to ../training  
  edit faster_rcnn_inception_v2_pets.config  
  replace the num_classes of your detetion numbers  
  replace fine_tune_checkpoint: "C:/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_18/model.ckpt"  
  change train_input_reader  
  input_path: "C:/tensorflow1/models/research/object_detection/train.record"  
  label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"  
  change eval_config  
  num_examples: the picture numbers in the test folder  
  eval_input_reader  
  input_path: "C:/tensorflow1/models/research/object_detection/train.record"  
  label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"  

- 6.train object detector  
  besure in the tensorflow1 env and cd C:/tensorflow1/models/research/object_detection  
```
python train.py --logtostderr --train_dir=training/ --pipeline_config=training/faster_rcnn_inception_v2_pets.config
```
  you can check the process   
```
tensorboard --logdir=training
```
  it will take 3 to5 hours utill the number around 0.02  
  check the highest number in the training folder  
- 7.export inference  
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```
  the xxxx is the highest number in the training folder  
- 8.test it  
  change the num_classes in the image.py,video.py and webcam.py  
  run the mod 
