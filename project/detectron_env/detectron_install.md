# INSTALL DETECTRON2 IN A VIRTUAL ENVIRONMENT
works only in Linux / MacOS

## Install and create virtual environment
```
pip install virtualenv
virtualenv detectron_env
virtualenv -p /usr/bin/python3 detectron_env
source detectron_env/bin/activate
```

## Install dependencies for cpu version
```
pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cpu/torch_stable.html
pip install cython pyyaml==5.1
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install opencv-python
pip install detectron2==0.1.2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/index.html
```

## Install dependencies for gpu version
```
pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html 
pip install cython pyyaml==5.1
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install opencv-python
pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html
```

## Close virtualenv
```
deactivate
```
