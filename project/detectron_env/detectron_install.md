# INSTALL DETECTRON2 IN A VIRTUAL ENVIRONMENT
works only in Linux / MacOS

## Install and create virtual environment
```
pip install virtualenv
virtualenv detectron_env
virtualenv -p /usr/bin/python3 detectron_env
source detectron_env/bin/activate
```

## Install dependencies
```
pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cpu/torch_stable.html
pip install cython pyyaml==5.1
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install opencv-python
pip install detectron2==0.1.2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/index.html
```

## Close virtualenv
```
deactivate
```
