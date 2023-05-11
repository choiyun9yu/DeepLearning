# DeepLearning

## Tensorflow
- conda install -c apple tensorflow-deps
- pip install tensorflow-macos
- pip install tensorflow-metal
- CPU인지 GPU인지 확인
  import tensorflow as tf
  tf.config.experimental.list_physical_devices('GPU')
- 할당된 GPU 조회
  from tensorflow.python.client import device_lib
	device_lib.list_local_devices()

## Pytorch
- conda install pytorch (mps 안되는 경우 밑에 코드 실행)
- conda install pytorch torchvision -c pytorch-nightly
- conda install -c conda-forge pytorch
- torchvision : conda install 0c conda-forge torchvision
- torchtext : 
- Pytorch GPU 사용
  import torch
	device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
	print (f"PyTorch version:{torch.__version__}")
	print(f"MPS 장치를 지원하도록 build 되었는지: {torch.backends.mps.is_built()}")
	print(f"MPS 장치가 사용 가능한지: {torch.backends.mps.is_available()}")
	!python -c 'import platform;print(platform.platform())
