Venv:  
mkdir venv && python3 -m venv venv/  
source venv/bin/activate  
(venv) pip3 install torch gym  
sudo apt-get install zlib1g-dev cmake  
(venv) pip3 install 'msgpack==1.0.2' gym[atari] tensorboard  
(venv) pip3 install wheel  
(venv) pip3 install pygame matplotlib  

Tensorboard:  
tensorboard --logdir ./logs/  
rm -rv logs/*  
