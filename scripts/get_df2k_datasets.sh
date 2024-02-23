wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
wget https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar

mkdir -p ../data/DF2K/DF2K
unzip DIV2K_train_HR.zip -d ../data/DF2K/DF2K
tar -xvf Flickr2K.tar -C ../data/DF2K/DF2K
