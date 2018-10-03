
wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
unzip dogImages.zip
rm dogImages.zip

wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip
unzip lfw.zip
rm lfw.zip

mkdir bottleneck_features
wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz -P bottleneck_features/
wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz -P bottleneck_features/

mkdir test_images
wget -cO - https://cdn.pixabay.com/photo/2017/04/06/09/16/arrow-2207748_960_720.png > test_images/arrow.png
wget -cO - https://cdn.pixabay.com/photo/2018/04/27/16/49/dog-3355192_960_720.jpg > test_images/Boxer.jpg
wget -cO - https://cdn.pixabay.com/photo/2014/11/14/05/27/yellow-530245_1280.jpg > test_images/Labrador_retriever.jpg
wget -cO - https://cdn.pixabay.com/photo/2018/08/20/14/08/dog-3619020_1280.jpg > test_images/Pomeranian.jpg
wget -cO - https://cdn.pixabay.com/photo/2018/03/27/21/50/nature-3267539_960_720.jpg > test_images/Golden_retriever.jpg
wget -cO - https://cdn.pixabay.com/photo/2018/02/21/15/06/woman-3170568_960_720.jpg > test_images/woman.jpg
wget -cO - https://cdn.pixabay.com/photo/2017/06/28/04/29/adult-2449725_960_720.jpg > test_images/man.jpg

#sudo yum install cmake
#sudo yum install libmpc-devel mpfr-devel gmp-devel
#sudo yum install boost boost-devel boost-doc
#sudo pip3 install --user opencv-python
#sudo pip3 install --user opencv-contrib-python
#sudo pip3 install --user opencv-python-headless
#sudo pip3 install --user opencv-contrib-python-headless
#sudo pip3 install --user dlib
python3 dog_breed.py
