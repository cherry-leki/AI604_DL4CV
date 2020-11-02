echo "Download summer2winter_yosemite dataset..."
URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/summer2winter_yosemite.zip
ZIP_FILE=./data/summer2winter_yosemite.zip
TARGET_DIR=./data/summer2winter_yosemite/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./data/
rm $ZIP_FILE
