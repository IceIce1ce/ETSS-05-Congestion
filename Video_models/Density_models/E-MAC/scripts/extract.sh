cd datasets/DroneBird
cat test.zip.001 test.zip.002 test.zip.003 > test.zip
unzip test.zip
rm test.zip
cat train.zip.001 train.zip.002 train.zip.003 train.zip.004 train.zip.005 train.zip.006 train.zip.007 train.zip.008 train.zip.009 train.zip.010 > train.zip
unzip train.zip
rm train.zip
unzip val.zip
cd ../..