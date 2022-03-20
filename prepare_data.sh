mkdir -p data_dir
mkdir -p data_dir/SPCOCO
mkdir -p data_dir/coco
wget http://images.cocodataset.org/zips/train2017.zip -P data_dir/coco/.
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P data_dir/coco/annotations_trainval2017.zip
unzip data_dir/coco/annotations_trainval2017.zip -d data_dir/coco
python coco_colours_ly2c.py --cons_ratio 0.999_0.7_0.1 --noise_ratio 0.05 --image_scale 32
