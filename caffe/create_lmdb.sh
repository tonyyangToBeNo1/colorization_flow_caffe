DATA=/media/Disk/ziyang/code/data
CAFFE_ROOT=/media/Disk/ziyang/code/colorization/caffe
TOOLS=build/tools
rm -r $DATA/caffe-train-lmdb
rm -r $DATA/caffe-val-lmdb
rm -r $DATA/caffe-small-lmdb
echo "creating training lmdb"

GLOG_logtostderr=1 $TOOLS/convert_imageset \
  --shuffle --resize_height=256 --resize_width=256 \
  $DATA/gray/ \
  $DATA/train.txt \
  $DATA/caffe-train-lmdb

echo "creating val lmdb"
GLOG_logtostderr=1 $TOOLS/convert_imageset \
  --shuffle --resize_height=256 --resize_width=256 \
  $DATA/gray/ \
  $DATA/val.txt \
  $DATA/caffe-val-lmdb

echo "done!"
