CAFFE_ROOT=/media/Disk/ziyang/code/colorization/caffe
COLOR_ROOT=/media/Disk/ziyang/code/colorization
#MODEL_ROOT=$COLOR_ROOT/models/colorization_release_v2.caffemodel
MODEL_ROOT=$COLOR_ROOT/train/models/color_all_iter_20000.caffemodel
LOG=$COLOR_ROOT/train/models/colornet/color.log
$CAFFE_ROOT/build/tools/caffe train -solver $COLOR_ROOT/train/solver.prototxt -weights $MODEL_ROOT -gpu 3 2>&1 | tee $LOG
