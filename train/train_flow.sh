CAFFE_ROOT=/media/Disk/ziyang/code/colorization/caffe
COLOR_ROOT=/media/Disk/ziyang/code/colorization
#MODEL_ROOT=$COLOR_ROOT/models/colorization_release_v2.caffemodel
MODEL_ROOT=$COLOR_ROOT/train/models/colorization_flow_pretrain.caffemodel

LOG=$COLOR_ROOT/train/models/colornet/flow_warp_small.log

$CAFFE_ROOT/build/tools/caffe train -solver $COLOR_ROOT/train/solver_flow.prototxt -weights $MODEL_ROOT -gpu 1 2>&1 | tee $LOG
