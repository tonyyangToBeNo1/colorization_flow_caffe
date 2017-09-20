CAFFE_ROOT=/media/Disk/ziyang/code/colorization/caffe
COLOR_ROOT=/media/Disk/ziyang/code/colorization
MODEL_ROOT=$COLOR_ROOT/models/colorization_release_v2.caffemodel
#MODEL_ROOT=$COLOR_ROOT/train/models/flow_warp_trainflow_iter_511.caffemodel
LOG=$COLOR_ROOT/train/models/colornet/flow_warp_trainflow.log
$CAFFE_ROOT/build/tools/caffe train -solver $COLOR_ROOT/train/solver_trainflow.prototxt -weights $MODEL_ROOT -gpu 2 2>&1 | tee $LOG
