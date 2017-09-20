#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/split_data_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SplitDataLayer<Dtype>::LayerSetUp (
  const vector<Blob<Dtype>* >& bottom, const vector<Blob<Dtype>* >& top) {
    //int frame = this->layer_param_.split_data_param().frame;
    count = bottom[0]->count();
    channel = bottom[0]->channels();
    height = bottom[0]->height();
    width = bottom[0]->width();
    count = count / channel/ height / width;
  }

template <typename Dtype>
void SplitDataLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>* >& top) {
    //LOG(INFO) << count << " " << channel << " " << height << " " << width;
    top[0] -> Reshape(count / 2, channel, height, width);
    top[1] -> Reshape(count / 2, channel, height, width);
    top[2] -> ReshapeLike(*bottom[1]);
}

template <typename Dtype>
void SplitDataLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>* >& bottom, const vector<Blob<Dtype>* >& top) {
   //LOG(INFO) << count;
   const Dtype* bottom_data = bottom[0]->cpu_data();
   const Dtype* label = bottom[1]->cpu_data();
   Dtype* key = top[0]->mutable_cpu_data();
   Dtype* others = top[1]->mutable_cpu_data();
   Dtype* gt = top[2]->mutable_cpu_data();
   int j = 0;
   // in training, for baseline, set frame as 2 
   // should be modify later
   for (int i = 0; i < count; i += 2) {
     for (int c = 0; c < channel; c++) {
       for (int h = 0; h < height; h++) {
         for (int w = 0; w < width; w++) {
           key[j * channel * height * width + c * height * width + h * width + w] = bottom_data[i * channel * height * width + c * height * width + h * width + w];
           others[j * channel * height * width + c * height * width + h * width + w] = bottom_data[(i + 1) * (channel * height * width) + c * height * width + h * width + w];
         }
       }
     }
     j++;
   }
   j = 0;
   for (int i = 0; i < count; i += 2) {
     for (int c = 0; c < channel; c++) {
       for (int h = 0; h < height; h++) {
         for (int w = 0; w < width; w++) {
           gt[j * channel * height * width + c * height * width + h * width + w] = label[i * channel * height * width + c * height * width + h * width + w];
         }
       }
     }
     j++;
   }
   for (int i = 1; i < count; i += 2) {
     for (int c = 0; c < channel; c++) {
       for (int h = 0; h < height; h++) {
         for (int w = 0; w < width; w++) {
           gt[j * channel * height * width + c * height * width + h * width + w] = label[i * channel * height * width + c * height * width + h * width + w];
         }
       }
     }
     j++;
   }
 }

template <typename Dtype>
void SplitDataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>* >& top,
       const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom) {
    if (!propagate_down[0]) {return; }
    if (top.size() == 3) {
      const Dtype* key = top[0]->cpu_diff();
      const Dtype* others = top[1]->cpu_diff();
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      Dtype* gt_diff = bottom[1]->mutable_cpu_diff();
      for (int c = 0; c < count / 2; c++) {
        for (int chan = 0; chan < channel; chan++) {
          for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
              bottom_diff[2 * c * channel * height * width + chan * height * width + h * width + w] = key[c * channel * height * width + chan * height * width + h * width + w];
              bottom_diff[(2 * c + 1) * channel * height * width + chan * height * width + h * width + w] = others[c * channel * height * width + chan * height * width + h * width + w];
              gt_diff[2 * c * channel * height * width + chan * height * width + h * width + w] = 0;
              gt_diff[(2 * c + 1) * channel * height * width + chan * height * width + h * width + w] = 0;
            }
          }
        }
      }
    }
}

INSTANTIATE_CLASS(SplitDataLayer);
REGISTER_LAYER_CLASS(SplitData);

} // namespace caffe
