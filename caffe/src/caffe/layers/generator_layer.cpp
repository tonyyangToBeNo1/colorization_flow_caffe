#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/generator_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GeneratorLayer<Dtype>::LayerSetUp (const vector<Blob<Dtype>* >& bottom, const vector<Blob<Dtype>* >&top) {
  count = bottom[0]->count();
  channel = bottom[0]->channels();
  height = bottom[0]->height();
  width = bottom[0]->width();
  count = count / channel / height / width;
} 

template <typename Dtype>
void GeneratorLayer<Dtype>::Reshape(const vector<Blob<Dtype>* >& bottom, const vector<Blob<Dtype>* >& top) {
  top[0]->ReshapeLike(*bottom[1]);
}

template <typename Dtype>
void GeneratorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>* >& bottom, const vector<Blob<Dtype>* >& top) {
  const Dtype* flow_data = bottom[0]->cpu_data();
  const Dtype* key_conv_data = bottom[1]->cpu_data();
  Dtype* warp_conv = top[0]->mutable_cpu_data();
  vector<vector<double> > x;
  vector<vector<double> > y;
  // get all the xi
  // get all the yi
  for (int c = 0; c < count; c++) {
    for (int h = 0; h < height; h++) {
      vector<double> temp_x;
      vector<double> temp_y;
      for (int w = 0; w < width; w++) {
        temp_x.push_back(flow_data[c * channel * height * width + h * width + w]);
        temp_y.push_back(flow_data[c * channel * height * width + height * width + h * width + w]);
      }
      x.push_back(temp_x);
      y.push_back(temp_y);
    }
  }
  int conv_channels = bottom[1]->channels();
  for (int c = 0; c < count; c++) {
    for (int chan = 0; chan < conv_channels; chan++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          double warp_h = h + x[h][w];
          double warp_w = w + y[h][w];
          if (warp_h >= height || warp_h <= 0 || warp_w >= width || warp_w <= 0) {
            warp_conv[c * conv_channels * height * width + chan * height * width + h * width + w] = 0;
	  } else {
          // interp to get the feature
          int min_max_h = (int)floor(warp_h);
          int max_min_h = (int)ceil(warp_h);
          int min_max_w = (int)floor(warp_w);
          int max_min_w = (int)ceil(warp_w);
          double h1 = warp_h - min_max_h;
          double h2 = max_min_h = warp_h;
          double w1 = warp_w - min_max_h;
          double w2 = max_min_w - warp_w;
          int offset = c * conv_channels * height * width + chan * height * width;
          warp_conv[c * conv_channels * height * width + chan * height * width + h * width + w] = 
              key_conv_data[offset + min_max_h * width + min_max_w] * h2 * w2 
            + key_conv_data[offset + min_max_h * width + max_min_w] * h2 * w1 
            + key_conv_data[offset + max_min_h * width + min_max_w] * h1 * w2
            + key_conv_data[offset + max_min_h * width + max_min_w] * h1 * w2;
        }
	}
      }
    }
  }
}

template <typename Dtype>
void GeneratorLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>* >& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom) {
  if (!propagate_down[0]) {return; }
  const Dtype* flow_diff = top[0]->cpu_diff();
  Dtype* flow_bottom = bottom[0]->mutable_cpu_diff();
  Dtype* conv_bottom = bottom[1]->mutable_cpu_diff();
  
  // bp to flownet
  //for (int c = 0)
  for (int c = 0; c < 2; c++) {
    for (int chan = 0; chan < channel; chan++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          flow_bottom[c * channel * height * width + chan * height * width + h * width + w] = flow_diff[c * channel * height * width + chan * height * width + h * width + w];
        }
      }
    }
  }

  
  // need not to bp to conv
  for (int c = 0; c < count; c++) {
    for (int chan = 0; chan < channel; chan++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          conv_bottom[c * channel * height * width + chan * height * width + h * width + w] = 0;
        }
      }
    }
  }


}

INSTANTIATE_CLASS(GeneratorLayer);
REGISTER_LAYER_CLASS(Generator);
}
