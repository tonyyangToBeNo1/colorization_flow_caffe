#ifndef CAFFE_SPLIT_DATA_LAYER_HPP_
#define CAFFE_SPLIT_DATA_LAYER_HPP_

#include <stdio.h>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class SplitDataLayer : public Layer<Dtype> {
  public:
    explicit SplitDataLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>* >& bottom, const vector<Blob<Dtype>* >& top);
    virtual void Reshape(const vector<Blob<Dtype>* >& bottom, const vector<Blob<Dtype>*>& top);
    
    virtual inline const char* type() const { return "SplitData"; }
    virtual inline int ExactNumBottomBlobs() const {return 2;}
    virtual inline int MinTopBlobs() const {return 3;}
    virtual inline int MaxTopBlobs() const {return 3;}
  
  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>* >& bottom, const vector<Blob<Dtype>* >& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  private:
    int frame; 
    int count;
    int height;
    int width;
    int channel; 
};

} // namespace caffe

#endif // CAFFE_SPLIT_DATA_LAYER_HPP_
