#ifndef CAFFE_GENERATOR_LAYER_HPP_
#define CAFFE_GENERATOR_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class GeneratorLayer : public Layer<Dtype> {
  public:
    explicit GeneratorLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>* >& bottom, const vector<Blob<Dtype>* >&top);
    virtual void Reshape(const vector<Blob<Dtype>* >& bottom, const vector<Blob<Dtype>* >&top);
    virtual inline const char* type() const {return "Generator";}
    virtual inline int ExactNumBottomBlobs() const {return 2;}
    virtual inline int MinTopBlobs() const {return 1;}
    virtual inline int MaxTopBlobs() const {return 1;}
    
  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>* >& bottom, const vector<Blob<Dtype>* >& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>* >& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  private:
    int count;
    int channel;
    int height;
    int width;      
};
} // namespace caffe

#endif // CAFFE_GENERATOR_LAYER_HPP_
