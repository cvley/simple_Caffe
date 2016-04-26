#include "blob.hpp"

namespace caffe {

template <typename Dtype>
void Blob<Dtype>::Reshape(const int num, const int channels, const int height,
    const int width) {
  CHECK_GE(num, 0);
  CHECK_GE(channels, 0);
  CHECK_GE(height, 0);
  CHECK_GE(width, 0);
  num_ = num;
  channels_ = channels;
  height_ = height;
  width_ = width;
  count_ = num_ * channels_ * height_ * width_;
  if (count_ > capacity_) {
    capacity_ = count_;
	delete [] data_;
	data_ = new Dtype[capacity_ * sizeof(Dtype)];
	delete [] diff_;
	diff_ = new Dtype[capacity_ * sizeof(Dtype)];
  }
}

template <typename Dtype>
Blob<Dtype>::Blob(const int num, const int channels, const int height,
    const int width)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0) {
  Reshape(num, channels, height, width);
}

template <typename Dtype>
void Blob<Dtype>::FromProto(const BlobProto& proto) {
  Reshape(proto.num(), proto.channels(), proto.height(), proto.width());
  // copy data
  for (int i = 0; i < count_; ++i) {
    data_[i] = proto.data(i);
  }
  if (proto.diff_size() > 0) {
    for (int i = 0; i < count_; ++i) {
      diff_[i] = proto.diff(i);
    }
  }
}

}  // namespace caffe

