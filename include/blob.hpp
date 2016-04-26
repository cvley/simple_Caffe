#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include <glog/logging.h>
#include "caffe.pb.h"

namespace caffe {

/**
 * @brief A wrapper around SyncedMemory holders serving as the basic
 *        computational unit through which Layer%s, Net%s, and Solver%s
 *        interact.
 *
 * TODO(dox): more thorough description.
 */
template <typename Dtype>
class Blob {
 public:
  Blob()
       : data_(), diff_(), num_(0), channels_(0), height_(0), width_(0),
       count_(0), capacity_(0) {}
  explicit Blob(const int num, const int channels, const int height,
    const int width);
  /**
   * @brief Change the dimensions of the blob, allocating new memory if
   *        necessary.
   *
   * This function can be called both to create an initial allocation
   * of memory, and to adjust the dimensions of a top blob during Layer::Reshape
   * or Layer::Forward. When changing the size of blob, memory will only be
   * reallocated if sufficient memory does not already exist, and excess memory
   * will never be freed.
   *
   * Note that reshaping an input blob and immediately calling Net::Backward is
   * an error; either Net::Forward or Net::Reshape need to be called to
   * propagate the new input shape to higher layers.
   */
  void Reshape(const int num, const int channels, const int height,
    const int width);
  inline int num() const { return num_; }
  inline int channels() const { return channels_; }
  inline int height() const { return height_; }
  inline int width() const { return width_; }
  inline int count() const { return count_; }
  inline int offset(const int n, const int c = 0, const int h = 0,
      const int w = 0) const {
    CHECK_GE(n, 0);
    CHECK_LE(n, num_);
    CHECK_GE(channels_, 0);
    CHECK_LE(c, channels_);
    CHECK_GE(height_, 0);
    CHECK_LE(h, height_);
    CHECK_GE(width_, 0);
    CHECK_LE(w, width_);
    return ((n * channels_ + c) * height_ + h) * width_ + w;
  }

  inline Dtype data_at(const int n, const int c, const int h,
      const int w) const {
    return *(data_ + offset(n, c, h, w));
  }

  inline Dtype diff_at(const int n, const int c, const int h,
      const int w) const {
    return *(diff_ + offset(n, c, h, w));
  }

  void FromProto(const BlobProto& proto);

 protected:
  Dtype* data_;
  Dtype* diff_;
  int num_;
  int channels_;
  int height_;
  int width_;
  int count_;
  int capacity_;

};  // class Blob

}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_
