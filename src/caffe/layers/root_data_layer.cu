/*
TODO:
- only load parts of the file, in accordance with a prototxt param "max_mem"
*/

#include <stdint.h>
#include <vector>

#include "caffe/layers/root_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void ROOTDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.root_data_param().batch_size();
  for (int i = 0; i < batch_size; ++i, ++current_row_) {
    if (current_row_ == root_blobs_[0]->shape(0)) {
      if (num_files_ > 1) {
        current_file_ += 1;
        if (current_file_ == num_files_) {
          current_file_ = 0;
          if (this->layer_param_.root_data_param().shuffle()) {
            std::random_shuffle(file_permutation_.begin(),
                                file_permutation_.end());
          }
          DLOG(INFO) << "Looping around to first file.";
        }
        LoadROOTFileData(
            root_filenames_[file_permutation_[current_file_]]);
      }
        LoadROOTFileData(
            root_filenames_[file_permutation_[current_file_]]);
      current_row_ = 0;
      if (this->layer_param_.root_data_param().shuffle())
        std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
    }
    for (int j = 0; j < this->layer_param_.top_size(); ++j) {
      int data_dim = top[j]->count() / top[j]->shape(0);
      caffe_copy(data_dim,
          &root_blobs_[j]->cpu_data()[data_permutation_[current_row_]
            * data_dim], &top[j]->mutable_gpu_data()[i * data_dim]);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ROOTDataLayer);

}  // namespace caffe
