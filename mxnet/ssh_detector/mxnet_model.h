#ifndef _MXNET_MODEL_H_
#define _MXNET_MODEL_H_

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
// Path for c_predict_api
#include "mxnet/c_predict_api.h"

// Read file to buffer
class mxBufferFile {
 public :
  std::string file_path_;
  std::size_t length_ = 0;
  std::unique_ptr<char[]> buffer_;

  mxBufferFile(const std::string& file_path);

  inline std::size_t GetLength() {
    return length_;
  }

  inline char* GetBuffer() {
    return buffer_.get();
  }
};

class mxInputShape {
public:
  mx_uint input_shape_indptr[2] = {0,4};
  mx_uint input_shape_data[4];

  mxInputShape(int width, int height, int channels);
};

void mxGetImageFile(const std::string& image_file, std::vector<mx_float>& image_data );

/*
 * Load mxnet model
 *
 * Inputs:
 * - json_file:  path to model-symbol.json
 * - param_file: path to model-0000.params
 * - shape: input shape to mxnet model (1, channels, height, width)
 * - dev_type: 1: cpu, 2:gpu
 * - dev_id: 0: arbitary
 *
 * Output:
 * - PredictorHandle
 */
void mxLoadMXNetModel ( PredictorHandle* pred_hnd, /* Output */
                      std::string json_file,     /* path to model-symbol.json */
                      std::string param_file,    /* path to model-0000.params */
                      mxInputShape shape,          /* input shape to mxnet model (1, channels, height, width) */
                      int dev_type = 1,          /* 1: cpu, 2:gpu */
                      int dev_id = 0             /* 0: arbitary */
                    );

void mxHandleReshape(PredictorHandle  handle,   /* mxnet model handle */
                     mxInputShape     shape,    /* new shape */
                     PredictorHandle* out);     /* new hanlde */

void mxInfer ( PredictorHandle pred_hnd,         /* mxnet model */
           std::vector<mx_float> &image_data);   /* input data */ 

void mxOutputOfIndex ( PredictorHandle pred_hnd, /* mxnet model */
           std::vector<float> &data,           /* output vector */
           std::vector<int> &out_shape,        /* output tensor shape */
           mx_uint output_index );

void mxPrintOutputResult(const std::vector<float>& output);

#endif // _MXNET_MODEL_H_