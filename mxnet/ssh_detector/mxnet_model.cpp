/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Xiao Liu, pertusa, caprice-j
 * \file image_classification-predict.cpp
 * \brief C++ predict example of mxnet
 *
 * This is a simple predictor which shows how to use c api for image classification. It uses
 * opencv for image reading.
 *
 * Created by liuxiao on 12/9/15.
 * Thanks to : pertusa, caprice-j, sofiawu, tqchen, piiswrong
 * Home Page: www.liuxiao.org
 * E-mail: liuxiao@foxmail.com
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <memory>
#include <iomanip>
#include <cassert>

#include "mxnet_model.h"

int m_channel=0;
int m_width=0;
int m_height=0;

mxBufferFile::mxBufferFile(const std::string& file_path):file_path_(file_path) {

    std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
    if (!ifs) {
      std::cerr << "Can't open the file. Please check " << file_path << ". \n";
      return;
    }

    ifs.seekg(0, std::ios::end);
    length_ = static_cast<std::size_t>(ifs.tellg());
    ifs.seekg(0, std::ios::beg);
    std::cout << file_path.c_str() << " ... " << length_ << " bytes\n";

    buffer_.reset(new char[length_]);
    ifs.read(buffer_.get(), length_);
    ifs.close();
}

mxInputShape::mxInputShape(int width, int height, int channels) {
    input_shape_data[0] = 1;
    input_shape_data[1] = static_cast<mx_uint>(channels); 
    input_shape_data[2] = static_cast<mx_uint>(height);
    input_shape_data[3] = static_cast<mx_uint>(width);
}

// input opencv image matrix
// output image vecotr for mxnet
// input mat will be resize if it does not have a shape equal to the mxnet model
void mxImgFormConvert( const cv::Mat input, std::vector<mx_float> & img_vec) {
  cv::Mat ori,im;
  if(!input.isContinuous()){
    input.copyTo(ori);
    if(!ori.isContinuous()){
      std::cerr << "not enough continous memory for opencv mat\n";
      exit(1);
    }
  } else {
    ori = input;
  }
  if(ori.channels()!=m_channel){
      std::cerr << "mat channel is " << ori.channels()  << " ,can not proccess!\n";
      exit(1);
  }

  if(ori.cols*ori.rows!= m_width*m_height) {
    cv::resize(ori,im,cv::Size(m_width,m_height));
  } else
    im = ori;

  int size = m_channel * m_width * m_height;
  img_vec.resize(size);
  mx_float* image_data  = img_vec.data();
  mx_float* ptr_image_r = image_data;
  mx_float* ptr_image_g = image_data + size / 3;
  mx_float* ptr_image_b = image_data + size / 3 * 2;

  for (int i = 0; i < im.rows; i++) {
    auto data = im.ptr<uchar>(i);

    for (int j = 0; j < im.cols; j++) {
        *ptr_image_b++ = static_cast<mx_float>(*data++);
        *ptr_image_g++ = static_cast<mx_float>(*data++);
        *ptr_image_r++ = static_cast<mx_float>(*data++);
    }
  }

}

void mxGetImageFile(const std::string& image_file, std::vector<mx_float> & image_data ) {
  // Read all kinds of file into a BGR color 3 channels image
  // the shape(height, width, channels)
  cv::Mat im = cv::imread(image_file, cv::IMREAD_COLOR);

  if (im.empty()) {
    std::cerr << "Can't open the image. Please check " << image_file << ". \n";
    assert(false);
  }

  int size = im.rows * im.cols * 3;
  image_data.resize(size);

  mx_float* ptr_image_r = image_data.data();
  mx_float* ptr_image_g = image_data.data() + size / 3;
  mx_float* ptr_image_b = image_data.data() + size / 3 * 2;

  for (int i = 0; i < im.rows; i++) {
    auto data = im.ptr<uchar>(i);

    for (int j = 0; j < im.cols; j++) {
        *ptr_image_b++ = static_cast<mx_float>(*data++);
        *ptr_image_g++ = static_cast<mx_float>(*data++);
        *ptr_image_r++ = static_cast<mx_float>(*data++);
    }
  }
}

void mxInfer ( PredictorHandle pred_hnd,         /* mxnet model */
           std::vector<mx_float> &image_data   /* input data */ )
{         
  // Set Input Image
  int res = MXPredSetInput(pred_hnd, "data", image_data.data(), static_cast<mx_uint>(image_data.size()));
  assert(res == 0);
  // Do Predict Forward
  res = MXPredForward(pred_hnd);
  assert(res == 0);
}

void mxOutputOfIndex ( PredictorHandle pred_hnd, /* mxnet model */
           std::vector<float> &data,           /* output vector */
           std::vector<int> &out_shape,        /* output tensor shape */
           mx_uint output_index ) { 

  mx_uint* shape = nullptr;
  mx_uint shape_len;

  // Get Output Result
  int res = MXPredGetOutputShape(pred_hnd, output_index, &shape, &shape_len);
  assert(res == 0);

  // std::cout << "output shape_len: " << shape_len << std::endl;
  // for(mx_uint index=0; index<shape_len; index++) std::cout << shape[index] << std::endl;

  assert(shape_len==4);

  std::size_t size = 1;
  out_shape.clear();
  for (mx_uint i = 0; i < shape_len; ++i) { 
      size *= shape[i]; 
      out_shape.push_back(shape[i]);
  }

  data.resize(size);

  res = MXPredGetOutput(pred_hnd, output_index, &(data[0]), static_cast<mx_uint>(size));
  assert(res == 0);

}

void mxPrintOutputResult(const std::vector<float>& output) {
    std::cout<< "embedding size: " << output.size() <<"\n";
    for(int i=0; i < output.size(); ++i) {
    std::cout << output[i];
    if((i+1) % 16 == 0) {
      std::cout << std::endl;
    } else {
      std::cout << " ";
    }
  }
}

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
                      int dev_type,              /* 1: cpu, 2:gpu */
                      int dev_id ) {             /* 0: arbitary */

  mxBufferFile json_data(json_file);
  mxBufferFile param_data(param_file);

  if (json_data.GetLength() == 0 || param_data.GetLength() == 0) {
    std::cerr << "Cannot load mxnet model" << std::endl;
    std::cerr << "\tjson file: " << json_file << std::endl;
    std::cerr << "\tparams file: " << param_file << std::endl;
    exit(EXIT_FAILURE);
  }

  // Parameters
  // int dev_type = 1;  // 1: cpu, 2: gpu
  // int dev_id = 0;  // arbitrary.
  mx_uint num_input_nodes = 1;  // 1 for feedforward
  const char* input_key[1] = { "data" };
  const char** input_keys = input_key;

  // Create Predictor
  int res = MXPredCreate(static_cast<const char*>(json_data.GetBuffer()),
               static_cast<const char*>(param_data.GetBuffer()),
               static_cast<int>(param_data.GetLength()),
               dev_type,
               dev_id,
               num_input_nodes,
               input_keys,
               shape.input_shape_indptr,
               shape.input_shape_data,
               pred_hnd);
  assert(res==0);
}

void mxHandleReshape(PredictorHandle  handle,   /* mxnet model handle */
                     mxInputShape     shape,    /* new shape */
                     PredictorHandle* out) {    /* new hanlde */
  
  mx_uint num_input_nodes = 1;  // 1 for feedforward
  const char* input_key[1] = { "data" };
  const char** input_keys = input_key;
  // mxnet 1.2.1 has bug in c api MXPredReshape
  // must merge bug fix commit 31244922d7861fef484fc315e633b4a0cc7e6ff6
  int res = MXPredReshape(num_input_nodes,
                input_keys,
                shape.input_shape_indptr,
                shape.input_shape_data,
                handle,
                out );
  assert(res == 0);
}
