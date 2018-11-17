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

#include "ssh_detector/mxnet_model.h"

#include "anchors.h"

int main(int argc, char* argv[]) {

  const std::string keys =
      "{help h usage ? |                | print this message   }"
      "{model          |../../model     | path to ssh model    }"
      "{index          |0               | mxnet output index   }"      
      "{@image         |                | input image          }"
  ;

  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("ssh detector");
  if (parser.has("help")) {
      parser.printMessage();
      return 0;
  }

  if (!parser.check()) {
      parser.printErrors();
      return EXIT_FAILURE;
  }

  std::string model_path = parser.get<std::string>("model");
  int index = parser.get<int>("index");
  std::string image_path = parser.get<std::string>(0);

  // Models path for your model, you have to modify it
  std::string json_file = model_path + "/mneti-symbol.json";
  std::string param_file = model_path + "/mneti-0000.params";

  // Image size and channels
  int width = 640;
  int height = 640;
  int channels = 3;
  mxInputShape input_shape (width, height, channels);
  
  // Load model
  PredictorHandle pred_hnd = nullptr;
  mxLoadMXNetModel(&pred_hnd, json_file, param_file, input_shape);

  // Read Image Data
  auto image_size = static_cast<std::size_t>(width * height * channels);
  std::vector<mx_float> image_data(image_size);
  mxGetImageFile(image_path, image_data);

  // Inference
  std::vector<float> data;
  mx_uint output_index = index;
  std::vector<int> shape;
  mxInfer(pred_hnd, image_data);
  mxOutputOfIndex(pred_hnd, data, shape, output_index);

  // normalize the output vector
  std::vector<float> output(data.size());
  cv::normalize(data, output);

  // Print Output Data
  // mxPrintOutputResult(output);

  // Release Predictor
  MXPredFree(pred_hnd);

  std::map<int, std::vector<cv::Rect2f>> anchors_fpn;
  std::map<int,int> num_anchors;

  generate_anchors_fpn(anchors_fpn, num_anchors);

  for(auto & element: anchors_fpn){
      std::cout << "k:" << element.first << std::endl;
      for(auto & anckor: element.second)
        std::cout << "anckor:" << anckor << std::endl;
  }
  for(auto & element: num_anchors){
      std::cout << "k:" << element.first << " v:" << element.second<< std::endl;
  }

  std::vector<float> tensor;
  std::vector<float> tensor_pad;
  tensor.push_back(1);
  tensor.push_back(2);
  tensor.push_back(3);
  tensor.push_back(4);
  tensor.push_back(5);
  tensor.push_back(6);
  tensor.push_back(7);
  tensor.push_back(8);
  tensor.push_back(9);
  tensor.push_back(10);
  tensor.push_back(11);
  tensor.push_back(12);  

  clip_pad(tensor,2,3, tensor_pad, 1,1);

  for(auto & t: tensor_pad)
    std::cout<< "t: " << t <<"\n";

  std::vector<float> tensor2;
  tensor_reshape(tensor, tensor2, 2, 3);

  for(auto & t: tensor2)
    std::cout<< "t2: " << t <<"\n";

  return EXIT_SUCCESS;
}
