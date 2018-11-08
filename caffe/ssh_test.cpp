#include <thread>
#include <unistd.h>

#include <glog/logging.h>

#include "ssh_detector.h"

void process(SSH * det_pointer, cv::Mat im, std::string model_path, std::string mode, int count) {
    std::vector<cv::Rect2f> bboxes;
    std::vector<cv::Point2f> landmarks;
        if(mode=="single"){
            for(int i=0; i<count; i++){
                det_pointer->detect(im,bboxes,landmarks);
                google::FlushLogFiles(google::GLOG_INFO);                
            }
                   
        } else if(mode=="multi"){
            SSH det(model_path, im.cols, im.rows);
            for(int i=0; i<count; i++){
                det.detect(im,bboxes,landmarks);
                google::FlushLogFiles(google::GLOG_INFO);                
            }
        }
}

int main(int argc, char* argv[]) {

  const std::string keys =
      "{help h usage ? |                                | print this message   }"
      "{model          |../../model/caffe/ResNet-50     | path to ssh model  }"
      "{parallel       |4                               | thread num for detector }" 
      "{count          |100                             | detect times for one thread }" 
      "{mode           |single   | single : share one detector instance in multi thread; multi: one thread one instance }"       
      "{@image         |../../1.png                     | input image          }"
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

  google::InitGoogleLogging("multi-thread-ssh-detector");
  FLAGS_log_dir = "./";

  std::string model_path = parser.get<std::string>("model");
  int num = parser.get<int>("parallel");
  int count = parser.get<int>("count");
  std::string mode = parser.get<std::string>("mode");  
  std::string image_path = parser.get<std::string>(0);

  cv::Mat im = cv::imread(image_path, cv::IMREAD_COLOR);
  SSH det(model_path, im.cols, im.rows);
  std::vector<cv::Rect2f> bboxes;
  std::vector<cv::Point2f> landmarks;
  det.detect(im,bboxes,landmarks);

  std::vector<std::thread> threads;
  for(int i = 0; i<num; i++){
      std::thread t(process, &det, im, model_path, mode, count );
      threads.push_back(std::move(t));
  }
  for( size_t i=0;i<threads.size();i++){
    if(threads[i].joinable()){
        threads[i].join();
    } 
  }
  std::cout<< "all thread proccess over!\n";

  /*
  assert(bboxes.size()*5 == landmarks.size());

  for(auto & b: bboxes) std::cout << "b:" << b << "\n";
  for(auto & l: landmarks) std::cout << "l:" << l << "\n";

  if(bboxes.size()==0) std::cout << "detect no face!\n";
  */
}