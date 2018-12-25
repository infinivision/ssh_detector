#include "ssh_detector_mxnet.h"

int main(int argc, char* argv[]) {

  const std::string keys =
      "{help h usage ? |                | print this message   }"
      "{model_path     |../../model     | path to ssh model  }"
      "{model_name     |mneti           | model name }"
      "{blur           |false           | if use blur scores }" 
      "{@image         |../../1.png     | input image          }"
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

  std::string model_path = parser.get<std::string>("model_path");
  std::string model_name = parser.get<std::string>("model_name");
  bool blur = parser.get<bool>("blur");
  std::string image_path = parser.get<std::string>(0);

  cv::Mat im = cv::imread(image_path, cv::IMREAD_COLOR);
  std::cout << "img resolution: " << im.cols << "*" << im.rows << "\n";
  // SSH det(model_path, im.cols, im.rows);
  std::vector<float> means;
  std::vector<float> stds;
  float scale;

  SSH det(model_path, model_name, 0.95, 0.3, blur);

  std::vector<cv::Rect2f> bboxes;
  std::vector<cv::Point2f> landmarks;
  std::vector<float> scores;
  std::vector<float> blur_scores;

  if(blur)
    det.detect(im,bboxes,landmarks,scores,blur_scores);
  else
    det.detect(im,bboxes,landmarks,scores);

  assert(bboxes.size()*5 == landmarks.size());

  for(auto & b: bboxes) std::cout << "b:" << b << "\n";
  for(auto & l: landmarks) std::cout << "l:" << l << "\n";
  for(auto & s: scores) std::cout << "s:" << s << "\n";
  for(auto & b: blur_scores) std::cout << "b:" << b << "\n";

  if(bboxes.size()==0) std::cout << "detect no face!\n";
  

}