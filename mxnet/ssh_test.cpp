#include "ssh_detector.h"

int main(int argc, char* argv[]) {

  const std::string keys =
      "{help h usage ? |                | print this message   }"
      "{model          |../../model     | path to ssh model  }" 
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

  std::string model_path = parser.get<std::string>("model");
  std::string image_path = parser.get<std::string>(0);

  cv::Mat im = cv::imread(image_path, cv::IMREAD_COLOR);
  SSH det(model_path, im.cols, im.rows);
  std::vector<cv::Rect2d> bbox;
  det.detect(im,bbox);

}