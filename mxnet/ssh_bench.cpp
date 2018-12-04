#include "ssh_detector.h"

#include <stdlib.h>
#include <sys/time.h>

static float getElapse(struct timeval *tv1,struct timeval *tv2)
{
    float t = 0.0f;
    if (tv1->tv_sec == tv2->tv_sec)
        t = (tv2->tv_usec - tv1->tv_usec)/1000.0f;
    else
        t = ((tv2->tv_sec - tv1->tv_sec) * 1000 * 1000 + tv2->tv_usec - tv1->tv_usec)/1000.0f;
    return t;
}

int main(int argc, char* argv[]) {

  const std::string keys =
      "{help h usage ? |                | print this message   }"
      "{model          |../../model     | path to ssh model    }"
      "{threshold      |0.95            | threshold for detect score }"
      "{nms_threshold  |0.3             | nms_threshold for ssh detector }"      
      "{output         |./output        | path to save detect output  }"
      "{input          |../../../video  | path to input video file  }"      
      "{@video         |camera-244-crop-8p.mov     | input video file }"
  ;

  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("ssh detector benchmark");
  if (parser.has("help")) {
      parser.printMessage();
      return 0;
  }

  if (!parser.check()) {
      parser.printErrors();
      return EXIT_FAILURE;
  }

  std::string model_path  = parser.get<std::string>("model");
  float threshold         = parser.get<float>("threshold");
  float nms_threshold     = parser.get<float>("nms_threshold");
  std::string output_path = parser.get<std::string>("output");
  std::string input_path  = parser.get<std::string>("input");
  std::string video_file  = parser.get<std::string>(0);
  std::string video_path  = input_path + "/" + video_file;

  std::string output_folder = output_path + "/" + video_file;
  std::string cmd = "mkdir -p " + output_folder;
  system(cmd.c_str());

  cv::VideoCapture capture(video_path);
  cv::Mat frame;
  int frame_count = 1;
  capture >> frame;
  if(!frame.data) {
      std::cout<< "read first frame failed!";
      exit(1);
  }
  SSH det(model_path, frame.cols, frame.rows, threshold, nms_threshold);

  std::cout << "frame resolution: " << frame.cols << "*" << frame.rows << "\n";

  std::vector<cv::Rect2f> boxes;
  std::vector<cv::Point2f> landmarks;
  std::vector<float> scores;

  struct timeval  tv1,tv2;

  while(1){
      capture >> frame;
      frame_count++;
      if(!frame.data)   break;
      if(frame_count%15!=0) continue;
      
      gettimeofday(&tv1,NULL);
      det.detect(frame,boxes,landmarks,scores);
      gettimeofday(&tv2,NULL);
      std::cout << "detected one frame, time eclipsed: " <<  getElapse(&tv1, &tv2) << " ms\n";

      for(auto & b: boxes)
        cv::rectangle( frame, b, cv::Scalar( 255, 0, 0 ), 2, 1 );
        
      for(int i=0;i<boxes.size();i++){
        cv::rectangle( frame, boxes[i], cv::Scalar( 255, 0, 0 ), 2, 1 );
        cv::Point middleHighPoint = cv::Point(boxes[i].x+boxes[i].width/2, boxes[i].y);
        std::string text = std::to_string(scores[i]);
        cv::putText(frame, text, middleHighPoint, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
      }

      for(auto & p: landmarks)
        cv::drawMarker(frame, p,  cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 10, 1);    

      if(boxes.size()>0)
        cv::imwrite( output_folder + "/" + std::to_string(frame_count) + ".jpg", frame);

  }

}