#ifndef __SSH_DETECTOR__
#define __SSH_DETECTOR__

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

class SSH{
public:
    SSH(const std::string& model_path, int w, int h);
    ~SSH();
    void detect(cv::Mat& img, std::vector<cv::Rect2f>  & target_boxes, 
                              std::vector<cv::Point2f> & target_landmarks);

private:

    const float pixel_means[3] = {0.406, 0.456, 0.485};
    const float pixel_stds[3] = {0.225, 0.224, 0.229};
    const float pixel_scale = 255.0;

    std::map<int, std::vector<cv::Rect2f>> anchors_fpn;
    std::map<int,int>                      num_anchors;

    const int   rpn_pre_nms_top_n = 1000;
    const float nms_threshold = 0.3;
    const float threshold = 0.5;

    void * handle;

    // nms();

};

#endif