#ifndef __SSH_DETECTOR__
#define __SSH_DETECTOR__

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

class SSH{
public:
    SSH(const std::string& model_path, float threshold = 0.95, float nms_threshold = 0.3, bool infer_blur_score=false);
    SSH(const std::string& model_path, std::vector<float> means, std::vector<float> stds, float scale,
                                       float threshold = 0.95, float nms_threshold = 0.3, bool infer_blur_score=false);
    ~SSH();
    void detect(cv::Mat& img, std::vector<cv::Rect2f>  & target_boxes,
                              std::vector<cv::Point2f> & target_landmarks,
                              std::vector<float>       & target_scores);
    void detect(cv::Mat& img, std::vector<cv::Rect2f>  & target_boxes,
                              std::vector<cv::Point2f> & target_landmarks,
                              std::vector<float>       & target_scores,
                              std::vector<float>       & target_blur_scores);
private:

    float pixel_means[3] = {0.406, 0.456, 0.485};
    float pixel_stds[3]  = {0.225, 0.224, 0.229};
    float pixel_scale = 255.0;

    std::map<int, std::vector<cv::Rect2f>> anchors_fpn;
    std::map<int,int>                      num_anchors;

    // const int   rpn_pre_nms_top_n = 1000;
    float nms_threshold = 0.3;
    float threshold = 0.95;

    bool infer_blur_score = false;

    void * handle;
    void * infer_buff = nullptr;
    
    // int w;
    // int h;
    // nms();

};

#endif