#include "ssh_detector.h"

#include <algorithm>
#include <iostream>
#include <iomanip>

#include "anchors.h"

#include "mxnet_model.h"

SSH::SSH(const std::string& model_path, int w, int h) {

    generate_anchors_fpn(anchors_fpn, num_anchors);

    std::string json_file = model_path + "/mneti-symbol.json";
    std::string param_file = model_path + "/mneti-0000.params";

    int channels = 3;
    InputShape input_shape (w, h, channels);
    
    // Load model
    PredictorHandle pred_hnd = nullptr;
    LoadMXNetModel(&pred_hnd, json_file, param_file, input_shape);

    handle = (void *) pred_hnd;

}

typedef struct  {
    float score;
    int index;
} s_index;

bool cmp(s_index & l, s_index & r){
    return l.score > r.score;
}

SSH::~SSH(){
    PredictorHandle pred_hnd = (PredictorHandle) handle;
    MXPredFree(pred_hnd);
}

void SSH::detect(cv::Mat& im, std::vector<cv::Rect2d>& bbox){

    assert(im.channels()==3);
    int size = im.rows * im.cols * 3;

    std::vector<mx_float> image_data(size);

    mx_float* ptr_image_r = image_data.data();
    mx_float* ptr_image_g = image_data.data() + size / 3;
    mx_float* ptr_image_b = image_data.data() + size / 3 * 2;

    for (int i = 0; i < im.rows; i++) {
        auto data = im.ptr<uchar>(i);

        for (int j = 0; j < im.cols; j++) {
            *ptr_image_b = static_cast<mx_float>(((*data)/pixel_scale - pixel_means[0]) / pixel_stds[0]);
            ptr_image_b++;
            data++;
            *ptr_image_g = static_cast<mx_float>(((*data)/pixel_scale - pixel_means[1]) / pixel_stds[1]);                
            ptr_image_g++;
            data++;
            *ptr_image_r = static_cast<mx_float>(((*data)/pixel_scale - pixel_means[2]) / pixel_stds[2]);
            ptr_image_r++;
            data++;
        }
    }
    PredictorHandle pred_hnd = (PredictorHandle) handle;
    for(size_t i = 0+size/3; i<10+size/3; i++) std::cout <<  std::setprecision(7) <<"image_data: " << image_data[i] << "\n";
    Infer(pred_hnd, image_data);

    // Inference
    std::vector<float> scores;
    std::vector<cv::Rect2f> boxes;
    std::vector<cv::Point2f> landmarks;

    for(int i=0; i< 3; i++) {
        std::vector<int> shape;
        std::vector<float> scores1;
        int index;
        index = i*3;
        OutputOfIndex(pred_hnd, scores1, shape, index);
        int hscore = shape[2];
        int wscore = shape[3];
        std::vector<float> scores2;
        int count = scores1.size()/2;
        scores2.resize(count);
        for(size_t i=0;i<scores2.size();i++){
            scores2[i] = scores1[i+count];
        }
        std::vector<float> scores3;
        tensor_reshape(scores2, scores3, hscore, wscore );

        index++;
        std::vector<float> bbox_deltas;
        OutputOfIndex(pred_hnd, bbox_deltas, shape, index);
        int h = shape[2];
        int w = shape[3];

        int stride = stride_fpn[i];
        std::vector<cv::Rect2f> anchors;
        anchor_plane(h,w, stride, anchors_fpn[stride], anchors);

        std::vector<cv::Rect2f> boxes1;
        bbox_pred(anchors, boxes1, bbox_deltas, h, w);
        clip_boxes(boxes1, im.rows, im.cols);
        // for(size_t i=0; i<5; i++) std::cout <<  "boxes1: " << boxes1[i] << "\n";
        index++;
        std::vector<float> landmark_deltas;
        OutputOfIndex(pred_hnd, landmark_deltas, shape, index);
        std::vector<cv::Point2f> landmarks1;
        landmark_pred(anchors, landmarks1, landmark_deltas, h, w);
        // for(size_t i=0; i<20; i++) std::cout <<  "landmarks: " << landmarks1[i] << "\n";

        std::vector<bool> idx;
        filter_threshold(idx, scores3, threshold);
        // for(size_t i=0; i<idx.size(); i++) if(idx[i]==true)std::cout <<  "idx: " << i << "\n";
        std::vector<float> scores4;
        tensor_slice(scores3, scores4, idx, 1);
        scores.insert(scores.end(), scores4.begin(), scores4.end());
        std::vector<cv::Rect2f> boxes2;
        tensor_slice(boxes1, boxes2, idx, 1);
        boxes.insert(boxes.end(), boxes2.begin(), boxes2.end());
        std::vector<cv::Point2f> landmarks2;
        tensor_slice(landmarks1, landmarks2, idx, 5);
        landmarks.insert(landmarks.end(), landmarks2.begin(), landmarks2.end());
    }
    
    std::vector<int> order;
    argsort(order, scores);
    for(size_t i=0;i<order.size();i++) std::cout << "order: " << order[i] << "\n";

    std::vector<float> order_scores;
    std::vector<cv::Rect2f> order_boxes;
    std::vector<cv::Point2f> order_landmarks;

    //std::cout << "scores.size() " << scores.size() << "\n";
    //std::cout << "landmarks.size() " << landmarks.size() << "\n";


    sort_with_idx(scores, order_scores, order, 1);
    sort_with_idx(boxes,  order_boxes, order, 1);
    sort_with_idx(landmarks, order_landmarks, order, 5);

    for(auto & s: order_scores) std::cout << "scores: " << s << "\n";
    for(auto & b: order_boxes) std::cout  << "boxes: "  << b << "\n";
    for(auto & l: order_landmarks) std::cout << "landmarks: " << l << "\n";

    std::vector<int> keep;
    nms(order_scores, order_boxes, keep, nms_threshold);
    for(auto & k:keep) std::cout << "keep: " << k << "\n";

}