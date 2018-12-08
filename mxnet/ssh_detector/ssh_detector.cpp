#include "ssh_detector.h"
#include <iostream>
#include <iomanip>
#include <sys/time.h>

#include "anchors.h"

#include "mxnet_model.h"

static float getElapse(struct timeval *tv1,struct timeval *tv2)
{
    float t = 0.0f;
    if (tv1->tv_sec == tv2->tv_sec)
        t = (tv2->tv_usec - tv1->tv_usec)/1000.0f;
    else
        t = ((tv2->tv_sec - tv1->tv_sec) * 1000 * 1000 + tv2->tv_usec - tv1->tv_usec)/1000.0f;
    return t;
}

SSH::SSH(const std::string& model_path, int w, int h, float threshold, float nms_threshold) {

    generate_anchors_fpn(anchors_fpn, num_anchors);

    std::string json_file = model_path + "/mneti-symbol.json";
    std::string param_file = model_path + "/mneti-0000.params";

    int channels = 3;
    mxInputShape input_shape (w, h, channels);
    
    // Load model
    PredictorHandle pred_hnd = nullptr;
    mxLoadMXNetModel(&pred_hnd, json_file, param_file, input_shape);

    handle = (void *) pred_hnd;

    this->w = w;
    this->h = h;
    this->threshold     = threshold;
    this->nms_threshold = nms_threshold;

}

SSH::SSH(const std::string& model_path, const std::string& model_name,
         int w, int h, float threshold, float nms_threshold,
         bool infer_blur_score) {

    generate_anchors_fpn(anchors_fpn, num_anchors);

    std::string json_file  = model_path + "/" + model_name + "-symbol.json";
    std::string param_file = model_path + "/" + model_name + "-0000.params";

    int channels = 3;
    mxInputShape input_shape (w, h, channels);
    
    // Load model
    PredictorHandle pred_hnd = nullptr;
    mxLoadMXNetModel(&pred_hnd, json_file, param_file, input_shape);

    handle = (void *) pred_hnd;

    this->w = w;
    this->h = h;
    this->threshold     = threshold;
    this->nms_threshold = nms_threshold;

    this->infer_blur_score = infer_blur_score;
    if(infer_blur_score){
        pixel_means[0] = 0;
        pixel_means[1] = 0;
        pixel_means[2] = 0;
        pixel_stds[0] = 1;
        pixel_stds[1] = 1;
        pixel_stds[2] = 1;
        pixel_scale = 1;              
    }

}


SSH::SSH(const std::string& model_path, const std::string& model_name,
         std::vector<float> means, std::vector<float> stds, float scale,
         int w, int h, float threshold, float nms_threshold,
         bool infer_blur_score) {

    generate_anchors_fpn(anchors_fpn, num_anchors);

    std::string json_file  = model_path + "/" + model_name + "-symbol.json";
    std::string param_file = model_path + "/" + model_name + "-0000.params";

    int channels = 3;
    mxInputShape input_shape (w, h, channels);
    
    // Load model
    PredictorHandle pred_hnd = nullptr;
    mxLoadMXNetModel(&pred_hnd, json_file, param_file, input_shape);

    handle = (void *) pred_hnd;

    this->w = w;
    this->h = h;
    this->threshold     = threshold;
    this->nms_threshold = nms_threshold;

    assert(means.size()==3);
    assert(stds.size()==3);

    for(int i=0;i<3;i++){
        pixel_means[i] = means[i];
        pixel_stds[i]  = stds[i];
    }
    pixel_scale = scale;

    this->infer_blur_score = infer_blur_score;
}

SSH::~SSH(){
    PredictorHandle pred_hnd = (PredictorHandle) handle;
    MXPredFree(pred_hnd);
}

void SSH::detect(cv::Mat& im, std::vector<cv::Rect2f>  & target_boxes, 
                              std::vector<cv::Point2f> & target_landmarks,
                              std::vector<float>       & target_scores,
                              std::vector<float>       & target_blur_scores) {

    assert(im.channels()==3);
    int size = im.rows * im.cols * 3;

    PredictorHandle pred_hnd = nullptr;    
    mxInputShape input_shape(im.cols, im.rows, 3);

    #ifdef BENCH_SSH
    struct timeval  tv1,tv2;
    float sum_time = 0;
    gettimeofday(&tv1,NULL);
    #endif   
    mxHandleReshape((PredictorHandle) handle, input_shape, &pred_hnd);
    #ifdef BENCH_SSH
    gettimeofday(&tv2,NULL);
    sum_time += getElapse(&tv1, &tv2);
    #endif

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
    
    // for(size_t i = 0+size/3; i<10+size/3; i++) std::cout <<  std::setprecision(7) <<"image_data: " << image_data[i] << "\n";

    #ifdef BENCH_SSH
    gettimeofday(&tv1,NULL);
    #endif
    mxInfer(pred_hnd, image_data);
    #ifdef BENCH_SSH
    gettimeofday(&tv2,NULL);
    sum_time += getElapse(&tv1, &tv2);
    #endif
    // Inference
    std::vector<float> scores;
    std::vector<cv::Rect2f> boxes;
    std::vector<float> blur_scores;
    std::vector<cv::Point2f> landmarks;

    for(int i=0; i< 3; i++) {
        std::vector<int> shape;
        std::vector<float> scores1;
        int index;
        index = i*3;
        #ifdef BENCH_SSH
        gettimeofday(&tv1,NULL);
        #endif
        mxOutputOfIndex(pred_hnd, scores1, shape, index);
        #ifdef BENCH_SSH
        gettimeofday(&tv2,NULL);
        sum_time += getElapse(&tv1, &tv2);
        #endif

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

        #ifdef BENCH_SSH
        gettimeofday(&tv1,NULL);
        #endif
        mxOutputOfIndex(pred_hnd, bbox_deltas, shape, index);
        #ifdef BENCH_SSH
        gettimeofday(&tv2,NULL);
        sum_time += getElapse(&tv1, &tv2);
        #endif

        int h = shape[2];
        int w = shape[3];

        int stride = stride_fpn[i];
        std::vector<cv::Rect2f> anchors;
        anchor_plane(h,w, stride, anchors_fpn[stride], anchors);

        std::vector<cv::Rect2f> boxes1;
        std::vector<float> blur_scores1;
        if(infer_blur_score){
            int pred_len=0;
            pred_len = shape[1] / num_anchors[stride];
            bbox_pred_blur(anchors, boxes1, blur_scores1, bbox_deltas, pred_len, h, w);
        } else
            bbox_pred(anchors, boxes1, bbox_deltas, h, w);
        clip_boxes(boxes1, im.rows, im.cols);
        // for(size_t i=0; i<5; i++) std::cout <<  "boxes1: " << boxes1[i] << "\n";
        // for(size_t i=0; i<5; i++) std::cout <<  "blur_scores1: " << blur_scores1[i] << "\n";
        index++;
        std::vector<float> landmark_deltas;

        #ifdef BENCH_SSH
        gettimeofday(&tv1,NULL);
        #endif
        mxOutputOfIndex(pred_hnd, landmark_deltas, shape, index);
        #ifdef BENCH_SSH
        gettimeofday(&tv2,NULL);
        sum_time += getElapse(&tv1, &tv2);
        #endif

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
        std::vector<float> blur_scores2;
        tensor_slice(boxes1, boxes2, idx, 1);
        if(infer_blur_score)  tensor_slice(blur_scores1, blur_scores2, idx, 1);
        boxes.insert(boxes.end(), boxes2.begin(), boxes2.end());
        if(infer_blur_score) blur_scores.insert(blur_scores.end(), blur_scores2.begin(), blur_scores2.end());
        std::vector<cv::Point2f> landmarks2;
        tensor_slice(landmarks1, landmarks2, idx, 5);
        landmarks.insert(landmarks.end(), landmarks2.begin(), landmarks2.end());
    }
    
    std::vector<int> order;
    argsort(order, scores);
    // for(size_t i=0;i<order.size();i++) std::cout << "order: " << order[i] << "\n";

    std::vector<float> order_scores;
    std::vector<cv::Rect2f> order_boxes;
    std::vector<float> order_blur_scores;
    std::vector<cv::Point2f> order_landmarks;

    sort_with_idx(scores, order_scores, order, 1);
    sort_with_idx(boxes,  order_boxes, order, 1);
    if(infer_blur_score) sort_with_idx(blur_scores,  order_blur_scores, order, 1);
    sort_with_idx(landmarks, order_landmarks, order, 5);

    // for(auto & s: order_scores) std::cout << "scores: " << s << "\n";
    // for(auto & b: order_boxes) std::cout  << "boxes: "  << b << "\n";
    // for(auto & l: order_landmarks) std::cout << "landmarks: " << l << "\n";

    std::vector<bool> keep(order_scores.size(),false);
    nms(order_scores, order_boxes, keep, nms_threshold);
    // for(size_t i=0;i<keep.size();i++) if(keep[i]) std::cout << "keep: " << i << "\n";

    tensor_slice(order_boxes,     target_boxes,     keep, 1);
    if(infer_blur_score) tensor_slice(order_blur_scores,     target_blur_scores,     keep, 1);
    tensor_slice(order_landmarks, target_landmarks, keep, 5);
    tensor_slice(order_scores,    target_scores,    keep, 1);

    #ifdef BENCH_SSH
    gettimeofday(&tv1,NULL);
    #endif
    MXPredFree(pred_hnd);
    #ifdef BENCH_SSH
    gettimeofday(&tv2,NULL);
    sum_time += getElapse(&tv1, &tv2);    
    std::cout << "mxnet infer, time eclipsed: " << sum_time  << " ms\n";
    #endif

}

void SSH::detect(cv::Mat& im, std::vector<cv::Rect2f>  & target_boxes, 
                              std::vector<cv::Point2f> & target_landmarks,
                              std::vector<float>       & target_scores){
    std::vector<float>  target_blur_scores;
    detect(im, target_boxes, target_landmarks, target_scores, target_blur_scores);
}