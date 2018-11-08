#include "ssh_detector.h"

#include <iostream>
#include <iomanip>
#include <sys/time.h>

#include "opencv2/dnn.hpp"

#include "anchors.h"

static float getElapse(struct timeval *tv1,struct timeval *tv2)
{
    float t = 0.0f;
    if (tv1->tv_sec == tv2->tv_sec)
        t = (tv2->tv_usec - tv1->tv_usec)/1000.0f;
    else
        t = ((tv2->tv_sec - tv1->tv_sec) * 1000 * 1000 + tv2->tv_usec - tv1->tv_usec)/1000.0f;
    return t;
}

SSH::SSH(const std::string& model_path, int w, int h) {

    generate_anchors_fpn(anchors_fpn, num_anchors);

    std::string model  = model_path + ".caffemodel";
    std::string config = model_path + ".prototxt";
    // Load model handle
    auto net = new cv::dnn::Net();

    *net= cv::dnn::readNetFromCaffe(config, model);

    std::cout << "opencv dnn read caffe net \n";

    net->setPreferableBackend(0);
    net->setPreferableTarget(0);

    handle = (void *) net;
}


SSH::~SSH(){
    // Free model handle
    delete (cv::dnn::Net *) handle;
    std::cout << "release opencv dnn inference handle!\n";
}

void SSH::detect(cv::Mat& im, std::vector<cv::Rect2f>  & target_boxes, 
                              std::vector<cv::Point2f> & target_landmarks) {


    cv::Mat blob;
    cv::dnn::blobFromImage(im, blob, 1.0, cv::Size(224,224) );

    cv::dnn::Net * net = ( cv::dnn::Net *) handle;

    net->setInput(blob);
    cv::Mat feature = net->forward("fc1000");

    std::cout << "cols: "<< feature.cols << " "
              << "rows: "<< feature.rows << " " 
              << "channels: "<<  feature.channels()
              << std::endl;

    /*
    struct timeval  tv1,tv2;
    float sum_time = 0;
    gettimeofday(&tv1,NULL);
    // Forward Inference 

    gettimeofday(&tv2,NULL);
    sum_time += getElapse(&tv1, &tv2);
    // Inference
    std::vector<float> scores;
    std::vector<cv::Rect2f> boxes;
    std::vector<cv::Point2f> landmarks;

    for(int i=0; i< 3; i++) {
        std::vector<int> shape;
        std::vector<float> scores1;
        int index;
        index = i*3;

        gettimeofday(&tv1,NULL);

        gettimeofday(&tv2,NULL);
        sum_time += getElapse(&tv1, &tv2);

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

        gettimeofday(&tv1,NULL);

        gettimeofday(&tv2,NULL);
        sum_time += getElapse(&tv1, &tv2);

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

        gettimeofday(&tv1,NULL);

        gettimeofday(&tv2,NULL);
        sum_time += getElapse(&tv1, &tv2);

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
    // for(size_t i=0;i<order.size();i++) std::cout << "order: " << order[i] << "\n";

    std::vector<float> order_scores;
    std::vector<cv::Rect2f> order_boxes;
    std::vector<cv::Point2f> order_landmarks;

    sort_with_idx(scores, order_scores, order, 1);
    sort_with_idx(boxes,  order_boxes, order, 1);
    sort_with_idx(landmarks, order_landmarks, order, 5);

    // for(auto & s: order_scores) std::cout << "scores: " << s << "\n";
    // for(auto & b: order_boxes) std::cout  << "boxes: "  << b << "\n";
    // for(auto & l: order_landmarks) std::cout << "landmarks: " << l << "\n";

    std::vector<bool> keep(order_scores.size(),false);
    nms(order_scores, order_boxes, keep, nms_threshold);
    // for(size_t i=0;i<keep.size();i++) if(keep[i]) std::cout << "keep: " << i << "\n";

    tensor_slice(order_boxes,     target_boxes,     keep, 1);
    tensor_slice(order_landmarks, target_landmarks, keep, 5);

    std::cout << "mxnet infer, time eclipsed: " << sum_time  << " ms\n";

    */
}