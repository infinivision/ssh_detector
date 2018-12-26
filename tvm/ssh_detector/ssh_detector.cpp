#include "ssh_detector.h"
#include "anchors.h"

#include "dlpack/dlpack.h"
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

#include <fstream>
#include <iomanip>
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

static void tvmOutputOfIndex(tvm::runtime::PackedFunc handler, /* mxnet model */
                             std::vector<float> &out_data,           /* output vector */
                             std::vector<int> &out_shape,        /* output tensor shape */
                             int output_index)
{
    // Get Output Result
    tvm::runtime::NDArray res = handler(output_index);

    out_shape.assign(res->shape, res->shape + res->ndim);

    int size = 1;
    for (int i = 0; i < res->ndim; ++i) {
        size *= res->shape[i];
    }

    float* data = (float*) res->data;
    out_data.assign(data, data + size);
}

SSH::SSH(const std::string& model_path, float threshold, float nms_threshold, bool infer_blur_score)
{
    generate_anchors_fpn(anchors_fpn, num_anchors);

    tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(model_path + "/deploy_lib.so");
    std::ifstream json_in(model_path + "/deploy_graph.json");
    std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
    json_in.close();
    int device_type = kDLCPU;
    int device_id = 0;
    // get global function module for graph runtime
    tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib, device_type, device_id);
    this->handle = new tvm::runtime::Module(mod);
    std::ifstream params_in(model_path + "/deploy_param.params", std::ios::binary);
    std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    params_in.close();
    TVMByteArray params_arr;
    params_arr.data = params_data.c_str();
    params_arr.size = params_data.length();
    tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
    load_params(params_arr);

    this->nms_threshold = nms_threshold;
    this->threshold     = threshold;
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

SSH::SSH(const std::string& model_path, std::vector<float> means, std::vector<float> stds, float scale, 
                                        float threshold, float nms_threshold, bool infer_blur_score)
{
    generate_anchors_fpn(anchors_fpn, num_anchors);

    tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(model_path + "/deploy_lib.so");
    std::ifstream json_in(model_path + "/deploy_graph.json");
    std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
    json_in.close();
    int device_type = kDLCPU;
    int device_id = 0;
    // get global function module for graph runtime
    tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib, device_type, device_id);
    this->handle = new tvm::runtime::Module(mod);
    std::ifstream params_in(model_path + "/deploy_param.params", std::ios::binary);
    std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    params_in.close();
    TVMByteArray params_arr;
    params_arr.data = params_data.c_str();
    params_arr.size = params_data.length();
    tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
    load_params(params_arr);

    assert(means.size()==3);
    assert(stds.size()==3);
    for(int i=0;i<3;i++){
        pixel_means[i] = means[i];
        pixel_stds[i]  = stds[i];
    }
    pixel_scale = scale;

    this->nms_threshold = nms_threshold;
    this->threshold     = threshold;

    this->infer_blur_score = infer_blur_score;
}

SSH::~SSH()
{
    tvm::runtime::Module* mod = (tvm::runtime::Module*) handle;
    delete mod;
}

void SSH::detect(cv::Mat& im, std::vector<cv::Rect2f>  & target_boxes,
                 std::vector<cv::Point2f> & target_landmarks,
                 std::vector<float>       & target_scores,
                 std::vector<float>       & target_blur_scores) {

    assert(im.channels() == 3);

    size_t size = im.channels() * im.rows * im.cols;
    std::vector<float> image_data(size);

    float* ptr_image_r = image_data.data();
    float* ptr_image_g = image_data.data() + size / 3;
    float* ptr_image_b = image_data.data() + size / 3 * 2;

    for (int i = 0; i < im.rows; i++) {
        auto data = im.ptr<uchar>(i);

        for (int j = 0; j < im.cols; j++) {
            *ptr_image_b = static_cast<float>(((*data)/pixel_scale - pixel_means[0]) / pixel_stds[0]);
            ptr_image_b++;
            data++;
            *ptr_image_g = static_cast<float>(((*data)/pixel_scale - pixel_means[1]) / pixel_stds[1]);
            ptr_image_g++;
            data++;
            *ptr_image_r = static_cast<float>(((*data)/pixel_scale - pixel_means[2]) / pixel_stds[2]);
            ptr_image_r++;
            data++;
        }
    }

    constexpr int dtype_code = kDLFloat;
    constexpr int dtype_bits = 32;
    constexpr int dtype_lanes = 1;
    constexpr int device_type = kDLCPU;
    constexpr int device_id = 0;

    constexpr int in_ndim = 4;
    const int64_t in_shape[in_ndim] = {1, 3, im.rows, im.cols};

    DLTensor* x;
    TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);

    memcpy(x->data, &image_data[0], sizeof(image_data[0]) * image_data.size());

    tvm::runtime::Module* mod = (tvm::runtime::Module*) handle;

    // parameters in binary


#ifdef BENCH_SSH
    struct timeval  tv1,tv2;
    float sum_time = 0;
    gettimeofday(&tv1,NULL);
#endif

    tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");
    set_input("data", x);

    tvm::runtime::PackedFunc run = mod->GetFunction("run");
    run();

    tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");

#ifdef BENCH_SSH
    gettimeofday(&tv2,NULL);
    sum_time += getElapse(&tv1, &tv2);
#endif

    std::vector<float> scores;
    std::vector<cv::Rect2f> boxes;
    std::vector<float> blur_scores;
    std::vector<cv::Point2f> landmarks;

    for (int index = 0; index < 3; ++index)
    {
        std::vector<int> shape;
        std::vector<float> scores1;

#ifdef BENCH_SSH
        gettimeofday(&tv1,NULL);
#endif
        tvmOutputOfIndex(get_output, scores1, shape, index * 3);
#ifdef BENCH_SSH
        gettimeofday(&tv2,NULL);
        sum_time += getElapse(&tv1, &tv2);
#endif
        /*
        std::cout << "output shape len: " << shape.size() << "\n";
        for(auto s: shape)
            std::cout << "output shape1: " << s << "\n";
        */
        int hscore = shape[2];
        int wscore = shape[3];
        std::vector<float> scores2;
        int count = scores1.size()/2;
        scores2.resize(count);
        for(size_t i = 0; i < scores2.size(); i++)
        {
            scores2[i] = scores1[i + count];
        }
        std::vector<float> scores3;
        tensor_reshape(scores2, scores3, hscore, wscore, 1);

        std::vector<float> bbox_deltas;

#ifdef BENCH_SSH
        gettimeofday(&tv1,NULL);
#endif
        tvmOutputOfIndex(get_output, bbox_deltas, shape, index * 3 + 1);
#ifdef BENCH_SSH
        gettimeofday(&tv2,NULL);
        sum_time += getElapse(&tv1, &tv2);
#endif

        int h = shape[2];
        int w = shape[3];
        int c = 1;
        if (shape.size() >= 5)
        {
            c = shape[4];
        }
        int stride = stride_fpn[index];
        std::vector<cv::Rect2f> anchors;
        anchor_plane(h,w, stride, anchors_fpn[stride], anchors);
        /*
        std::cout << "output shape len: " << shape.size() << "\n";
        for(auto s: shape)
            std::cout << "output shape2: " << s << "\n";
        */
        std::vector<cv::Rect2f> boxes1;
        std::vector<float> blur_scores1;
        if(infer_blur_score){
            int pred_len=0;
            if(shape.size() >= 5)
                pred_len = shape[4];
            else
                pred_len = shape[1] / num_anchors[stride];
            bbox_pred_blur(anchors, boxes1, blur_scores1, bbox_deltas, pred_len, h, w, c);
        } else
            bbox_pred(anchors, boxes1, bbox_deltas, h, w, c);
        clip_boxes(boxes1, im.rows, im.cols);

        std::vector<float> landmark_deltas;

#ifdef BENCH_SSH
        gettimeofday(&tv1,NULL);
#endif
        tvmOutputOfIndex(get_output, landmark_deltas, shape, index * 3 + 2);
#ifdef BENCH_SSH
        gettimeofday(&tv2,NULL);
        sum_time += getElapse(&tv1, &tv2);
#endif

        std::vector<cv::Point2f> landmarks1;
        landmark_pred(anchors, landmarks1, landmark_deltas, h, w, c);

        std::vector<bool> idx;
        filter_threshold(idx, scores3, threshold);

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

    std::vector<float> order_scores;
    std::vector<cv::Rect2f> order_boxes;
    std::vector<float> order_blur_scores;
    std::vector<cv::Point2f> order_landmarks;

    sort_with_idx(scores, order_scores, order, 1);
    sort_with_idx(boxes,  order_boxes, order, 1);
    if(infer_blur_score) sort_with_idx(blur_scores,  order_blur_scores, order, 1);
    sort_with_idx(landmarks, order_landmarks, order, 5);

    std::vector<bool> keep(order_scores.size(),false);
    nms(order_scores, order_boxes, keep, nms_threshold);

    tensor_slice(order_boxes,     target_boxes,     keep, 1);
    if(infer_blur_score) tensor_slice(order_blur_scores,     target_blur_scores,     keep, 1);
    tensor_slice(order_landmarks, target_landmarks, keep, 5);
    tensor_slice(order_scores,    target_scores,    keep, 1);

#ifdef BENCH_SSH
    gettimeofday(&tv1,NULL);
#endif
#ifdef BENCH_SSH
    gettimeofday(&tv2,NULL);
    sum_time += getElapse(&tv1, &tv2);
    std::cout << "tvm infer, time eclipsed: " << sum_time  << " ms\n";
#endif
    /*
    for (auto& b: target_boxes)
    {
        cv::rectangle(im, b, cv::Scalar(255, 0, 0), 2, 1);
    }

    for(auto & p: target_landmarks)
    {
        cv::drawMarker(im, p,  cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 10, 1);
    }

    if (boxes.size() > 0)
        cv::imwrite("test_out.jpg", im);
    */
}

void SSH::detect(cv::Mat& im, std::vector<cv::Rect2f>  & target_boxes,
                              std::vector<cv::Point2f> & target_landmarks,
                              std::vector<float>       & target_scores){
    std::vector<float>  target_blur_scores;
    detect(im, target_boxes, target_landmarks, target_scores, target_blur_scores);
}