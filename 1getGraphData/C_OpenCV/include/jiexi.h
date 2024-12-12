#include< iostream>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>



struct Box
{
    float left, top, right, bottom, confidence;
    int class_label;

    Box() = default;

    Box(float left, float top, float right, float bottom, float confidence, int class_label)
        :left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label) {}
};


static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Box>& objects)
{
    const int num_grid = feat_blob.h;

    int num_grid_x;
    int num_grid_y;

    num_grid_y = in_pad.h / stride;
    num_grid_x = num_grid / num_grid_y;

    const int num_class = feat_blob.w - 5;

    const int num_anchors = anchors.w / 2;

    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        const ncnn::Mat feat = feat_blob.channel(q);

        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                const float* featptr = feat.row(i * num_grid_x + j);
                float box_confidence = sigmoid(featptr[4]);
                if (box_confidence >= prob_threshold)
                {
                    int class_index = 0;
                    float class_score = -FLT_MAX;
                    for (int k = 0; k < num_class; k++)
                    {
                        float score = featptr[5 + k];
                        if (score > class_score)
                        {
                            class_index = k;
                            class_score = score;
                        }
                    }
                    float confidence = sigmoid(class_score);

                    if (confidence >= prob_threshold)
                    {


                        float dx = sigmoid(featptr[0]);
                        float dy = sigmoid(featptr[1]);
                        float dw = sigmoid(featptr[2]);
                        float dh = sigmoid(featptr[3]);

                        float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                        float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                        float pb_w = pow(dw * 2.f, 2) * anchor_w;
                        float pb_h = pow(dh * 2.f, 2) * anchor_h;

                        float x0 = pb_cx - pb_w * 0.5f;
                        float y0 = pb_cy - pb_h * 0.5f;
                        float x1 = pb_cx + pb_w * 0.5f;
                        float y1 = pb_cy + pb_h * 0.5f;

                        Box obj;
                        obj.left = x0;
                        obj.top = y0;
                        obj.right = x1;
                        obj.bottom = y1;
                        obj.class_label = class_index;
                        obj.confidence = confidence;

                        objects.push_back(obj);
                    }
                }
            }
        }
    }
}




static float iou(const Box& a, const Box& b)
{
    float cleft = std::max(a.left, b.left);
    float ctop = std::max(a.top, b.top);
    float cright = std::min(a.right, b.right);
    float cbottom = std::min(a.bottom, b.bottom);

    float c_area = std::max(cright - cleft, 0.0f) * std::max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f)
        return 0.0f;
 
    float a_area = std::max(0.0f, a.right - a.left) * std::max(0.0f, a.bottom - a.top);
    float b_area = std::max(0.0f, b.right - b.left) * std::max(0.0f, b.bottom - b.top);
    return c_area / (a_area + b_area - c_area);
}









static std::vector<Box> cpu_nms(std::vector<Box>& boxes, float threshold) {

    std::sort(boxes.begin(), boxes.end(), [](std::vector<Box>::const_reference a, std::vector<Box>::const_reference b)
        {
        return a.confidence > b.confidence;
        });

    std::vector<Box> output;
    output.reserve(boxes.size());

    std::vector<bool> remove_flags(boxes.size());


    for (int i = 0; i < boxes.size(); ++i) {
        
        if (remove_flags[i]) continue;

        auto& a = boxes[i];
        output.emplace_back(a);
       

        for (int j = i + 1; j < boxes.size(); ++j) {
            if (remove_flags[j]) continue;

            auto& b = boxes[j];
            if (b.class_label == a.class_label) {
                if (iou(a, b) >= threshold)
                    remove_flags[j] = true;
            }
        }
    }

    return output;
}



