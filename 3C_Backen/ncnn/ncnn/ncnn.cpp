
#include "layer.h"
#include "net.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <float.h>
#include <stdio.h>
#include <vector>
#include <jiexi.h>
#include <cap.h>
#include <mmsystem.h>

#define SHOWCV2 1;

using namespace std;

int main()
{

    timeBeginPeriod(1);

    ncnn::Net yolov5;
    yolov5.opt.use_vulkan_compute = true;

    if (yolov5.load_param("2_f16.param"))
        exit(-1);
    if (yolov5.load_model("2_f16.bin"))
        exit(-1);


    const int imgsize = 320;
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.5f;

    int gamex = 1920;
    int gamey = 1080;

    capture c(gamex, gamey, imgsize, imgsize, "CrossFire");

    ncnn::Mat anchors1(6);
    anchors1[0] = 10.f;
    anchors1[1] = 13.f;
    anchors1[2] = 16.f;
    anchors1[3] = 30.f;
    anchors1[4] = 33.f;
    anchors1[5] = 23.f;
    ncnn::Mat anchors2(6);
    anchors2[0] = 30.f;
    anchors2[1] = 61.f;
    anchors2[2] = 62.f;
    anchors2[3] = 45.f;
    anchors2[4] = 59.f;
    anchors2[5] = 119.f;

    ncnn::Mat anchors3(6);
    anchors3[0] = 116.f;
    anchors3[1] = 90.f;
    anchors3[2] = 156.f;
    anchors3[3] = 198.f;
    anchors3[4] = 373.f;
    anchors3[5] = 326.f;
    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };

    while (true)
    {

        auto img = (BYTE*)c.cap();
        ncnn::Mat in_pad = ncnn::Mat::from_pixels(img, ncnn::Mat::PIXEL_BGR2RGB, imgsize, imgsize);
        in_pad.substract_mean_normalize(0, norm_vals);

        auto t1 = std::chrono::steady_clock::now();

        ncnn::Extractor ex = yolov5.create_extractor();
        ex.input("images", in_pad);

        std::vector<Box> proposals;

  
        // stride 8
        {
            ncnn::Mat out;
            ex.extract("output", out);


            std::vector<Box> objects8;
            generate_proposals(anchors1, 8, in_pad, out, prob_threshold, objects8);

            proposals.insert(proposals.end(), objects8.begin(), objects8.end());
        }

        // stride 16
        {
            ncnn::Mat out;

            ex.extract("354", out);

            std::vector<Box> objects16;
            generate_proposals(anchors2, 16, in_pad, out, prob_threshold, objects16);
            proposals.insert(proposals.end(), objects16.begin(), objects16.end());
        }

        // stride 32
        {
            ncnn::Mat out;

            ex.extract("366", out);

            std::vector<Box> objects32;
            generate_proposals(anchors3, 32, in_pad, out, prob_threshold, objects32);

            proposals.insert(proposals.end(), objects32.begin(), objects32.end());
        }

        auto t2 = std::chrono::steady_clock::now();
        double dr_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        cout  <<"infer time : "<< dr_ms << "ms" << endl;

        vector<Box> newbox = cpu_nms(proposals, nms_threshold );
    
    #if SHOWCV2        

        cv::Mat a = cv::Mat(imgsize, imgsize, CV_8UC3, img);
    #endif 


        if (!newbox.empty())
        {

            for (const Box& detection : newbox)
            {

    #if SHOWCV2        

                cv::rectangle(a, cv::Point((int)detection.left, (int)detection.top), cv::Point((int)detection.right, (int)detection.bottom), cv::Scalar(0, 255, 0), 1);
    #endif 

            }
        }

    #if SHOWCV2
        cv::imshow("c", a);
        cv::waitKey(1);
    #endif
        }


    return 0;
}