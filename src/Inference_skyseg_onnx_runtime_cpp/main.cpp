#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

void preprocess(const cv::Mat &frame, std::vector<float> &input_tensor_values)
{
    cv::Mat resized_frame, float_frame;
    cv::resize(frame, resized_frame, cv::Size(256, 256));
    resized_frame.convertTo(float_frame, CV_32F, 1.0 / 255.0);

    cv::Mat normalized_frame = (float_frame - cv::Scalar(0.485, 0.456, 0.406)) / cv::Scalar(0.229, 0.224, 0.225);
    std::vector<cv::Mat> chw_frames(3);
    cv::split(normalized_frame, chw_frames);

    for (int i = 0; i < 3; ++i)
    {
        input_tensor_values.insert(input_tensor_values.end(), (float *)chw_frames[i].datastart, (float *)chw_frames[i].dataend);
    }
}

int main()
{
    const std::string model_path = "/Users/tristanleduc/Documents/Code_projects/stereolabs_technical_test/models/sky_segmentation_model_25_epochs.onnx";
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SkySegmentation");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::Session session(env, model_path.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    // Gettting the input and output names
    auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
    auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    const char *input_name = input_name_ptr.get();
    const char *output_name = output_name_ptr.get();

    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open camera." << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break;

        std::vector<float> input_tensor_values;
        preprocess(frame, input_tensor_values);

        std::array<int64_t, 4> input_shape{1, 3, 256, 256};
        size_t input_tensor_size = input_tensor_values.size();
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_shape.data(), input_shape.size());

        std::vector<Ort::Value> ort_inputs;
        ort_inputs.emplace_back(std::move(input_tensor));
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, &input_name, ort_inputs.data(), 1, &output_name, 1);

        float *raw_output = output_tensors.front().GetTensorMutableData<float>();
        cv::Mat output(256, 256, CV_32F, raw_output);
        cv::Mat sigmoid_output;
        cv::exp(-output, sigmoid_output);
        sigmoid_output = 1.0 / (1.0 + sigmoid_output);

        cv::Mat binary_mask;
        cv::threshold(sigmoid_output, binary_mask, 0.5, 1.0, cv::THRESH_BINARY);
        binary_mask.convertTo(binary_mask, CV_8U, 255);
        cv::resize(binary_mask, binary_mask, frame.size());

        cv::Mat red_mask = cv::Mat::zeros(frame.size(), frame.type());
        std::vector<cv::Mat> channels;
        cv::split(red_mask, channels);
        channels[2] = binary_mask;
        cv::merge(channels, red_mask);

        cv::Mat overlay;
        cv::addWeighted(frame, 0.8, red_mask, 0.5, 0.0, overlay);

        cv::imshow("Sky Segmentation", overlay);
        if (cv::waitKey(1) >= 0)
            break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}