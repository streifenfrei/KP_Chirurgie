#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include <time.h>

#include <common.h>

using namespace cv;

bool is_reading = true;
bool is_running = true;
queue <vector<Mat>> read_queue;
mutex mtx_read_queue;
vector <cv::String> inputFiles;

void inference(vart::Runner *runner, String outputPath) {
    bool saveOutput = outputPath != "NULL";
    // tensors
    auto inputTensors = runner->get_input_tensors();
    auto outputTensors = runner->get_output_tensors();
    // shapes
    auto inShape = inputTensors[0]->get_shape();
    auto outShapeSegmentation = outputTensors[1]->get_shape();
    auto outShapeLocalisation = outputTensors[0]->get_shape();
    int inSize = inputTensors[0]->get_element_num() / inShape.at(0);
    int outSizeSegmentation = outputTensors[1]->get_element_num() / outShapeSegmentation.at(0);
    int outSizeLocalisation = outputTensors[0]->get_element_num() / outShapeLocalisation.at(0);
    int batchSize = inShape.at(0);
    int inHeight = inShape.at(1);
    int inWidth = inShape.at(2);
    int segmentationHeight = outShapeSegmentation.at(1);
    int segmentationWidth = outShapeSegmentation.at(2);
    int segmentationChannels = outShapeSegmentation.at(3);
    int localisationHeight = outShapeLocalisation.at(1);
    int localisationWidth = outShapeLocalisation.at(2);
    int localisationChannels = outShapeLocalisation.at(3);
    // buffer
    std::vector <std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
    std::vector < vart::TensorBuffer * > inputsPtr, outputsPtr;
    std::vector <std::shared_ptr<xir::Tensor>> batchTensors;
    float *imageInputs = new float[inSize * batchSize];
    float *resultSegmentation = new float[outSizeSegmentation * batchSize];
    float *resultLocalisation = new float[outSizeLocalisation * batchSize];
    int index = 0;
    char stringBuffer[100] = {};
    double totalTime = 0.0;
    for (unsigned int n = 0; n < inputFiles.size(); n += batchSize) {
        cout << index << " / " << inputFiles.size() << "\n";
        cout << "Populating input data\n";
        if (inputFiles.size() < (n + batchSize)) {
            break;
        }
        for (int b = 0; b < batchSize; b++) {
            Mat image = imread(inputFiles[n + b]);
            cvtColor(image, image, COLOR_BGR2RGB);
            // crop the image
            const int x = (image.cols - inWidth) / 2;
            const int y = (image.rows - inHeight) / 2;
            image = image(Rect(x, y, inWidth, inHeight)).clone();

            // populate input data
            for (int h = 0; h < inHeight; h++) {
                for (int w = 0; w < inWidth; w++) {
                    for (int c = 0; c < 3; c++) {
                        imageInputs[b * inSize + h * inWidth * 3 + w * 3 + c] = image.at<Vec3b>(h, w)[c];
                    }
                }
            }
        }
        // input tensor refactory
        batchTensors.push_back(std::shared_ptr<xir::Tensor>(
                xir::Tensor::create(inputTensors[0]->get_name(), inShape,
                                    xir::DataType::FLOAT, sizeof(float) * 8u)));
        inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
                imageInputs, batchTensors.back().get()));
        // output tensor refactory
        batchTensors.push_back(std::shared_ptr<xir::Tensor>(
                xir::Tensor::create(outputTensors[1]->get_name(), outShapeSegmentation,
                                    xir::DataType::FLOAT, sizeof(float) * 8u)));
        outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
                resultSegmentation, batchTensors.back().get()));
        batchTensors.push_back(std::shared_ptr<xir::Tensor>(
                xir::Tensor::create(outputTensors[0]->get_name(), outShapeLocalisation,
                                    xir::DataType::FLOAT, sizeof(float) * 8u)));
        outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
                resultLocalisation, batchTensors.back().get()));

        inputsPtr.clear();
        outputsPtr.clear();
        inputsPtr.push_back(inputs[0].get());
        outputsPtr.push_back(outputs[0].get());
        outputsPtr.push_back(outputs[1].get());
        // run
        cout << "Doing inference\n";
        clock_t start, end;
        start = clock();
        auto job_id = runner->execute_async(inputsPtr, outputsPtr);
        runner->wait(job_id.first, -1);
        end = clock();
        double duration = ((double) (end - start)) / CLOCKS_PER_SEC;
        totalTime += duration;
        cout << "Duration: " << duration << " seconds\n";
        if (saveOutput) {
            // write output images
            cout << "Writing output data\n";
            for (int b = 0; b < batchSize; b++) {
                // segmentation
                for (int c = 0; c < segmentationChannels; c++) {
                    Mat segmentationMat(segmentationHeight, segmentationWidth, CV_32FC1);

                    for (int h = 0; h < segmentationHeight; h++) {
                        for (int w = 0; w < segmentationWidth; w++) {
                                 segmentationMat.at<float>(h, w) = resultSegmentation[
                                    b * outSizeSegmentation +
                                    h * segmentationWidth * segmentationChannels +
                                    w * segmentationChannels + c];
                        }
                    }
                    sprintf(stringBuffer, "%d_segmentation%d.yml", index, c);
                    FileStorage fileStorageSegmentation(outputPath + stringBuffer, FileStorage::WRITE);
                    fileStorageSegmentation << "niceMatrix" << segmentationMat;
                    cout << "Saved \"" << outputPath + stringBuffer << "\"\n";
                }
                // localisation
                for (int c = 0; c < localisationChannels; c++) {
                    Mat localisationMat(localisationHeight, localisationWidth, CV_32FC1);
                    for (int h = 0; h < localisationHeight; h++) {
                        for (int w = 0; w < localisationWidth; w++) {
                            localisationMat.at<float>(h, w) = resultLocalisation[
                                    b * outSizeLocalisation +
                                    h * localisationWidth * localisationChannels +
                                    w * localisationChannels + c];
                        }
                    }
                    sprintf(stringBuffer, "%d_localisation%d.yml", index, c);
                    FileStorage fileStorageLocalisation(outputPath + stringBuffer, FileStorage::WRITE);
                    fileStorageLocalisation << "niceMatrix" << localisationMat;
                    cout << "Saved \"" << outputPath + stringBuffer << "\"\n";
                }
                index++;
            }
        } else {
            index += batchSize;
        }
        inputs.clear();
        outputs.clear();
        cout << "\n";
    }
    cout << "Average inference time: " << (totalTime / inputFiles.size()) << "\n";
    delete imageInputs;
    delete resultSegmentation;
    delete resultLocalisation;
}


int main(int argc, char *argv[]) {
    // first argument: path to .xmodel file
    // second argument: directory containing the dataset (with / at end)
    // third argument: output path (optional, with / at end)
    // get model
    cout << "Parsing model file \n";
    auto graph = xir::Graph::deserialize(argv[1]);
    cout << "Getting model subgraph \n";
    auto subgraph = get_dpu_subgraph(graph.get());
    cout << "Creating DPU runner \n";
    auto runner = vart::Runner::create_runner(subgraph[0], "run");
    // input files
    char filePattern[80];
    strcpy(filePattern, argv[2]);
    strcat(filePattern, "*.png");
    glob(filePattern, inputFiles, true);

    String outputPath = argc < 4 ? String("NULL") : String(argv[3]);
    inference(runner.get(), outputPath);
    return 0;
}
