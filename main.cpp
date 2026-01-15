#include <iostream>
#include <filesystem>
#include <vector>
#include <thread>
#include <atomic>//For safe counter
#include <queue>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// Blocking queue
template<typename T>
class BlockingQueue {
private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable not_empty_;  // Signal when queue has items
    std::condition_variable not_full_;   // Signal when queue has space
    size_t max_size_;

public:
    BlockingQueue(size_t max_size = 100) : max_size_(max_size) {}

    // Add item to queue,and block if full
    void put(const T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        while (queue_.size() >= max_size_) {
            not_full_.wait(lock);
        }
        queue_.push(item);
        not_empty_.notify_one();
    }

    // Remove item from queue and block if empty
    T take() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (queue_.empty()) {
            not_empty_.wait(lock);
        }
        T item = queue_.front();
        queue_.pop();
        not_full_.notify_one();
        return item;
    }

};

// Convert image to cartoon style
cv::Mat cartoonizeImage(const cv::Mat& img) {
    cv::Mat gray, edges, color, result;

    // 1. Convert to grayscale
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // 2. Reduce noise with median filter
    cv::medianBlur(gray, gray, 9);

    // 3. Detect edges using Laplacian
    cv::Laplacian(gray, edges, CV_8U, 5);

    // 4. Binarize edges
    cv::threshold(edges, edges, 120, 255, cv::THRESH_BINARY_INV);

    // 5. Smooth colors while preserving edges
    cv::bilateralFilter(img, color, 5, 50, 50);

    // 6. Combine smoothed image with edges
    cv::Mat edgesBGR;
    cv::cvtColor(edges, edgesBGR, cv::COLOR_GRAY2BGR);
    cv::bitwise_and(color, edgesBGR, result);

    return result;
}

// Check if input folder exists
bool checkInputFolder(const std::string& inputDir) {
    if (!fs::exists(inputDir)) {
        std::cout << "Error: Input folder '" << inputDir << "' not found!" << std::endl;
        return false;
    }
    return true;
}

// Create output folder if it doesn't exist
bool createOutputFolder(const std::string& outputDir) {
    if (!fs::exists(outputDir)) {
        if (fs::create_directory(outputDir)) {
            std::cout << "Created output folder: " << outputDir << std::endl;
            return true;
        }
        return false;
    }
    return true;
}

// Get all JPG images from input folder
std::vector<fs::path> getImageFiles(const std::string& inputDir) {
    std::vector<fs::path> imageFiles;
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".JPG" || ext == ".JPEG") {
                imageFiles.push_back(entry.path());
            }
        }
    }
    return imageFiles;
}

// Verify that we found some images
bool hasImageFiles(const std::vector<fs::path>& imageFiles, const std::string& inputDir) {
    if (imageFiles.empty()) {
        std::cout << "Error: No JPG images found in '" << inputDir << "' folder!" << std::endl;
        std::cout << "Supported formats: .jpg, .jpeg, .JPG, .JPEG" << std::endl;
        return false;
    }
    return true;
}

// Producer thread: creates tasks and adds them to queue
void producer(BlockingQueue<std::string>& queue,
              const std::vector<fs::path>& imageFiles,
              const std::string& outputDir) {
    std::cout << "Producer started..." << std::endl;

    // Create task for each image: "input_path|output_path"
    for (const auto& imagePath : imageFiles) {
        std::string task = imagePath.string() + "|" +
                          outputDir + "/" + imagePath.filename().string();
        queue.put(task);
        std::cout << "Producer: Added task for " << imagePath.filename() << std::endl;
    }
    for (int i = 0; i < 4; i++) {
        queue.put("STOP|STOP");
    }
    std::cout << "Producer finished, added " << imageFiles.size() << " tasks total" << std::endl;
}

// Consumer thread: processes images from queue
void consumer(BlockingQueue<std::string>& queue,
              std::atomic<int>& processed,
              std::atomic<int>& failed,
              int consumer_id) {
    std::cout << "Consumer " << consumer_id << " started..." << std::endl;

    while (true) {
        // Get next task from queue (blocks if empty)
        std::string task = queue.take();
        if (task == "STOP|STOP") {
            break;
        }

        // Parse task string
        size_t split_pos = task.find('|');
        std::string input_path = task.substr(0, split_pos);
        std::string output_path = task.substr(split_pos + 1);
        std::string filename = fs::path(input_path).filename().string();

        std::cout << "Consumer " << consumer_id << " processing: " << filename << " ... ";

        // Read image
        cv::Mat image = cv::imread(input_path);
        if (image.empty()) {
            std::cout << "Failed (cannot read image)" << std::endl;
            failed++;
            continue;
        }

        // Process image
        cv::Mat cartoon = cartoonizeImage(image);

        // Save result
        if (cv::imwrite(output_path, cartoon)) {
            std::cout << "Success -> " << output_path << std::endl;
            processed++;
        } else {
            std::cout << "Failed (save error)" << std::endl;
            failed++;
        }
    }

    std::cout << "Consumer " << consumer_id << " finished" << std::endl;
}

int main() {
    // Reduce OpenCV log output
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

    // Folder paths
    std::string inputDir = "input";
    std::string outputDir = "output";

    // Check
    if (!checkInputFolder(inputDir)) return -1;
    if (!createOutputFolder(outputDir)) {
        std::cout << "Error: Cannot create output folder!" << std::endl;
        return -1;
    }

    // Get list of images
    std::vector<fs::path> imageFiles = getImageFiles(inputDir);
    if (!hasImageFiles(imageFiles, inputDir)) return -1;
    std::cout << "Found " << imageFiles.size() << " images to process" << std::endl;

    // Create shared resources
    BlockingQueue<std::string> taskQueue(100);
    std::atomic<int> processed(0);
    std::atomic<int> failed(0);

    // Start producer-consumer threads
    std::cout << "\nStart...\n" << std::endl;

    // Producer thread
    std::thread producer_thread(producer, std::ref(taskQueue),
                                std::ref(imageFiles), std::ref(outputDir));

    // Consumer threads
    std::vector<std::thread> consumer_threads;
    for (int i = 0; i < 4; i++) {
        consumer_threads.emplace_back(consumer, std::ref(taskQueue),
                                      std::ref(processed), std::ref(failed), i + 1);
    }

    // Wait for all threads to complete
    producer_thread.join();
    std::cout << "\nProducer finished, waiting for consumers..." << std::endl;

    for (auto& t : consumer_threads) {
        t.join();
    }

    // Final statistics
    std::cout << "\nProcessing Completed" << std::endl;
    std::cout << "Total images: " << imageFiles.size() << std::endl;
    std::cout << "Successfully processed: " << processed << " images" << std::endl;
    std::cout << "Failed: " << failed << " images" << std::endl;

    return 0;
}