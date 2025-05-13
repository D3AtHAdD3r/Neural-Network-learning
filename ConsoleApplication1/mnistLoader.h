#pragma once
#include <fstream>
#include <vector>
#include <stdexcept>
#include <Eigen/Dense>
#include <string>
#include <iomanip>

/*
Useage:
int main() {
    try {
        std::string train_images = "data/train-images-idx3-ubyte";
        std::string train_labels = "data/train-labels-idx1-ubyte";
        std::string test_images = "data/t10k-images-idx3-ubyte";
        std::string test_labels = "data/t10k-labels-idx1-ubyte";

        // Load smaller dataset for testing
        auto training_data = load_mnist_training(train_images, train_labels, 1000);
        auto test_data = load_mnist_test(test_images, test_labels, 1000);

        std::cout << "Training data size: " << training_data.size() << std::endl;
        std::cout << "Test data size: " << test_data.size() << std::endl;
    }
    catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation error: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}
*/


// Helper function to reverse bytes (MNIST files are big-endian)
int32_t reverse_int(int32_t i) {
    return ((i & 0xFF) << 24) | ((i >> 8 & 0xFF) << 16) | ((i >> 16 & 0xFF) << 8) | (i >> 24 & 0xFF);
}

std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> load_mnist_training(
    const std::string& image_path, const std::string& label_path, int max_images = -1) {
    std::ifstream image_file(image_path, std::ios::binary);
    std::ifstream label_file(label_path, std::ios::binary);
    if (!image_file.is_open() || !label_file.is_open()) {
        throw std::runtime_error("Failed to open MNIST training files: " + image_path + " or " + label_path);
    }

    // Read and verify image file header
    int32_t magic_number, num_images, num_rows, num_cols;
    image_file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);
    if (magic_number != 2051) {
        throw std::runtime_error("Invalid image magic number: " + std::to_string(magic_number) + ", expected 2051");
    }
    image_file.read((char*)&num_images, sizeof(num_images));
    num_images = reverse_int(num_images);
    image_file.read((char*)&num_rows, sizeof(num_rows));
    num_rows = reverse_int(num_rows);
    image_file.read((char*)&num_cols, sizeof(num_cols));
    num_cols = reverse_int(num_cols);

    // Debug header
    /*std::cout << "Image File Header:\n";
    std::cout << "Magic Number: " << magic_number << "\n";
    std::cout << "Number of Images: " << num_images << "\n";
    std::cout << "Rows: " << num_rows << "\n";
    std::cout << "Cols: " << num_cols << "\n";*/

    // Read and verify label file header
    int32_t label_magic, num_labels;
    label_file.read((char*)&label_magic, sizeof(label_magic));
    label_magic = reverse_int(label_magic);
    if (label_magic != 2049) {
        throw std::runtime_error("Invalid label magic number: " + std::to_string(label_magic) + ", expected 2049");
    }
    label_file.read((char*)&num_labels, sizeof(num_labels));
    num_labels = reverse_int(num_labels);

    /*std::cout << "Label File Header:\n";
    std::cout << "Magic Number: " << label_magic << "\n";
    std::cout << "Number of Labels: " << num_labels << "\n";*/

    if (num_images != num_labels) {
        throw std::runtime_error("Mismatch between number of images and labels");
    }

    if (max_images > 0 && max_images < num_images) {
        num_images = max_images;
    }

    const int image_size = num_rows * num_cols;
    if (image_size != 784) {
        throw std::runtime_error("Invalid image size: " + std::to_string(image_size) + ", expected 784");
    }

    std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> data;
    data.reserve(std::min(num_images, 10000));

    std::vector<unsigned char> image_buffer(image_size);
    for (int i = 0; i < num_images; ++i) {
        image_file.read((char*)image_buffer.data(), image_size);
        if (!image_file) {
            throw std::runtime_error("Failed to read image " + std::to_string(i));
        }

        Eigen::VectorXd image(image_size);
        for (int j = 0; j < image_size; ++j) {
            image(j) = static_cast<double>(image_buffer[j]) / 255.0;
        }

        unsigned char label;
        label_file.read((char*)&label, 1);
        if (!label_file) {
            throw std::runtime_error("Failed to read label " + std::to_string(i));
        }
        Eigen::VectorXd one_hot = Eigen::VectorXd::Zero(10);
        one_hot(label) = 1.0;

        data.emplace_back(image, one_hot);
    }

    image_file.close();
    label_file.close();
    return data;
}

std::vector<std::pair<Eigen::VectorXd, int>> load_mnist_test(
    const std::string& image_path, const std::string& label_path, int max_images = -1) {
    std::ifstream image_file(image_path, std::ios::binary);
    std::ifstream label_file(label_path, std::ios::binary);
    if (!image_file.is_open() || !label_file.is_open()) {
        throw std::runtime_error("Failed to open MNIST test files: " + image_path + " or " + label_path);
    }

    int32_t magic_number, num_images, num_rows, num_cols;
    image_file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);
    if (magic_number != 2051) {
        throw std::runtime_error("Invalid image magic number: " + std::to_string(magic_number) + ", expected 2051");
    }
    image_file.read((char*)&num_images, sizeof(num_images));
    num_images = reverse_int(num_images);
    image_file.read((char*)&num_rows, sizeof(num_rows));
    num_rows = reverse_int(num_rows);
    image_file.read((char*)&num_cols, sizeof(num_cols));
    num_cols = reverse_int(num_cols);

    int32_t label_magic, num_labels;
    label_file.read((char*)&label_magic, sizeof(label_magic));
    label_magic = reverse_int(label_magic);
    if (label_magic != 2049) {
        throw std::runtime_error("Invalid label magic number: " + std::to_string(label_magic) + ", expected 2049");
    }
    label_file.read((char*)&num_labels, sizeof(num_labels));
    num_labels = reverse_int(num_labels);

    if (num_images != num_labels) {
        throw std::runtime_error("Mismatch between number of images and labels");
    }

    if (max_images > 0 && max_images < num_images) {
        num_images = max_images;
    }

    const int image_size = num_rows * num_cols;
    if (image_size != 784) {
        throw std::runtime_error("Invalid image size: " + std::to_string(image_size) + ", expected 784");
    }

    std::vector<std::pair<Eigen::VectorXd, int>> data;
    data.reserve(std::min(num_images, 5000));

    std::vector<unsigned char> image_buffer(image_size);
    for (int i = 0; i < num_images; ++i) {
        image_file.read((char*)image_buffer.data(), image_size);
        if (!image_file) {
            throw std::runtime_error("Failed to read image " + std::to_string(i));
        }

        Eigen::VectorXd image(image_size);
        for (int j = 0; j < image_size; ++j) {
            image(j) = static_cast<double>(image_buffer[j]) / 255.0;
        }

        unsigned char label;
        label_file.read((char*)&label, 1);
        if (!label_file) {
            throw std::runtime_error("Failed to read label " + std::to_string(i));
        }
        int label_int = static_cast<int>(label);

        data.emplace_back(image, label_int);
    }

    image_file.close();
    label_file.close();
    return data;
}



//Helpers
void display_training_data(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& training_data, int num_samples = 5) {
    if (training_data.empty()) {
        std::cout << "No training data to display." << std::endl;
        return;
    }

    // Limit number of samples to display
    num_samples = std::min(num_samples, static_cast<int>(training_data.size()));

    for (int i = 0; i < num_samples; ++i) {
        const auto& [image, label] = training_data[i];

        //For debugging
        int imgSize = image.size();
        int labelSize = label.size();

        // Verify image and label sizes
        if (image.size() != 784 || label.size() != 10) {
            std::cout << "Invalid image or label size for sample " << i << std::endl;
            continue;
        }

        // Print sample info
        std::cout << "\nSample " << i + 1 << ":\n";

        // Display the 28x28 image as a grid of normalized pixel values
        std::cout << "Image (Normalized Pixel Values [0,1]):\n";
        for (int row = 0; row < 28; ++row) {
            for (int col = 0; col < 28; ++col) {
                int pixel_idx = row * 28 + col;
                double pixel = image(pixel_idx);
                // Print pixel value to 2 decimal places, fixed width
                std::cout << std::fixed << std::setprecision(2) << std::setw(5) << pixel << " ";
            }
            std::cout << "\n";
        }

        // Display the one-hot label vector
        std::cout << "One-Hot Label Vector: ";
        for (int j = 0; j < 10; ++j) {
            // Print 1 if value is >0.5, else 0
            std::cout << (label(j) > 0.5 ? "1" : "0");
        }
        std::cout << " (Digit: ";
        for (int j = 0; j < 10; ++j) {
            if (label(j) > 0.5) {
                std::cout << j;
                break;
            }
        }
        std::cout << ")\n";

        std::cout << "------------------------" << std::endl;
    }

    std::cout << "Displayed " << num_samples << " of " << training_data.size() << " samples." << std::endl;
}


void display_test_data(const std::vector<std::pair<Eigen::VectorXd, int>>& test_data, int num_samples = 5) {
    if (test_data.empty()) {
        std::cout << "No test data to display." << std::endl;
        return;
    }

    // Limit number of samples to display
    num_samples = std::min(num_samples, static_cast<int>(test_data.size()));

    for (int i = 0; i < num_samples; ++i) {
        const auto& [image, label] = test_data[i];

        // Verify image size and label
        if (image.size() != 784 || label < 0 || label > 9) {
            std::cout << "Invalid image size or label for sample " << i << std::endl;
            continue;
        }

        // Print sample info
        std::cout << "\nSample " << i + 1 << ":\n";

        // Display the 28x28 image as a grid of normalized pixel values
        std::cout << "Image (Normalized Pixel Values [0,1]):\n";
        for (int row = 0; row < 28; ++row) {
            for (int col = 0; col < 28; ++col) {
                int pixel_idx = row * 28 + col;
                double pixel = image(pixel_idx);
                // Print pixel value to 2 decimal places, fixed width
                std::cout << std::fixed << std::setprecision(2) << std::setw(5) << pixel << " ";
            }
            std::cout << "\n";
        }

        // Display the integer label
        std::cout << "Label: " << label << "\n";

        std::cout << "------------------------" << std::endl;
    }

    std::cout << "Displayed " << num_samples << " of " << test_data.size() << " samples." << std::endl;
}