#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

#define INPUT_NODES 784  // 28*28 pixels
#define HIDDEN_NODES 256 // Number of hidden nodes
#define OUTPUT_NODES 10  // 10 digits (0-9)

#define NUM_TRAINING_IMAGES 8000
#define NUM_TEST_IMAGES 1000

#define NUMBER_OF_EPOCHS 10

// CUDA block size
#define BLOCK_SIZE 256

// Host data
double training_images[NUM_TRAINING_IMAGES][INPUT_NODES];
double training_labels[NUM_TRAINING_IMAGES][OUTPUT_NODES];
double test_images[NUM_TEST_IMAGES][INPUT_NODES];
double test_labels[NUM_TEST_IMAGES][OUTPUT_NODES];

// Host weights and biases
double weight1[INPUT_NODES][HIDDEN_NODES];
double weight2[HIDDEN_NODES][OUTPUT_NODES];
double bias1[HIDDEN_NODES];
double bias2[OUTPUT_NODES];

// Device pointers
double *d_input, *d_target;
double *d_weight1, *d_weight2;
double *d_bias1, *d_bias2;
double *d_hidden, *d_output;
double *d_delta_out, *d_delta_hidden;

int correct_predictions;
int forward_prob_output;

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

void load_mnist()
{
    // Open the training images file
    FILE *training_images_file = fopen("mnist_train_images.bin", "rb");
    if (training_images_file == NULL)
    {
        printf("Error opening training images file\n");
        exit(1);
    }

    // Open the training labels file
    FILE *training_labels_file = fopen("mnist_train_labels.bin", "rb");
    if (training_labels_file == NULL)
    {
        printf("Error opening training labels file\n");
        exit(1);
    }

    // Open the test images file
    FILE *test_images_file = fopen("mnist_test_images.bin", "rb");
    if (test_images_file == NULL)
    {
        printf("Error opening test images file\n");
        exit(1);
    }

    // Open the test labels file
    FILE *test_labels_file = fopen("mnist_test_labels.bin", "rb");
    if (test_labels_file == NULL)
    {
        printf("Error opening test labels file\n");
        exit(1);
    }

    unsigned char t;
    for (int i = 0; i < 8; i++)
    {
        fread(&t, sizeof(unsigned char), 1, training_images_file);
    }

    // Read the training images
    for (int i = 0; i < NUM_TRAINING_IMAGES; i++)
    {
        for (int j = 0; j < INPUT_NODES; j++)
        {
            unsigned char pixel;
            fread(&pixel, sizeof(unsigned char), 1, training_images_file);
            training_images[i][j] = (double)pixel / 255.0;
        }
    }
    for (int i = 0; i < 8; i++)
    {
        fread(&t, sizeof(unsigned char), 1, training_labels_file);
    }

    // Read the training labels
    for (int i = 0; i < NUM_TRAINING_IMAGES; i++)
    {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, training_labels_file);
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            if (j == label)
            {
                training_labels[i][j] = 1;
            }
            else
            {
                training_labels[i][j] = 0;
            }
        }
    }

    // Read the test images
    for (int i = 0; i < 8; i++)
    {
        fread(&t, sizeof(unsigned char), 1, test_images_file);
    }

    for (int i = 0; i < NUM_TEST_IMAGES; i++)
    {
        for (int j = 0; j < INPUT_NODES; j++)
        {
            unsigned char pixel;
            fread(&pixel, sizeof(unsigned char), 1, test_images_file);
            test_images[i][j] = (double)pixel / 255.0;
        }
    }

    // Read the test labels
    for (int i = 0; i < 8; i++)
    {
        fread(&t, sizeof(unsigned char), 1, test_labels_file);
    }

    for (int i = 0; i < NUM_TEST_IMAGES; i++)
    {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, test_labels_file);
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            if (j == label)
            {
                test_labels[i][j] = 1;
            }
            else
            {
                test_labels[i][j] = 0;
            }
        }
    }

    // Close the files
    fclose(training_images_file);
    fclose(training_labels_file);
    fclose(test_images_file);
    fclose(test_labels_file);
}

__device__ double sigmoid_cuda(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

__device__ double reLU_cuda(double x)
{
    return x > 0.0 ? x : 0.0;
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double reLU(double x)
{
    return x > 0.0 ? x : 0.0;
}

int max_index(double arr[], int size)
{
    int max_i = 0;
    for (int i = 1; i < size; i++)
    {
        if (arr[i] > arr[max_i])
        {
            max_i = i;
        }
    }
    return max_i;
}

// CUDA kernel for forward pass: input to hidden layer
__global__ void forward_input_to_hidden(double *input, double *weight1, double *bias1, double *hidden)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < HIDDEN_NODES)
    {
        double sum = bias1[idx];
        for (int j = 0; j < INPUT_NODES; j++)
        {
            sum += input[j] * weight1[j * HIDDEN_NODES + idx];
        }
        hidden[idx] = reLU_cuda(sum);
    }
}

// CUDA kernel for forward pass: hidden to output layer
__global__ void forward_hidden_to_output(double *hidden, double *weight2, double *bias2, double *output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < OUTPUT_NODES)
    {
        double sum = bias2[idx];
        for (int j = 0; j < HIDDEN_NODES; j++)
        {
            sum += hidden[j] * weight2[j * OUTPUT_NODES + idx];
        }
        output[idx] = sigmoid_cuda(sum);
    }
}

// CUDA kernel for computing output layer deltas
__global__ void compute_output_deltas(double *output, double *target, double *delta_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < OUTPUT_NODES)
    {
        double a = output[idx];
        delta_out[idx] = (a - target[idx]) * a * (1.0 - a);
    }
}

// CUDA kernel for computing hidden layer deltas
__global__ void compute_hidden_deltas(double *delta_out, double *weight2, double *hidden, double *delta_hidden)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < HIDDEN_NODES)
    {
        double sum = 0.0;
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            sum += delta_out[j] * weight2[idx * OUTPUT_NODES + j];
        }
        delta_hidden[idx] = sum * (hidden[idx] > 0.0 ? 1.0 : 0.0);
    }
}

// CUDA kernel for updating weights from hidden to output
__global__ void update_weights_hidden_to_output(double *weight2, double *delta_out, double *hidden, double lr)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // hidden index
    int j = blockIdx.x * blockDim.x + threadIdx.x; // output index
    
    if (i < HIDDEN_NODES && j < OUTPUT_NODES)
    {
        weight2[i * OUTPUT_NODES + j] -= lr * delta_out[j] * hidden[i];
    }
}

// CUDA kernel for updating weights from input to hidden
__global__ void update_weights_input_to_hidden(double *weight1, double *delta_hidden, double *input, double lr)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // input index
    int j = blockIdx.x * blockDim.x + threadIdx.x; // hidden index
    
    if (i < INPUT_NODES && j < HIDDEN_NODES)
    {
        weight1[i * HIDDEN_NODES + j] -= lr * delta_hidden[j] * input[i];
    }
}

// CUDA kernel for updating output biases
__global__ void update_output_biases(double *bias2, double *delta_out, double lr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < OUTPUT_NODES)
    {
        bias2[idx] -= lr * delta_out[idx];
    }
}

// CUDA kernel for updating hidden biases
__global__ void update_hidden_biases(double *bias1, double *delta_hidden, double lr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < HIDDEN_NODES)
    {
        bias1[idx] -= lr * delta_hidden[idx];
    }
}

void allocate_gpu_memory()
{
    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&d_input, INPUT_NODES * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_target, OUTPUT_NODES * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_weight1, INPUT_NODES * HIDDEN_NODES * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_weight2, HIDDEN_NODES * OUTPUT_NODES * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_bias1, HIDDEN_NODES * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_bias2, OUTPUT_NODES * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_hidden, HIDDEN_NODES * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_output, OUTPUT_NODES * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_delta_out, OUTPUT_NODES * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_delta_hidden, HIDDEN_NODES * sizeof(double)));
}

void free_gpu_memory()
{
    cudaFree(d_input);
    cudaFree(d_target);
    cudaFree(d_weight1);
    cudaFree(d_weight2);
    cudaFree(d_bias1);
    cudaFree(d_bias2);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_delta_out);
    cudaFree(d_delta_hidden);
}

void train_cuda(double input[INPUT_NODES], double target[OUTPUT_NODES], int correct_label, int epoch)
{
    // Copy input and target to GPU
    CUDA_CHECK(cudaMemcpy(d_input, input, INPUT_NODES * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target, target, OUTPUT_NODES * sizeof(double), cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    dim3 hidden_blocks((HIDDEN_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 output_blocks((OUTPUT_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Forward pass
    forward_input_to_hidden<<<hidden_blocks, BLOCK_SIZE>>>(d_input, d_weight1, d_bias1, d_hidden);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    forward_hidden_to_output<<<output_blocks, BLOCK_SIZE>>>(d_hidden, d_weight2, d_bias2, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output back to host for accuracy calculation
    double host_output[OUTPUT_NODES];
    CUDA_CHECK(cudaMemcpy(host_output, d_output, OUTPUT_NODES * sizeof(double), cudaMemcpyDeviceToHost));
    
    int prediction = max_index(host_output, OUTPUT_NODES);
    if (prediction == correct_label)
        forward_prob_output++;

    // Backpropagation
    compute_output_deltas<<<output_blocks, BLOCK_SIZE>>>(d_output, d_target, d_delta_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    compute_hidden_deltas<<<hidden_blocks, BLOCK_SIZE>>>(d_delta_out, d_weight2, d_hidden, d_delta_hidden);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Calculate learning rate
    double lr = 0.1;
    if (epoch % 5 == 0 && epoch > 0)
    {
        lr *= 0.5;
    }

    // Update weights and biases
    dim3 weight2_block(16, 16);
    dim3 weight2_grid((OUTPUT_NODES + 15) / 16, (HIDDEN_NODES + 15) / 16);
    update_weights_hidden_to_output<<<weight2_grid, weight2_block>>>(d_weight2, d_delta_out, d_hidden, lr);
    CUDA_CHECK(cudaDeviceSynchronize());

    dim3 weight1_block(16, 16);
    dim3 weight1_grid((HIDDEN_NODES + 15) / 16, (INPUT_NODES + 15) / 16);
    update_weights_input_to_hidden<<<weight1_grid, weight1_block>>>(d_weight1, d_delta_hidden, d_input, lr);
    CUDA_CHECK(cudaDeviceSynchronize());

    update_output_biases<<<output_blocks, BLOCK_SIZE>>>(d_bias2, d_delta_out, lr);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    update_hidden_biases<<<hidden_blocks, BLOCK_SIZE>>>(d_bias1, d_delta_hidden, lr);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void test(double input[INPUT_NODES], double weight1[INPUT_NODES][HIDDEN_NODES], double weight2[HIDDEN_NODES][OUTPUT_NODES], double bias1[HIDDEN_NODES], double bias2[OUTPUT_NODES], int correct_label)
{
    double hidden[HIDDEN_NODES];
    double output_layer[OUTPUT_NODES];

    // Feedforward
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        double sum = bias1[i];
        for (int j = 0; j < INPUT_NODES; j++)
        {
            sum += input[j] * weight1[j][i];
        }
        hidden[i] = reLU(sum);
    }

    for (int i = 0; i < OUTPUT_NODES; i++)
    {
        double sum = bias2[i];
        for (int j = 0; j < HIDDEN_NODES; j++)
        {
            sum += hidden[j] * weight2[j][i];
        }
        output_layer[i] = sigmoid(sum);
    }

    int index = max_index(output_layer, OUTPUT_NODES);

    if (index == correct_label)
    {
        correct_predictions++;
    }
}

void save_weights_biases(char *file_name)
{
    FILE *file = fopen(file_name, "wb");
    if (file == NULL)
    {
        printf("Error opening file\n");
        exit(1);
    }
    fwrite(weight1, sizeof(double), HIDDEN_NODES * INPUT_NODES, file);
    fwrite(weight2, sizeof(double), HIDDEN_NODES * OUTPUT_NODES, file);
    fwrite(bias1, sizeof(double), HIDDEN_NODES, file);
    fwrite(bias2, sizeof(double), OUTPUT_NODES, file);
    fclose(file);
}

void load_weights_biases(char *file_name)
{
    FILE *file = fopen(file_name, "rb");
    if (file == NULL)
    {
        printf("Error opening file\n");
        exit(1);
    }
    fread(weight1, sizeof(double), HIDDEN_NODES * INPUT_NODES, file);
    fread(weight2, sizeof(double), HIDDEN_NODES * OUTPUT_NODES, file);
    fread(bias1, sizeof(double), HIDDEN_NODES, file);
    fread(bias2, sizeof(double), OUTPUT_NODES, file);
    fclose(file);
}

int main()
{
    clock_t start, end;
    float seconds;

    // Initialize weights and biases with small random values
    for (int i = 0; i < INPUT_NODES; i++)
    {
        for (int j = 0; j < HIDDEN_NODES; j++)
        {
            weight1[i][j] = (double)rand() / RAND_MAX * 0.1 - 0.05;
        }
    }
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        bias1[i] = (double)rand() / RAND_MAX * 0.1 - 0.05;
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            weight2[i][j] = (double)rand() / RAND_MAX * 0.1 - 0.05;
        }
    }
    for (int i = 0; i < OUTPUT_NODES; i++)
    {
        bias2[i] = (double)rand() / RAND_MAX * 0.1 - 0.05;
    }

    // Load MNIST dataset
    load_mnist();

    // Allocate GPU memory
    allocate_gpu_memory();

    // Copy initial weights and biases to GPU
    CUDA_CHECK(cudaMemcpy(d_weight1, weight1, INPUT_NODES * HIDDEN_NODES * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight2, weight2, HIDDEN_NODES * OUTPUT_NODES * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias1, bias1, HIDDEN_NODES * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias2, bias2, OUTPUT_NODES * sizeof(double), cudaMemcpyHostToDevice));

    // Train the network
    start = clock();

    for (int epoch = 0; epoch < NUMBER_OF_EPOCHS; epoch++)
    {
        forward_prob_output = 0;
        for (int i = 0; i < NUM_TRAINING_IMAGES; i++)
        {
            int correct_label = max_index(training_labels[i], OUTPUT_NODES);
            train_cuda(training_images[i], training_labels[i], correct_label, epoch);
        }
        printf("Epoch %d : Training Accuracy: %lf\n", epoch, (double)forward_prob_output / NUM_TRAINING_IMAGES);
    }

    end = clock();
    seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("Time to train: %f\n", seconds);

    // Copy weights and biases back from GPU
    CUDA_CHECK(cudaMemcpy(weight1, d_weight1, INPUT_NODES * HIDDEN_NODES * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(weight2, d_weight2, HIDDEN_NODES * OUTPUT_NODES * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bias1, d_bias1, HIDDEN_NODES * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bias2, d_bias2, OUTPUT_NODES * sizeof(double), cudaMemcpyDeviceToHost));

    save_weights_biases("model.bin");

    // Test the network
    start = clock();

    correct_predictions = 0;
    for (int i = 0; i < NUM_TEST_IMAGES; i++)
    {
        int correct_label = max_index(test_labels[i], OUTPUT_NODES);
        test(test_images[i], weight1, weight2, bias1, bias2, correct_label);
    }

    end = clock();
    seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("Time to test: %f\n", seconds);
    printf("Testing Accuracy: %f\n", (double)correct_predictions / NUM_TEST_IMAGES);

    // Free GPU memory
    free_gpu_memory();

    return 0;
}