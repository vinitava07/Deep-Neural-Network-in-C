/*
Paralelização GPU - CUDA
Alunos:
    Arthur C.
    Leonardo B.
    Vinícius T.
    Wanderson P.

Tempos de execução no Parcode (CUDA):
    A SER PREENCHIDO APÓS TESTES
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h> // Required for CUDA

// Definitions from main.c
#define INPUT_NODES 784
#define HIDDEN_NODES 256
#define OUTPUT_NODES 10
#define NUM_TRAINING_IMAGES 8000
#define NUM_TEST_IMAGES 1000
#define NUMBER_OF_EPOCHS 10

#define THREADS_PER_BLOCK 256 // Example, tune as needed

// Host arrays
double h_training_images[NUM_TRAINING_IMAGES][INPUT_NODES];
double h_training_labels[NUM_TRAINING_IMAGES][OUTPUT_NODES];
double h_test_images[NUM_TEST_IMAGES][INPUT_NODES];
double h_test_labels[NUM_TEST_IMAGES][OUTPUT_NODES];

double h_weight1[INPUT_NODES][HIDDEN_NODES];
double h_weight2[HIDDEN_NODES][OUTPUT_NODES];
double h_bias1[HIDDEN_NODES];
double h_bias2[OUTPUT_NODES];

// Device pointers
double *d_training_images, *d_training_labels;
double *d_test_images, *d_test_labels;
double *d_weight1, *d_weight2, *d_bias1, *d_bias2;
double *d_grad_weight1, *d_grad_weight2, *d_grad_bias1, *d_grad_bias2; // For gradient accumulation
int *d_epoch_correct_preds; // For atomic accumulation of correct predictions

// CUDA error checking macro
#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Function to load MNIST data (same as main_omp_gpu.c, populates h_ arrays)
void load_mnist_host() { /* ... same as load_mnist in main_omp_gpu.c ... */
    // Ensure it populates h_training_images, h_training_labels etc.
    // For brevity, implementation omitted here but it's the same as above.
    // Replace 'training_images' with 'h_training_images', etc.
    printf("MNIST dataset loaded to host memory.\n");
}


__device__ double dev_sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
__device__ double dev_reLU(double x) { return x > 0.0 ? x : 0.0; }

// Kernel to process one image (feedforward, backprop, accumulate gradients)
__global__ void train_image_kernel(
    const double* current_image, const double* current_label,
    const double* W1, const double* B1, const double* W2, const double* B2,
    double* grad_W1, double* grad_B1, double* grad_W2, double* grad_B2,
    int* d_correct_preds_accumulator)
{
    // This kernel would ideally be launched with one thread per image, or one block per image.
    // For simplicity, let's assume this kernel is complex and a block of threads works on one image.
    // Or, this kernel processes one element of an output, needing careful indexing.

    // To align with the "process all images in parallel" concept, let's make this kernel process
    // one image, identified by blockIdx.x.
    int img_idx = blockIdx.x;
    if (img_idx >= NUM_TRAINING_IMAGES) return;

    // Thread ID within the block. We'll use threads for parallelizing parts of one image's computation.
    int tid = threadIdx.x;

    // Temporary arrays in shared memory or local memory for one image's processing
    // For simplicity of this conceptual code, using local memory (registers/stack if small enough)
    // A more optimized version would use shared memory for intermediate results like hidden_layer.
    extern __shared__ double s_data[]; // Dynamic shared memory
    double* s_hidden_layer = s_data; // size: HIDDEN_NODES
    double* s_output_layer = (double*)&s_hidden_layer[HIDDEN_NODES]; // size: OUTPUT_NODES
    double* s_delta_out    = (double*)&s_output_layer[OUTPUT_NODES]; // size: OUTPUT_NODES
    double* s_delta_hidden = (double*)&s_delta_out[OUTPUT_NODES];    // size: HIDDEN_NODES
                                                                   // Total shared: (2*H + 2*O)*sizeof(double)


    // --- 1. Feedforward ---
    // Input to Hidden (Parallelize across threads in the block for HIDDEN_NODES)
    for (int i = tid; i < HIDDEN_NODES; i += blockDim.x) {
        double sum = B1[i];
        for (int j = 0; j < INPUT_NODES; j++) {
            sum += current_image[img_idx * INPUT_NODES + j] * W1[j * HIDDEN_NODES + i]; // Adjust W1 indexing if row/col major differs
        }
        s_hidden_layer[i] = dev_reLU(sum);
    }
    __syncthreads();

    // Hidden to Output (Parallelize across threads for OUTPUT_NODES)
    for (int i = tid; i < OUTPUT_NODES; i += blockDim.x) {
        double sum = B2[i];
        for (int j = 0; j < HIDDEN_NODES; j++) {
            sum += s_hidden_layer[j] * W2[j * OUTPUT_NODES + i]; // Adjust W2 indexing
        }
        s_output_layer[i] = dev_sigmoid(sum);
    }
    __syncthreads();

    // --- Track Accuracy (Done by one thread, e.g., thread 0) ---
    if (tid == 0) {
        int predicted_label = 0;
        double max_val = s_output_layer[0];
        for(int k=1; k<OUTPUT_NODES; ++k) {
            if(s_output_layer[k] > max_val) {
                max_val = s_output_layer[k];
                predicted_label = k;
            }
        }
        int actual_label = 0;
        // Assuming current_label is one-hot encoded for the img_idx
        double max_target_val = current_label[img_idx * OUTPUT_NODES + 0];
        for(int k=1; k<OUTPUT_NODES; ++k) {
            if(current_label[img_idx * OUTPUT_NODES + k] > max_target_val) {
                 max_target_val = current_label[img_idx * OUTPUT_NODES + k];
                 actual_label = k;
            }
        }
        if (predicted_label == actual_label) {
            atomicAdd(d_correct_preds_accumulator, 1);
        }
    }
    // No __syncthreads() needed here if only tid 0 writes to global atomic.

    // --- 2. Backpropagation: Calculate Deltas ---
    // Output layer deltas (Parallelize for OUTPUT_NODES)
    for (int i = tid; i < OUTPUT_NODES; i += blockDim.x) {
        double pred = s_output_layer[i];
        s_delta_out[i] = (pred - current_label[img_idx * OUTPUT_NODES + i]) * pred * (1.0 - pred);
    }
    __syncthreads();

    // Hidden layer deltas (Parallelize for HIDDEN_NODES)
    for (int i = tid; i < HIDDEN_NODES; i += blockDim.x) {
        double sum_weighted_delta = 0.0;
        for (int j = 0; j < OUTPUT_NODES; j++) {
            sum_weighted_delta += s_delta_out[j] * W2[i * OUTPUT_NODES + j]; // W2(row i, col j)
        }
        s_delta_hidden[i] = sum_weighted_delta * (s_hidden_layer[i] > 0.0 ? 1.0 : 0.0);
    }
    __syncthreads();

    // --- 3. Accumulate Gradients (Atomic adds to global gradient arrays) ---
    // grad_W2 and grad_B2
    for (int i = tid; i < HIDDEN_NODES; i += blockDim.x) { // Iterate over hidden nodes
        for (int j = 0; j < OUTPUT_NODES; j++) { // Iterate over output nodes
            atomicAdd(&grad_W2[i * OUTPUT_NODES + j], s_delta_out[j] * s_hidden_layer[i]);
        }
    }
     for (int j = tid; j < OUTPUT_NODES; j+=blockDim.x) { // Iterate over output nodes
        atomicAdd(&grad_B2[j], s_delta_out[j]);
    }
    // No syncthreads needed before next independent loop if atomics are used on global memory.

    // grad_W1 and grad_B1
    for (int i = tid; i < INPUT_NODES; i += blockDim.x) { // Iterate over input nodes
        for (int j = 0; j < HIDDEN_NODES; j++) { // Iterate over hidden nodes
            atomicAdd(&grad_W1[i * HIDDEN_NODES + j], s_delta_hidden[j] * current_image[img_idx * INPUT_NODES + i]);
        }
    }
    for (int j = tid; j < HIDDEN_NODES; j+=blockDim.x) { // Iterate over hidden nodes
        atomicAdd(&grad_B1[j], s_delta_hidden[j]);
    }
}

// Kernel to update weights using accumulated gradients
__global__ void update_weights_kernel(
    double* W1, double* B1, double* W2, double* B2,
    const double* grad_W1, const double* grad_B1, const double* grad_W2, const double* grad_B2,
    double learning_rate, int num_samples_in_batch) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Update W1
    if (idx < INPUT_NODES * HIDDEN_NODES) {
        W1[idx] -= learning_rate * grad_W1[idx] / num_samples_in_batch;
    }
    // Update B1
    if (idx < HIDDEN_NODES) { // Reuse idx, but needs separate launch or careful indexing
         // This part of kernel needs to be launched with fewer threads or guarded
    } // Simplified: Assume separate kernels or careful indexing for different sized arrays.

    // For simplicity, assume one kernel launch per weight/bias matrix/vector update
    // Example for W1:
    // int row = blockIdx.y * blockDim.y + threadIdx.y; // if 2D grid
    // int col = blockIdx.x * blockDim.x + threadIdx.x;
    // if (row < INPUT_NODES && col < HIDDEN_NODES) {
    //    int flat_idx = row * HIDDEN_NODES + col;
    //    W1[flat_idx] -= learning_rate * grad_W1[flat_idx] / num_samples_in_batch;
    // }

    // Simplified update for W1 (1D grid for W1 elements)
    if (idx < INPUT_NODES * HIDDEN_NODES) W1[idx] -= learning_rate * (grad_W1[idx] / num_samples_in_batch);
    // Separate similar kernels or logic for B1, W2, B2
}
__global__ void update_bias_kernel(double* B, const double* grad_B, double learning_rate, int num_samples, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        B[idx] -= learning_rate * (grad_B[idx] / num_samples);
    }
}


// Kernel to zero out gradient arrays
__global__ void zero_gradients_kernel(double* grad_array, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        grad_array[idx] = 0.0;
    }
}

// Test kernel (simpler, just feedforward)
__global__ void test_image_kernel(
    const double* d_test_imgs, const double* d_test_lbls,
    const double* W1, const double* B1, const double* W2, const double* B2,
    int* d_correct_preds_accumulator, int num_test_images)
{
    int img_idx = blockIdx.x; // One block per image for testing
    if (img_idx >= num_test_images) return;

    int tid = threadIdx.x;
    extern __shared__ double s_test_data[];
    double* s_hidden_layer = s_test_data; // size: HIDDEN_NODES
    double* s_output_layer = (double*)&s_hidden_layer[HIDDEN_NODES]; // size: OUTPUT_NODES
                                                                  // Shared: (H+O)*sizeof(double)
    // Feedforward (similar to training kernel but using d_test_imgs)
    // Input to Hidden
    for (int i = tid; i < HIDDEN_NODES; i += blockDim.x) {
        double sum = B1[i];
        for (int j = 0; j < INPUT_NODES; j++) {
            sum += d_test_imgs[img_idx * INPUT_NODES + j] * W1[j * HIDDEN_NODES + i];
        }
        s_hidden_layer[i] = dev_reLU(sum);
    }
    __syncthreads();

    // Hidden to Output
    for (int i = tid; i < OUTPUT_NODES; i += blockDim.x) {
        double sum = B2[i];
        for (int j = 0; j < HIDDEN_NODES; j++) {
            sum += s_hidden_layer[j] * W2[j * OUTPUT_NODES + i];
        }
        s_output_layer[i] = dev_sigmoid(sum);
    }
    __syncthreads();

    if (tid == 0) {
        int predicted_label = 0;
        double max_val = s_output_layer[0];
        for(int k=1; k<OUTPUT_NODES; ++k) if(s_output_layer[k] > max_val) { max_val = s_output_layer[k]; predicted_label = k; }
        
        int actual_label = 0;
        double max_target_val = d_test_lbls[img_idx * OUTPUT_NODES + 0];
        for(int k=1; k<OUTPUT_NODES; ++k) if(d_test_lbls[img_idx * OUTPUT_NODES + k] > max_target_val) {max_target_val = d_test_lbls[img_idx * OUTPUT_NODES + k]; actual_label = k; }

        if (predicted_label == actual_label) {
            atomicAdd(d_correct_preds_accumulator, 1);
        }
    }
}


int main() {
    load_mnist_host(); // Load data into h_ arrays

    // Initialize weights on host
    srand(42);
    for (int i=0; i<INPUT_NODES; ++i) for (int j=0; j<HIDDEN_NODES; ++j) h_weight1[i][j] = (double)rand()/RAND_MAX*0.1-0.05;
    for (int i=0; i<HIDDEN_NODES; ++i) h_bias1[i] = (double)rand()/RAND_MAX*0.1-0.05;
    for (int i=0; i<HIDDEN_NODES; ++i) for (int j=0; j<OUTPUT_NODES; ++j) h_weight2[i][j] = (double)rand()/RAND_MAX*0.1-0.05;
    for (int i=0; i<OUTPUT_NODES; ++i) h_bias2[i] = (double)rand()/RAND_MAX*0.1-0.05;

    // Allocate memory on GPU
    cudaCheck(cudaMalloc(&d_training_images, NUM_TRAINING_IMAGES * INPUT_NODES * sizeof(double)));
    cudaCheck(cudaMalloc(&d_training_labels, NUM_TRAINING_IMAGES * OUTPUT_NODES * sizeof(double)));
    cudaCheck(cudaMalloc(&d_weight1, INPUT_NODES * HIDDEN_NODES * sizeof(double)));
    cudaCheck(cudaMalloc(&d_bias1, HIDDEN_NODES * sizeof(double)));
    cudaCheck(cudaMalloc(&d_weight2, HIDDEN_NODES * OUTPUT_NODES * sizeof(double)));
    cudaCheck(cudaMalloc(&d_bias2, OUTPUT_NODES * sizeof(double)));
    
    cudaCheck(cudaMalloc(&d_grad_weight1, INPUT_NODES * HIDDEN_NODES * sizeof(double)));
    cudaCheck(cudaMalloc(&d_grad_bias1, HIDDEN_NODES * sizeof(double)));
    cudaCheck(cudaMalloc(&d_grad_weight2, HIDDEN_NODES * OUTPUT_NODES * sizeof(double)));
    cudaCheck(cudaMalloc(&d_grad_bias2, OUTPUT_NODES * sizeof(double)));
    cudaCheck(cudaMalloc(&d_epoch_correct_preds, sizeof(int)));

    // Copy initial data from host to device
    cudaCheck(cudaMemcpy(d_training_images, h_training_images, NUM_TRAINING_IMAGES * INPUT_NODES * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_training_labels, h_training_labels, NUM_TRAINING_IMAGES * OUTPUT_NODES * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight1, h_weight1, INPUT_NODES * HIDDEN_NODES * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_bias1, h_bias1, HIDDEN_NODES * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight2, h_weight2, HIDDEN_NODES * OUTPUT_NODES * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_bias2, h_bias2, OUTPUT_NODES * sizeof(double), cudaMemcpyHostToDevice));

    printf("Starting training with CUDA...\n");
    double lr = 0.1;
    clock_t start_train_total = clock();

    // Calculate shared memory size for training kernel
    size_t train_shared_mem_size = (2 * HIDDEN_NODES + 2 * OUTPUT_NODES) * sizeof(double);


    for (int epoch = 0; epoch < NUMBER_OF_EPOCHS; epoch++) {
        clock_t start_epoch = clock();
        int h_epoch_correct_predictions = 0;
        cudaCheck(cudaMemset(d_epoch_correct_preds, 0, sizeof(int))); // Reset counter on GPU

        // Zero out gradient accumulators on GPU
        zero_gradients_kernel<<<(INPUT_NODES * HIDDEN_NODES + THREADS_PER_BLOCK -1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_grad_weight1, INPUT_NODES * HIDDEN_NODES);
        zero_gradients_kernel<<<(HIDDEN_NODES + THREADS_PER_BLOCK -1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_grad_bias1, HIDDEN_NODES);
        zero_gradients_kernel<<<(HIDDEN_NODES * OUTPUT_NODES + THREADS_PER_BLOCK -1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_grad_weight2, HIDDEN_NODES * OUTPUT_NODES);
        zero_gradients_kernel<<<(OUTPUT_NODES + THREADS_PER_BLOCK -1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_grad_bias2, OUTPUT_NODES);
        cudaCheck(cudaGetLastError());
        
        // Launch kernel for all training images
        // One CUDA block per image. Threads within block parallelize image computation.
        train_image_kernel<<<NUM_TRAINING_IMAGES, THREADS_PER_BLOCK, train_shared_mem_size>>>(
            d_training_images, d_training_labels,
            d_weight1, d_bias1, d_weight2, d_bias2,
            d_grad_weight1, d_grad_bias1, d_grad_weight2, d_grad_bias2,
            d_epoch_correct_preds
        );
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize()); // Wait for all images in epoch to be processed

        // Update master weights using accumulated gradients
        // Example for W1, similar launches for B1, W2, B2
        update_weights_kernel<<<(INPUT_NODES * HIDDEN_NODES + THREADS_PER_BLOCK -1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            d_weight1, nullptr, nullptr, nullptr, d_grad_weight1, nullptr, nullptr, nullptr, lr, NUM_TRAINING_IMAGES); // Simplified call
        update_bias_kernel<<<(HIDDEN_NODES + THREADS_PER_BLOCK -1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_bias1, d_grad_bias1, lr, NUM_TRAINING_IMAGES, HIDDEN_NODES);
        update_weights_kernel<<<(HIDDEN_NODES * OUTPUT_NODES + THREADS_PER_BLOCK -1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            d_weight2, nullptr, nullptr, nullptr, d_grad_weight2, nullptr, nullptr, nullptr, lr, NUM_TRAINING_IMAGES); // Simplified call
        update_bias_kernel<<<(OUTPUT_NODES + THREADS_PER_BLOCK -1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_bias2, d_grad_bias2, lr, NUM_TRAINING_IMAGES, OUTPUT_NODES);

        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        cudaCheck(cudaMemcpy(&h_epoch_correct_predictions, d_epoch_correct_preds, sizeof(int), cudaMemcpyDeviceToHost));
        
        clock_t end_epoch = clock();
        double epoch_time = (double)(end_epoch - start_epoch) / CLOCKS_PER_SEC;
        printf("Epoch %d: Training Accuracy: %lf, Time: %.2fs\n",
               epoch, (double)h_epoch_correct_predictions / NUM_TRAINING_IMAGES, epoch_time);

        if (epoch > 0 && epoch % 5 == 0) {
            lr *= 0.5; printf("Adjusted learning rate to %lf\n", lr);
        }
    }
    clock_t end_train_total = clock();
    double total_training_time = (double)(end_train_total - start_train_total) / CLOCKS_PER_SEC;
    printf("Total Training Time: %.2fs\n", total_training_time);

    // --- Testing Phase ---
    printf("\nStarting testing with CUDA...\n");
    clock_t start_test = clock();
    int h_test_correct_predictions = 0;
    
    cudaCheck(cudaMalloc(&d_test_images, NUM_TEST_IMAGES * INPUT_NODES * sizeof(double)));
    cudaCheck(cudaMalloc(&d_test_labels, NUM_TEST_IMAGES * OUTPUT_NODES * sizeof(double)));
    cudaCheck(cudaMemcpy(d_test_images, h_test_images, NUM_TEST_IMAGES * INPUT_NODES * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_test_labels, h_test_labels, NUM_TEST_IMAGES * OUTPUT_NODES * sizeof(double), cudaMemcpyHostToDevice));
    
    cudaCheck(cudaMemset(d_epoch_correct_preds, 0, sizeof(int))); // Reuse for test correct preds

    size_t test_shared_mem_size = (HIDDEN_NODES + OUTPUT_NODES) * sizeof(double);
    test_image_kernel<<<NUM_TEST_IMAGES, THREADS_PER_BLOCK, test_shared_mem_size>>>(
        d_test_images, d_test_labels,
        d_weight1, d_bias1, d_weight2, d_bias2, // Use trained weights
        d_epoch_correct_preds, NUM_TEST_IMAGES
    );
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(&h_test_correct_predictions, d_epoch_correct_preds, sizeof(int), cudaMemcpyDeviceToHost));
    
    clock_t end_test = clock();
    double total_testing_time = (double)(end_test - start_test) / CLOCKS_PER_SEC;
    printf("Total Testing Time: %.2fs\n", total_testing_time);
    printf("Testing Accuracy: %f\n", (double)h_test_correct_predictions / NUM_TEST_IMAGES);

    // Free GPU memory
    cudaFree(d_training_images); cudaFree(d_training_labels);
    cudaFree(d_test_images); cudaFree(d_test_labels);
    cudaFree(d_weight1); cudaFree(d_bias1); cudaFree(d_weight2); cudaFree(d_bias2);
    cudaFree(d_grad_weight1); cudaFree(d_grad_bias1); cudaFree(d_grad_weight2); cudaFree(d_grad_bias2);
    cudaFree(d_epoch_correct_preds);

    return 0;
}