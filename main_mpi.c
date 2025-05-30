#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

#define INPUT_NODES 784  // 28*28 pixels
#define HIDDEN_NODES 256 // Number of hidden nodes
#define OUTPUT_NODES 10  // 10 digits (0-9)

#define NUM_TRAINING_IMAGES 8000
#define NUM_TEST_IMAGES 1000

#define NUMBER_OF_EPOCHS 10

double training_images[NUM_TRAINING_IMAGES][INPUT_NODES];
double training_labels[NUM_TRAINING_IMAGES][OUTPUT_NODES];
double test_images[NUM_TEST_IMAGES][INPUT_NODES];
double test_labels[NUM_TEST_IMAGES][OUTPUT_NODES];

// Initialize weights and biases
double weight1[INPUT_NODES][HIDDEN_NODES];
double weight2[HIDDEN_NODES][OUTPUT_NODES];
double bias1[HIDDEN_NODES];
double bias2[OUTPUT_NODES];

int correct_predictions;
int forward_prob_output;

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

void train(
    double input[INPUT_NODES],
    double target[OUTPUT_NODES],
    double weight1[INPUT_NODES][HIDDEN_NODES],
    double weight2[HIDDEN_NODES][OUTPUT_NODES],
    double bias1[HIDDEN_NODES],
    double bias2[OUTPUT_NODES],
    int correct_label,
    int epoch,
    int num_threads)
{
    double hidden[HIDDEN_NODES];
    double output_layer[OUTPUT_NODES];
    
    // Set number of threads for OpenMP
    omp_set_num_threads(num_threads);

    // 1) Feedforward
    #pragma omp parallel
    {
        // First layer: input -> hidden
        #pragma omp for
        for (int i = 0; i < HIDDEN_NODES; i++)
        {
            double sum = bias1[i];
            for (int j = 0; j < INPUT_NODES; j++)
                sum += input[j] * weight1[j][i];
            hidden[i] = reLU(sum);
        }

        // Second layer: hidden -> output
        #pragma omp for
        for (int i = 0; i < OUTPUT_NODES; i++)
        {
            double sum = bias2[i];
            for (int j = 0; j < HIDDEN_NODES; j++)
                sum += hidden[j] * weight2[j][i];
            output_layer[i] = sigmoid(sum);
        }
    }

    // (Optional) track accuracy - not parallelized as it's a scalar operation
    int prediction = max_index(output_layer, OUTPUT_NODES);
    if (prediction == correct_label)
        forward_prob_output++;

    // 2) Backpropagation deltas
    double delta_out[OUTPUT_NODES];
    double delta_hidden[HIDDEN_NODES];
    
    #pragma omp parallel
    {
        // Calculate output deltas
        #pragma omp for
        for (int i = 0; i < OUTPUT_NODES; i++)
        {
            // dL/dz = (y_pred - y_true) * σ'(z)
            double a = output_layer[i];
            delta_out[i] = (a - target[i]) * a * (1.0 - a);
        }

        // Calculate hidden layer deltas
        #pragma omp for
        for (int i = 0; i < HIDDEN_NODES; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < OUTPUT_NODES; j++)
                sum += delta_out[j] * weight2[i][j];
            // σ'(z_hidden)
            delta_hidden[i] = sum * (hidden[i] > 0.0 ? 1.0 : 0.0);
        }
    }

    // 3) Update weights & biases
    double lr = 0.1;

    if (epoch % 5 == 0 && epoch > 0)
    {
        lr *= 0.5;
    }

    #pragma omp parallel
    {
        // (a) Hidden → Output weights & biases
        #pragma omp for collapse(2)
        for (int i = 0; i < HIDDEN_NODES; i++)
        {
            for (int j = 0; j < OUTPUT_NODES; j++)
            {
                weight2[i][j] -= lr * delta_out[j] * hidden[i];
            }
        }

        #pragma omp for
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            bias2[j] -= lr * delta_out[j];
        }

        // (b) Input → Hidden weights & biases
        #pragma omp for collapse(2)
        for (int i = 0; i < INPUT_NODES; i++)
        {
            for (int j = 0; j < HIDDEN_NODES; j++)
            {
                weight1[i][j] -= lr * delta_hidden[j] * input[i];
            }
        }

        #pragma omp for
        for (int j = 0; j < HIDDEN_NODES; j++)
        {
            bias1[j] -= lr * delta_hidden[j];
        }
    }
}

void test(double input[INPUT_NODES], double weight1[INPUT_NODES][HIDDEN_NODES], double weight2[HIDDEN_NODES][OUTPUT_NODES], double bias1[HIDDEN_NODES], double bias2[OUTPUT_NODES], int correct_label)
{
    double hidden[HIDDEN_NODES];
    double output_layer[OUTPUT_NODES];

    // Feedforward
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        double sum = bias1[i]; // Add bias first
        for (int j = 0; j < INPUT_NODES; j++)
        {
            sum += input[j] * weight1[j][i];
        }
        hidden[i] = reLU(sum);
    }

    for (int i = 0; i < OUTPUT_NODES; i++)
    {
        double sum = bias2[i]; // Add bias first
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

int main(int argc, char **argv)
{
    int rank, size;
    double start_time, end_time;
    int threads_per_process = 1;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Configure threads based on total process count
    if (size == 1) {
        threads_per_process = 4;  // 1 process with 4 threads
    } else if (size == 2) {
        threads_per_process = 2;  // 2 processes with 2 threads each
    } else {
        threads_per_process = 1;  // 4 processes with no threads (pure MPI)
    }
    
    // Set number of OpenMP threads
    omp_set_num_threads(threads_per_process);
    
    if (rank == 0) {
        printf("Running with %d MPI processes, %d OpenMP threads per process\n", 
               size, threads_per_process);
    }
    
    // Initialize weights and biases with small random values (same on all processes)
    //srand(42);  // Use same seed on all processes for identical initialization
    
    for (int i = 0; i < INPUT_NODES; i++) {
        for (int j = 0; j < HIDDEN_NODES; j++) {
            weight1[i][j] = (double)rand() / RAND_MAX * 0.1 - 0.05;
        }
    }
    
    for (int i = 0; i < HIDDEN_NODES; i++) {
        bias1[i] = (double)rand() / RAND_MAX * 0.1 - 0.05;
        for (int j = 0; j < OUTPUT_NODES; j++) {
            weight2[i][j] = (double)rand() / RAND_MAX * 0.1 - 0.05;
        }
    }
    
    for (int i = 0; i < OUTPUT_NODES; i++) {
        bias2[i] = (double)rand() / RAND_MAX * 0.1 - 0.05;
    }
    
    // Load MNIST dataset (all processes need the data)
    load_mnist();
    
    // Calculate how many training images each process will handle
    int images_per_process = NUM_TRAINING_IMAGES / size;
    int start_image = rank * images_per_process;
    int end_image = (rank == size - 1) ? NUM_TRAINING_IMAGES : start_image + images_per_process;
    
    // Synchronize before starting timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Start timing
    start_time = MPI_Wtime();
    
    // Training loop
    for (int epoch = 0; epoch < NUMBER_OF_EPOCHS; epoch++) {
        // Reset counters
        forward_prob_output = 0;
        
        // Each process handles its portion of training data
        for (int i = start_image; i < end_image; i++) {
            int correct_label = max_index(training_labels[i], OUTPUT_NODES);
            train(training_images[i], training_labels[i], weight1, weight2, 
                  bias1, bias2, correct_label, epoch, threads_per_process);
        }
        
        // Synchronize the weights and biases across all processes
        MPI_Allreduce(MPI_IN_PLACE, weight1, INPUT_NODES * HIDDEN_NODES, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, weight2, HIDDEN_NODES * OUTPUT_NODES, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, bias1, HIDDEN_NODES, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, bias2, OUTPUT_NODES, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        // Average the weights and biases
        for (int i = 0; i < INPUT_NODES; i++) {
            for (int j = 0; j < HIDDEN_NODES; j++) {
                weight1[i][j] /= size;
            }
        }
        
        for (int i = 0; i < HIDDEN_NODES; i++) {
            bias1[i] /= size;
            for (int j = 0; j < OUTPUT_NODES; j++) {
                weight2[i][j] /= size;
            }
        }
        
        for (int i = 0; i < OUTPUT_NODES; i++) {
            bias2[i] /= size;
        }
        
        // Gather accuracy statistics
        int global_forward_prob = 0;
        MPI_Reduce(&forward_prob_output, &global_forward_prob, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            printf("Epoch %d : Training Accuracy: %lf\n", epoch, 
                   (double)global_forward_prob / NUM_TRAINING_IMAGES);
            printf("Example weight: %lf\n", weight1[0][0]);
        }
    }
    
    // End training timing
    end_time = MPI_Wtime();
    double training_time = end_time - start_time;
    
    if (rank == 0) {
        printf("Time to train: %f seconds\n", training_time);
        save_weights_biases("model_mpi.bin");
    }
    
    // Testing phase
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    // Calculate how many test images each process will handle
    int test_images_per_process = NUM_TEST_IMAGES / size;
    int start_test = rank * test_images_per_process;
    int end_test = (rank == size - 1) ? NUM_TEST_IMAGES : start_test + test_images_per_process;
    
    correct_predictions = 0;
    for (int i = start_test; i < end_test; i++) {
        int correct_label = max_index(test_labels[i], OUTPUT_NODES);
        test(test_images[i], weight1, weight2, bias1, bias2, correct_label);
    }
    
    // Gather test results
    int global_correct = 0;
    MPI_Reduce(&correct_predictions, &global_correct, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    end_time = MPI_Wtime();
    double testing_time = end_time - start_time;
    
    if (rank == 0) {
        printf("Time to test: %f seconds\n", testing_time);
        printf("Testing Accuracy: %f\n", (double)global_correct / NUM_TEST_IMAGES);
    }
    
    MPI_Finalize();
    return 0;
}