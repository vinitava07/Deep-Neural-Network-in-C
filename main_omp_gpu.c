#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
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

void train_parallel(int epoch)
{
    double lr = 0.1;
    if (epoch % 5 == 0 && epoch > 0)
    {
        lr *= 0.5;
    }

    forward_prob_output = 0;

    // Parallel training with GPU offloading
    #pragma omp target teams distribute parallel for \
            map(to: training_images[0:NUM_TRAINING_IMAGES][0:INPUT_NODES], \
                    training_labels[0:NUM_TRAINING_IMAGES][0:OUTPUT_NODES]) \
            map(tofrom: weight1[0:INPUT_NODES][0:HIDDEN_NODES], \
                        weight2[0:HIDDEN_NODES][0:OUTPUT_NODES], \
                        bias1[0:HIDDEN_NODES], bias2[0:OUTPUT_NODES]) \
            reduction(+:forward_prob_output) schedule(static)
    for (int img_idx = 0; img_idx < NUM_TRAINING_IMAGES; img_idx++)
    {
        double hidden[HIDDEN_NODES];
        double output_layer[OUTPUT_NODES];
        
        // Find correct label
        int correct_label = 0;
        for (int k = 0; k < OUTPUT_NODES; k++)
        {
            if (training_labels[img_idx][k] == 1.0)
            {
                correct_label = k;
                break;
            }
        }

        // Forward pass - Hidden layer
        for (int i = 0; i < HIDDEN_NODES; i++)
        {
            double sum = bias1[i];
            for (int j = 0; j < INPUT_NODES; j++)
            {
                sum += training_images[img_idx][j] * weight1[j][i];
            }
            hidden[i] = (sum > 0.0) ? sum : 0.0; // ReLU
        }

        // Forward pass - Output layer
        for (int i = 0; i < OUTPUT_NODES; i++)
        {
            double sum = bias2[i];
            for (int j = 0; j < HIDDEN_NODES; j++)
            {
                sum += hidden[j] * weight2[j][i];
            }
            output_layer[i] = 1.0 / (1.0 + exp(-sum)); // Sigmoid
        }

        // Check prediction for accuracy
        int prediction = 0;
        double max_output = output_layer[0];
        for (int j = 1; j < OUTPUT_NODES; j++)
        {
            if (output_layer[j] > max_output)
            {
                max_output = output_layer[j];
                prediction = j;
            }
        }
        
        if (prediction == correct_label)
        {
            forward_prob_output++;
        }

        // Backward pass - Output layer deltas
        double delta_out[OUTPUT_NODES];
        for (int i = 0; i < OUTPUT_NODES; i++)
        {
            double a = output_layer[i];
            delta_out[i] = (a - training_labels[img_idx][i]) * a * (1.0 - a);
        }

        // Backward pass - Hidden layer deltas
        double delta_hidden[HIDDEN_NODES];
        for (int i = 0; i < HIDDEN_NODES; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < OUTPUT_NODES; j++)
            {
                sum += delta_out[j] * weight2[i][j];
            }
            delta_hidden[i] = sum * ((hidden[i] > 0.0) ? 1.0 : 0.0); // ReLU derivative
        }

        // Update weights and biases (using critical section to avoid race conditions)
        #pragma omp critical
        {
            // Update hidden to output weights
            for (int i = 0; i < HIDDEN_NODES; i++)
            {
                for (int j = 0; j < OUTPUT_NODES; j++)
                {
                    weight2[i][j] -= lr * delta_out[j] * hidden[i];
                }
            }

            // Update output biases
            for (int j = 0; j < OUTPUT_NODES; j++)
            {
                bias2[j] -= lr * delta_out[j];
            }

            // Update input to hidden weights
            for (int i = 0; i < INPUT_NODES; i++)
            {
                for (int j = 0; j < HIDDEN_NODES; j++)
                {
                    weight1[i][j] -= lr * delta_hidden[j] * training_images[img_idx][i];
                }
            }

            // Update hidden biases
            for (int j = 0; j < HIDDEN_NODES; j++)
            {
                bias1[j] -= lr * delta_hidden[j];
            }
        }
    }
}

void test_parallel()
{
    correct_predictions = 0;
    
    // Test in parallel on GPU
    #pragma omp target teams distribute parallel for \
            map(to: test_images[0:NUM_TEST_IMAGES][0:INPUT_NODES], \
                    test_labels[0:NUM_TEST_IMAGES][0:OUTPUT_NODES], \
                    weight1[0:INPUT_NODES][0:HIDDEN_NODES], \
                    weight2[0:HIDDEN_NODES][0:OUTPUT_NODES], \
                    bias1[0:HIDDEN_NODES], bias2[0:OUTPUT_NODES]) \
            reduction(+:correct_predictions) schedule(static)
    for (int i = 0; i < NUM_TEST_IMAGES; i++)
    {
        double hidden[HIDDEN_NODES];
        double output_layer[OUTPUT_NODES];
        
        // Forward pass - Hidden layer
        for (int j = 0; j < HIDDEN_NODES; j++)
        {
            double sum = bias1[j];
            for (int k = 0; k < INPUT_NODES; k++)
            {
                sum += test_images[i][k] * weight1[k][j];
            }
            hidden[j] = (sum > 0.0) ? sum : 0.0; // ReLU
        }
        
        // Forward pass - Output layer
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            double sum = bias2[j];
            for (int k = 0; k < HIDDEN_NODES; k++)
            {
                sum += hidden[k] * weight2[k][j];
            }
            output_layer[j] = 1.0 / (1.0 + exp(-sum)); // Sigmoid
        }
        
        // Find prediction and correct label
        int prediction = 0;
        int correct_label = 0;
        double max_output = output_layer[0];
        
        for (int j = 1; j < OUTPUT_NODES; j++)
        {
            if (output_layer[j] > max_output)
            {
                max_output = output_layer[j];
                prediction = j;
            }
        }
        
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            if (test_labels[i][j] == 1.0)
            {
                correct_label = j;
                break;
            }
        }
        
        if (prediction == correct_label)
        {
            correct_predictions++;
        }
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
    
    // Check OpenMP GPU support
    printf("OpenMP version: %d\n", _OPENMP);
    printf("Number of devices: %d\n", omp_get_num_devices());
    printf("Default device: %d\n", omp_get_default_device());
    
    if (omp_get_num_devices() == 0)
    {
        printf("Warning: No GPU devices found. Running on CPU only.\n");
    }
    
    // Set number of threads for CPU parallel regions
    omp_set_num_threads(omp_get_max_threads());
    printf("Number of CPU threads: %d\n", omp_get_max_threads());
    
    // Initialize weights and biases with small random values
    srand(time(NULL));
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < INPUT_NODES; i++)
    {
        for (int j = 0; j < HIDDEN_NODES; j++)
        {
            weight1[i][j] = (double)rand() / RAND_MAX * 0.1 - 0.05;
        }
    }
    
    #pragma omp parallel for
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        bias1[i] = (double)rand() / RAND_MAX * 0.1 - 0.05;
        
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            weight2[i][j] = (double)rand() / RAND_MAX * 0.1 - 0.05;
        }
    }
    
    #pragma omp parallel for
    for (int i = 0; i < OUTPUT_NODES; i++)
    {
        bias2[i] = (double)rand() / RAND_MAX * 0.1 - 0.05;
    }

    // Load MNIST dataset
    printf("Loading MNIST dataset...\n");
    load_mnist();

    // Train the network
    printf("Starting training...\n");
    start = clock();

    for (int epoch = 0; epoch < NUMBER_OF_EPOCHS; epoch++)
    {
        train_parallel(epoch);
        printf("Epoch %d : Training Accuracy: %lf\n", epoch, (double)forward_prob_output / NUM_TRAINING_IMAGES);
        printf("Example weight: %lf\n", weight1[0][0]);
    }

    end = clock();
    seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("Time to train: %f seconds\n", seconds);

    save_weights_biases("model.bin");

    // Test the network
    printf("Starting testing...\n");
    start = clock();
    
    test_parallel();

    end = clock();
    seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("Time to test: %f seconds\n", seconds);
    printf("Testing Accuracy: %f\n", (double)correct_predictions / NUM_TEST_IMAGES);

    return 0;
}