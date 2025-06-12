
#include <stdio.h>
#include <stdlib.h>
#include <string.h>   // memcpy
#include <math.h>
#include <omp.h>

#define INPUT_NODES    784
#define HIDDEN_NODES   128
#define OUTPUT_NODES   10
#define NUM_TRAIN      8000
#define NUM_TEST       1000
#define BATCH_SIZE     64
#define EPOCHS         10

// pesos e vieses
static double weight1[INPUT_NODES][HIDDEN_NODES];
static double weight2[HIDDEN_NODES][OUTPUT_NODES];
static double bias1[HIDDEN_NODES];
static double bias2[OUTPUT_NODES];

// dados de treino/teste
static double (*training_images)[INPUT_NODES];
static double (*training_labels)[OUTPUT_NODES];
static double (*test_images)[INPUT_NODES];
static double (*test_labels)[OUTPUT_NODES];

// Carregar MNIST
void load_mnist() {
    FILE *training_images_file = fopen("mnist_train_images.bin", "rb");
    FILE *training_labels_file = fopen("mnist_train_labels.bin", "rb");
    FILE *test_images_file = fopen("mnist_test_images.bin", "rb");
    FILE *test_labels_file = fopen("mnist_test_labels.bin", "rb");

    if (!training_images_file || !training_labels_file ||
        !test_images_file || !test_labels_file) {
        printf("Erro ao abrir arquivos MNIST\n");
        exit(1);
    }

    unsigned char t;

    // Pular headers
    for (int i = 0; i < 8; i++) {
        fread(&t, 1, 1, training_images_file);
        fread(&t, 1, 1, training_labels_file);
        fread(&t, 1, 1, test_images_file);
        fread(&t, 1, 1, test_labels_file);
    }

    // Carregar imagens de treino
    for (int i = 0; i < NUM_TRAIN; i++) {
        for (int j = 0; j < INPUT_NODES; j++) {
            unsigned char pixel;
            fread(&pixel, 1, 1, training_images_file);
            training_images[i][j] = (double)pixel / 255.0;
        }
    }

    // Carregar labels de treino
    for (int i = 0; i < NUM_TRAIN; i++) {
        unsigned char label;
        fread(&label, 1, 1, training_labels_file);
        for (int j = 0; j < OUTPUT_NODES; j++) {
            training_labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }

    // Carregar imagens de teste
    for (int i = 0; i < NUM_TEST; i++) {
        for (int j = 0; j < INPUT_NODES; j++) {
            unsigned char pixel;
            fread(&pixel, 1, 1, test_images_file);
            test_images[i][j] = (double)pixel / 255.0;
        }
    }

    // Carregar labels de teste
    for (int i = 0; i < NUM_TEST; i++) {
        unsigned char label;
        fread(&label, 1, 1, test_labels_file);
        for (int j = 0; j < OUTPUT_NODES; j++) {
            test_labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }

    fclose(training_images_file);
    fclose(training_labels_file);
    fclose(test_images_file);
    fclose(test_labels_file);
}

#pragma omp declare target
double fast_exp(double x) {
    x = 1.0 + x / 1024.0;
    x *= x; x *= x; x *= x; x *= x;
    x *= x; x *= x; x *= x; x *= x;
    x *= x; x *= x;
    return x;
}

double relu_safe(double x) {
    return x > 0 ? x : 0;
}

double sigmoid_safe(double x) {
    return 1.0 / (1.0 + fast_exp(-x));
}
#pragma omp end declare target
void train_gpu_optimized(int epoch);
void test_gpu();

int main() {
    training_images = malloc(NUM_TRAIN * sizeof *training_images);
    training_labels = malloc(NUM_TRAIN * sizeof *training_labels);
    test_images     = malloc(NUM_TEST  * sizeof *test_images);
    test_labels     = malloc(NUM_TEST  * sizeof *test_labels);

    load_mnist();

    srand(123);
    #pragma omp target enter data map(to: weight1, weight2, bias1, bias2)

    #pragma omp target
    {
        // Inicializando weight1 com valores aleatórios
        #pragma omp teams distribute parallel for collapse(2)
        for(int input_idx = 0; input_idx < INPUT_NODES; input_idx++) {
            for(int hidden_idx = 0; hidden_idx < HIDDEN_NODES; hidden_idx++) {
                weight1[input_idx][hidden_idx] = (rand()/(double)RAND_MAX - 0.5) * 0.1;
            }
        }

        // Inicializando bias1 e weight2
        #pragma omp teams distribute parallel for
        for(int hidden_idx = 0; hidden_idx < HIDDEN_NODES; hidden_idx++) {
            bias1[hidden_idx] = 0;
            for(int output_idx = 0; output_idx < OUTPUT_NODES; output_idx++) {
                weight2[hidden_idx][output_idx] = (rand()/(double)RAND_MAX - 0.5) * 0.1;
            }
        }

        // Inicializando bias2
        #pragma omp teams distribute parallel for
        for(int output_idx = 0; output_idx < OUTPUT_NODES; output_idx++) {
            bias2[output_idx] = 0;
        }
    }

    // Loop de treinamento
    double total_start = omp_get_wtime();
    for(int epoch_idx = 0; epoch_idx < EPOCHS; epoch_idx++) {
        double epoch_start = omp_get_wtime();
        train_gpu_optimized(epoch_idx); // Função de treinamento
        double epoch_end = omp_get_wtime();
        printf("Época %2d: Tempo = %5.2f seg\n", epoch_idx+1, epoch_end - epoch_start);
    }
    double total_end = omp_get_wtime();
    printf("Tempo total treino: %.2f s\n", total_end - total_start);

    // Retirando os dados do dispositivo
    #pragma omp target exit data map(from: weight1, weight2, bias1, bias2)
    test_gpu();

    free(training_images);
    free(training_labels);
    free(test_images);
    free(test_labels);
    return 0;
}

void train_gpu_optimized(int epoch) {
    double learning_rate = 0.1 * pow(0.1, epoch/3);
    int num_batches = (NUM_TRAIN + BATCH_SIZE - 1) / BATCH_SIZE;

    for(int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        int start = batch_idx * BATCH_SIZE;
        int batch_size = (start + BATCH_SIZE > NUM_TRAIN) ? (NUM_TRAIN - start) : BATCH_SIZE;

        double batch_images[BATCH_SIZE][INPUT_NODES];
        double batch_labels[BATCH_SIZE][OUTPUT_NODES];
        #pragma omp parallel for simd
        for(int img = 0; img < batch_size; img++) {
            memcpy(batch_images[img], training_images[start + img], INPUT_NODES * sizeof(double));
            memcpy(batch_labels[img], training_labels[start + img], OUTPUT_NODES * sizeof(double));
        }

        double batch_hidden[BATCH_SIZE][HIDDEN_NODES];
        double batch_output[BATCH_SIZE][OUTPUT_NODES];
        double batch_delta_out[BATCH_SIZE][OUTPUT_NODES];
        double batch_delta_hidden[BATCH_SIZE][HIDDEN_NODES];

        // FORWARD HIDDEN
        #pragma omp target teams distribute parallel for collapse(2) map(to: batch_images[0:batch_size][0:INPUT_NODES]) map(from: batch_hidden[0:batch_size][0:HIDDEN_NODES])
        for(int img = 0; img < batch_size; img++) {
            for(int hid = 0; hid < HIDDEN_NODES; hid++) {
                double sum = bias1[hid];
                for(int inp = 0; inp < INPUT_NODES; inp++) sum += batch_images[img][inp] * weight1[inp][hid];
                batch_hidden[img][hid] = relu_safe(sum);
            }
        }
        // FORWARD OUTPUT
        #pragma omp target teams distribute parallel for collapse(2) map(to: batch_hidden[0:batch_size][0:HIDDEN_NODES]) map(from: batch_output[0:batch_size][0:OUTPUT_NODES])
        for(int img = 0; img < batch_size; img++) {
            for(int out = 0; out < OUTPUT_NODES; out++) {
                double sum = bias2[out];
                for(int hid = 0; hid < HIDDEN_NODES; hid++) sum += batch_hidden[img][hid] * weight2[hid][out];
                batch_output[img][out] = sigmoid_safe(sum);
            }
        }

        // BACKWARD OUTPUT DELTA
        #pragma omp target teams distribute parallel for collapse(2) map(to: batch_output[0:batch_size][0:OUTPUT_NODES], batch_labels[0:batch_size][0:OUTPUT_NODES]) map(from: batch_delta_out[0:batch_size][0:OUTPUT_NODES])
        for(int img = 0; img < batch_size; img++) {
            for(int out = 0; out < OUTPUT_NODES; out++) {
                double a = batch_output[img][out];
                batch_delta_out[img][out] = (a - batch_labels[img][out]) * a * (1.0 - a);
            }
        }
        // BACKWARD HIDDEN DELTA
        #pragma omp target teams distribute parallel for collapse(2) map(to: batch_delta_out[0:batch_size][0:OUTPUT_NODES], batch_hidden[0:batch_size][0:HIDDEN_NODES]) map(from: batch_delta_hidden[0:batch_size][0:HIDDEN_NODES])
        for(int img = 0; img < batch_size; img++) {
            for(int hid = 0; hid < HIDDEN_NODES; hid++) {
                double sum = 0;
                for(int out = 0; out < OUTPUT_NODES; out++) sum += batch_delta_out[img][out] * weight2[hid][out];
                batch_delta_hidden[img][hid] = sum * (batch_hidden[img][hid] > 0);
            }
        }

        // UPDATE weight2 & bias2
        #pragma omp target teams distribute parallel for collapse(2) map(to: batch_delta_out[0:batch_size][0:OUTPUT_NODES], batch_hidden[0:batch_size][0:HIDDEN_NODES])
        for(int hid = 0; hid < HIDDEN_NODES; hid++) {
            for(int out = 0; out < OUTPUT_NODES; out++) {
                double grad = 0;
                for(int img = 0; img < batch_size; img++) grad += batch_delta_out[img][out] * batch_hidden[img][hid];
                weight2[hid][out] -= learning_rate * grad / batch_size;
            }
        }
        #pragma omp target teams distribute parallel for map(to: batch_delta_out[0:batch_size][0:OUTPUT_NODES])
        for(int out = 0; out < OUTPUT_NODES; out++) {
            double grad = 0;
            for(int img = 0; img < batch_size; img++) grad += batch_delta_out[img][out];
            bias2[out] -= learning_rate * grad / batch_size;
        }

        // UPDATE weight1 & bias1
        #pragma omp target teams distribute parallel for collapse(2) map(to: batch_delta_hidden[0:batch_size][0:HIDDEN_NODES], batch_images[0:batch_size][0:INPUT_NODES])
        for(int inp = 0; inp < INPUT_NODES; inp++) {
            for(int hid = 0; hid < HIDDEN_NODES; hid++) {
                double grad = 0;
                for(int img = 0; img < batch_size; img++) grad += batch_delta_hidden[img][hid] * batch_images[img][inp];
                weight1[inp][hid] -= learning_rate * grad / batch_size;
            }
        }
        #pragma omp target teams distribute parallel for map(to: batch_delta_hidden[0:batch_size][0:HIDDEN_NODES])
        for(int hid = 0; hid < HIDDEN_NODES; hid++) {
            double grad = 0;
            for(int img = 0; img < batch_size; img++) grad += batch_delta_hidden[img][hid];
            bias1[hid] -= learning_rate * grad / batch_size;
        }
    }
}

void test_gpu() {
    int correct = 0;

    #pragma omp target map(to: test_images[0:NUM_TEST][0:INPUT_NODES], weight1[0:INPUT_NODES][0:HIDDEN_NODES], bias1[0:HIDDEN_NODES], weight2[0:HIDDEN_NODES][0:OUTPUT_NODES], bias2[0:OUTPUT_NODES], test_labels[0:NUM_TEST][0:OUTPUT_NODES]) map(tofrom: correct)
    #pragma omp teams distribute parallel for reduction(+:correct)
    for (int img = 0; img < NUM_TEST; img++) {
        double hidden[HIDDEN_NODES];
        double output[OUTPUT_NODES];

        for (int hid = 0; hid < HIDDEN_NODES; hid++) {
            double sum = bias1[hid];
            for (int inp = 0; inp < INPUT_NODES; inp++) {
                sum += test_images[img][inp] * weight1[inp][hid];
            }
            hidden[hid] = relu_safe(sum);
        }

        for (int out = 0; out < OUTPUT_NODES; out++) {
            double sum = bias2[out];
            for (int hid = 0; hid < HIDDEN_NODES; hid++) {
                sum += hidden[hid] * weight2[hid][out];
            }
            output[out] = sigmoid_safe(sum);
        }

        int pred = 0;
        for (int out = 1; out < OUTPUT_NODES; out++) {
            if (output[out] > output[pred]) pred = out;
        }

        int act = 0;
        for (int out = 0; out < OUTPUT_NODES; out++) {
            if (test_labels[img][out] == 1.0) act = out;
        }

        if (pred == act) correct++;
    }

    printf("Accuracy teste: %.2f%%\n", correct * 100.0 / NUM_TEST);
}
