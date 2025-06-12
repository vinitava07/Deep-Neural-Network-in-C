#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <string.h>

#define INPUT_NODES 784
#define HIDDEN_NODES 256
#define OUTPUT_NODES 10

#define NUM_TRAINING_IMAGES 8000
#define NUM_TEST_IMAGES 1000
#define NUMBER_OF_EPOCHS 10
#define BATCH_SIZE 128  // Aumentado para melhor eficiência na GPU

// Arrays globais
double training_images[NUM_TRAINING_IMAGES][INPUT_NODES];
double training_labels[NUM_TRAINING_IMAGES][OUTPUT_NODES];
double test_images[NUM_TEST_IMAGES][INPUT_NODES];
double test_labels[NUM_TEST_IMAGES][OUTPUT_NODES];

// Parâmetros da rede neural
double weight1[INPUT_NODES][HIDDEN_NODES];
double weight2[HIDDEN_NODES][OUTPUT_NODES];
double bias1[HIDDEN_NODES];
double bias2[OUTPUT_NODES];

// Estatísticas globais
int correct_predictions = 0;
int forward_prob_output = 0;

// Função sigmoid que funciona em CPU e GPU
#pragma omp declare target
double sigmoid_safe(double x) {
    if (x > 10.0) return 1.0;
    if (x < -10.0) return 0.0;
    // Usar 1/(1+exp(-x)) de forma segura
    double exp_neg_x = 1.0;
    if (x > -10.0 && x < 10.0) {
        // Aproximação de Taylor para exp(-x) quando x é pequeno
        double neg_x = -x;
        exp_neg_x = 1.0 + neg_x + (neg_x * neg_x) / 2.0 +
                    (neg_x * neg_x * neg_x) / 6.0 +
                    (neg_x * neg_x * neg_x * neg_x) / 24.0;
    } else if (x <= -10.0) {
        exp_neg_x = 1000000.0; // exp(10) aproximadamente
    } else {
        exp_neg_x = 0.0001; // exp(-10) aproximadamente
    }
    return 1.0 / (1.0 + exp_neg_x);
}

double relu_safe(double x) {
    return x > 0.0 ? x : 0.0;
}
#pragma omp end declare target

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
    for (int i = 0; i < NUM_TRAINING_IMAGES; i++) {
        for (int j = 0; j < INPUT_NODES; j++) {
            unsigned char pixel;
            fread(&pixel, 1, 1, training_images_file);
            training_images[i][j] = (double)pixel / 255.0;
        }
    }

    // Carregar labels de treino
    for (int i = 0; i < NUM_TRAINING_IMAGES; i++) {
        unsigned char label;
        fread(&label, 1, 1, training_labels_file);
        for (int j = 0; j < OUTPUT_NODES; j++) {
            training_labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }

    // Carregar imagens de teste
    for (int i = 0; i < NUM_TEST_IMAGES; i++) {
        for (int j = 0; j < INPUT_NODES; j++) {
            unsigned char pixel;
            fread(&pixel, 1, 1, test_images_file);
            test_images[i][j] = (double)pixel / 255.0;
        }
    }

    // Carregar labels de teste
    for (int i = 0; i < NUM_TEST_IMAGES; i++) {
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
// Processa batches inteiros na GPU
void train_gpu_optimized(int epoch) {
    double lr = 0.1;
    if (epoch % 3 == 0 && epoch > 0) {
        lr *= 0.1;
    }

    int correct_count = 0;

    #pragma omp target enter data map(to: weight1[0:INPUT_NODES][0:HIDDEN_NODES], \
                                         weight2[0:HIDDEN_NODES][0:OUTPUT_NODES], \
                                         bias1[0:HIDDEN_NODES], \
                                         bias2[0:OUTPUT_NODES])

    for (int batch_start = 0; batch_start < NUM_TRAINING_IMAGES; batch_start += BATCH_SIZE) {
        int batch_end = batch_start + BATCH_SIZE;
        if (batch_end > NUM_TRAINING_IMAGES) batch_end = NUM_TRAINING_IMAGES;
        int current_batch_size = batch_end - batch_start;

        // Arrays para o batch atual
        double batch_hidden[BATCH_SIZE][HIDDEN_NODES];
        double batch_output[BATCH_SIZE][OUTPUT_NODES];
        double batch_delta_out[BATCH_SIZE][OUTPUT_NODES];
        double batch_delta_hidden[BATCH_SIZE][HIDDEN_NODES];

        double batch_images[BATCH_SIZE][INPUT_NODES];
        double batch_labels[BATCH_SIZE][OUTPUT_NODES];

        for (int i = 0; i < current_batch_size; i++) {
            memcpy(batch_images[i], training_images[batch_start + i], INPUT_NODES * sizeof(double));
            memcpy(batch_labels[i], training_labels[batch_start + i], OUTPUT_NODES * sizeof(double));
        }

        // FORWARD PASS
        #pragma omp target teams distribute parallel for collapse(2) \
                map(to: batch_images[0:current_batch_size][0:INPUT_NODES], \
                       batch_labels[0:current_batch_size][0:OUTPUT_NODES]) \
                map(from: batch_hidden[0:current_batch_size][0:HIDDEN_NODES], \
                         batch_output[0:current_batch_size][0:OUTPUT_NODES])
        for (int img = 0; img < current_batch_size; img++) {
            for (int h = 0; h < HIDDEN_NODES; h++) {
                double sum = bias1[h];
                for (int i = 0; i < INPUT_NODES; i++) {
                    sum += batch_images[img][i] * weight1[i][h];
                }
                batch_hidden[img][h] = relu_safe(sum);
            }
        }

        // Segunda camada
        #pragma omp target teams distribute parallel for collapse(2) \
                map(to: batch_hidden[0:current_batch_size][0:HIDDEN_NODES]) \
                map(tofrom: batch_output[0:current_batch_size][0:OUTPUT_NODES])
        for (int img = 0; img < current_batch_size; img++) {
            for (int o = 0; o < OUTPUT_NODES; o++) {
                double sum = bias2[o];
                for (int h = 0; h < HIDDEN_NODES; h++) {
                    sum += batch_hidden[img][h] * weight2[h][o];
                }
                batch_output[img][o] = sigmoid_safe(sum);
            }
        }

        // Calcular acurácia
        for (int img = 0; img < current_batch_size; img++) {
            int pred = 0;
            double max_val = batch_output[img][0];
            for (int i = 1; i < OUTPUT_NODES; i++) {
                if (batch_output[img][i] > max_val) {
                    max_val = batch_output[img][i];
                    pred = i;
                }
            }

            int label = 0;
            for (int i = 0; i < OUTPUT_NODES; i++) {
                if (batch_labels[img][i] == 1.0) {
                    label = i;
                    break;
                }
            }

            if (pred == label) correct_count++;
        }

        // BACKWARD PASS
        #pragma omp target teams distribute parallel for collapse(2) \
                map(to: batch_output[0:current_batch_size][0:OUTPUT_NODES], \
                       batch_labels[0:current_batch_size][0:OUTPUT_NODES]) \
                map(from: batch_delta_out[0:current_batch_size][0:OUTPUT_NODES])
        for (int img = 0; img < current_batch_size; img++) {
            for (int o = 0; o < OUTPUT_NODES; o++) {
                double a = batch_output[img][o];
                batch_delta_out[img][o] = (a - batch_labels[img][o]) * a * (1.0 - a);
            }
        }

        // Deltas da camada oculta
        #pragma omp target teams distribute parallel for collapse(2) \
                map(to: batch_delta_out[0:current_batch_size][0:OUTPUT_NODES], \
                       batch_hidden[0:current_batch_size][0:HIDDEN_NODES]) \
                map(from: batch_delta_hidden[0:current_batch_size][0:HIDDEN_NODES])
        for (int img = 0; img < current_batch_size; img++) {
            for (int h = 0; h < HIDDEN_NODES; h++) {
                double sum = 0.0;
                for (int o = 0; o < OUTPUT_NODES; o++) {
                    sum += batch_delta_out[img][o] * weight2[h][o];
                }
                batch_delta_hidden[img][h] = sum * (batch_hidden[img][h] > 0.0 ? 1.0 : 0.0);
            }
        }

        // ATUALIZAÇÃO DE PESOS
        // Weight2
        #pragma omp target teams distribute parallel for collapse(2) \
                map(to: batch_delta_out[0:current_batch_size][0:OUTPUT_NODES], \
                       batch_hidden[0:current_batch_size][0:HIDDEN_NODES])
        for (int h = 0; h < HIDDEN_NODES; h++) {
            for (int o = 0; o < OUTPUT_NODES; o++) {
                double gradient = 0.0;
                for (int img = 0; img < current_batch_size; img++) {
                    gradient += batch_delta_out[img][o] * batch_hidden[img][h];
                }
                weight2[h][o] -= lr * gradient / current_batch_size;
            }
        }

        // Bias2
        #pragma omp target teams distribute parallel for \
                map(to: batch_delta_out[0:current_batch_size][0:OUTPUT_NODES])
        for (int o = 0; o < OUTPUT_NODES; o++) {
            double gradient = 0.0;
            for (int img = 0; img < current_batch_size; img++) {
                gradient += batch_delta_out[img][o];
            }
            bias2[o] -= lr * gradient / current_batch_size;
        }

        // Weight1
        #pragma omp target teams distribute parallel for collapse(2) \
                map(to: batch_delta_hidden[0:current_batch_size][0:HIDDEN_NODES], \
                       batch_images[0:current_batch_size][0:INPUT_NODES])
        for (int i = 0; i < INPUT_NODES; i++) {
            for (int h = 0; h < HIDDEN_NODES; h++) {
                double gradient = 0.0;
                for (int img = 0; img < current_batch_size; img++) {
                    gradient += batch_delta_hidden[img][h] * batch_images[img][i];
                }
                weight1[i][h] -= lr * gradient / current_batch_size;
            }
        }

        // Bias1
        #pragma omp target teams distribute parallel for \
                map(to: batch_delta_hidden[0:current_batch_size][0:HIDDEN_NODES])
        for (int h = 0; h < HIDDEN_NODES; h++) {
            double gradient = 0.0;
            for (int img = 0; img < current_batch_size; img++) {
                gradient += batch_delta_hidden[img][h];
            }
            bias1[h] -= lr * gradient / current_batch_size;
        }
    }

    // Trazer pesos de volta apenas no final da época
    #pragma omp target exit data map(from: weight1[0:INPUT_NODES][0:HIDDEN_NODES], \
                                          weight2[0:HIDDEN_NODES][0:OUTPUT_NODES], \
                                          bias1[0:HIDDEN_NODES], \
                                          bias2[0:OUTPUT_NODES])

    forward_prob_output = correct_count;
}

// Versão otimizada do teste
void test_gpu_optimized() {
    correct_predictions = 0;

    #pragma omp target enter data map(to: weight1[0:INPUT_NODES][0:HIDDEN_NODES], \
                                         weight2[0:HIDDEN_NODES][0:OUTPUT_NODES], \
                                         bias1[0:HIDDEN_NODES], \
                                         bias2[0:OUTPUT_NODES])

    // Processar em batches
    for (int batch_start = 0; batch_start < NUM_TEST_IMAGES; batch_start += BATCH_SIZE) {
        int batch_end = batch_start + BATCH_SIZE;
        if (batch_end > NUM_TEST_IMAGES) batch_end = NUM_TEST_IMAGES;
        int current_batch_size = batch_end - batch_start;

        double batch_images[BATCH_SIZE][INPUT_NODES];
        double batch_labels[BATCH_SIZE][OUTPUT_NODES];
        double batch_output[BATCH_SIZE][OUTPUT_NODES];

        // Copiar batch
        for (int i = 0; i < current_batch_size; i++) {
            memcpy(batch_images[i], test_images[batch_start + i], INPUT_NODES * sizeof(double));
            memcpy(batch_labels[i], test_labels[batch_start + i], OUTPUT_NODES * sizeof(double));
        }

        // Forward pass completo na GPU
        #pragma omp target teams distribute parallel for \
                map(to: batch_images[0:current_batch_size][0:INPUT_NODES]) \
                map(from: batch_output[0:current_batch_size][0:OUTPUT_NODES])
        for (int img = 0; img < current_batch_size; img++) {
            // Camada oculta
            double hidden[HIDDEN_NODES];
            for (int h = 0; h < HIDDEN_NODES; h++) {
                double sum = bias1[h];
                for (int i = 0; i < INPUT_NODES; i++) {
                    sum += batch_images[img][i] * weight1[i][h];
                }
                hidden[h] = relu_safe(sum);
            }

            // Camada de saída
            for (int o = 0; o < OUTPUT_NODES; o++) {
                double sum = bias2[o];
                for (int h = 0; h < HIDDEN_NODES; h++) {
                    sum += hidden[h] * weight2[h][o];
                }
                batch_output[img][o] = sigmoid_safe(sum);
            }
        }

        // Calcular acurácia (CPU)
        for (int img = 0; img < current_batch_size; img++) {
            int pred = 0;
            double max_val = batch_output[img][0];
            for (int i = 1; i < OUTPUT_NODES; i++) {
                if (batch_output[img][i] > max_val) {
                    max_val = batch_output[img][i];
                    pred = i;
                }
            }

            int label = 0;
            for (int i = 0; i < OUTPUT_NODES; i++) {
                if (batch_labels[img][i] == 1.0) {
                    label = i;
                    break;
                }
            }

            if (pred == label) correct_predictions++;
        }
    }

    // Liberar dados da GPU
    #pragma omp target exit data map(delete: weight1[0:INPUT_NODES][0:HIDDEN_NODES], \
                                            weight2[0:HIDDEN_NODES][0:OUTPUT_NODES], \
                                            bias1[0:HIDDEN_NODES], \
                                            bias2[0:OUTPUT_NODES])
}
int main() {
    printf("Rede Neural: %d -> %d -> %d\n", INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES);
    printf("Batch Size: %d\n", BATCH_SIZE);

    srand(42);

    double xavier_std1 = sqrt(2.0 / (INPUT_NODES + HIDDEN_NODES));
    for (int i = 0; i < INPUT_NODES; i++) {
        for (int j = 0; j < HIDDEN_NODES; j++) {
            weight1[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * xavier_std1;
        }
    }

    double xavier_std2 = sqrt(2.0 / (HIDDEN_NODES + OUTPUT_NODES));
    for (int i = 0; i < HIDDEN_NODES; i++) {
        bias1[i] = 0.01; // Pequeno bias positivo para ReLU
        for (int j = 0; j < OUTPUT_NODES; j++) {
            weight2[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * xavier_std2;
        }
    }

    // Bias inicializado com zeros
    for (int i = 0; i < OUTPUT_NODES; i++) {
        bias2[i] = 0.0;
    }

    // Carregar dados
    printf("Carregando dataset MNIST...\n");
    load_mnist();

    // Treinar
    printf("Learning rate: 0.1 (*0.1 a cada 3 épocas)\n");
    printf("Processamento em batches de %d imagens\n", BATCH_SIZE);

    clock_t total_start = clock();

    for (int epoch = 0; epoch < NUMBER_OF_EPOCHS; epoch++) {
        clock_t epoch_start = clock();

        train_gpu_optimized(epoch);

        clock_t epoch_end = clock();
        double epoch_time = (double)(epoch_end - epoch_start) / CLOCKS_PER_SEC;
        double accuracy = (double)forward_prob_output / NUM_TRAINING_IMAGES * 100.0;

        printf("Época %2d: Acurácia = %6.2f%% | Tempo = %5.2f seg\n",
               epoch + 1, accuracy, epoch_time);

    }

    clock_t total_end = clock();
    double total_time = (double)(total_end - total_start) / CLOCKS_PER_SEC;

    printf("Tempo total: %.2f segundos (%.2f seg/época)\n",
           total_time, total_time / NUMBER_OF_EPOCHS);

    // Testar
    printf("Executando teste final...\n");
    clock_t test_start = clock();

    test_gpu_optimized();

    clock_t test_end = clock();
    double test_time = (double)(test_end - test_start) / CLOCKS_PER_SEC;

    double test_accuracy = (double)correct_predictions / NUM_TEST_IMAGES * 100.0;

    printf("Acurácia de teste: %.2f%%\n", test_accuracy);
    printf("Tempo de teste: %.3f segundos\n", test_time);

    return 0;
}
