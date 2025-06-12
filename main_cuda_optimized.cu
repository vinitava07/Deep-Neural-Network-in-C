/*
CUDA Otimizado - Processamento em Mini-batches
Versão corrigida com cálculo de acurácia por época
Compilar com: nvcc -O3 -arch=sm_61 main_cuda_optimized.cu -o cuda_optimized
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define INPUT_NODES 784
#define HIDDEN_NODES 256
#define OUTPUT_NODES 10
#define NUM_TRAINING_IMAGES 8000
#define NUM_TEST_IMAGES 1000
#define NUMBER_OF_EPOCHS 10

// Configurações de otimização
#define BATCH_SIZE 64
#define THREADS_PER_BLOCK 256

// Arrays do host
double h_training_images[NUM_TRAINING_IMAGES][INPUT_NODES];
double h_training_labels[NUM_TRAINING_IMAGES][OUTPUT_NODES];
double h_test_images[NUM_TEST_IMAGES][INPUT_NODES];
double h_test_labels[NUM_TEST_IMAGES][OUTPUT_NODES];

double h_weight1[INPUT_NODES][HIDDEN_NODES];
double h_weight2[HIDDEN_NODES][OUTPUT_NODES];
double h_bias1[HIDDEN_NODES];
double h_bias2[OUTPUT_NODES];

// Ponteiros do device
double *d_training_images, *d_training_labels;
double *d_test_images, *d_test_labels;
double *d_weight1, *d_weight2, *d_bias1, *d_bias2;
double *d_batch_input, *d_batch_target;
double *d_batch_hidden, *d_batch_output;
double *d_batch_delta_out, *d_batch_delta_hidden;
double *d_grad_weight1, *d_grad_weight2;
double *d_grad_bias1, *d_grad_bias2;
int *d_batch_predictions;  // Para calcular acurácia
// Verificação de erros CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("Erro CUDA em %s:%d - %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Funções de ativação no device
__device__ double dev_sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

__device__ double dev_relu(double x) {
    return fmax(0.0, x);
}

__device__ double dev_relu_derivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

// Kernel para calcular acurácia em um batch
__global__ void calculate_batch_accuracy(
    double *batch_output,    // [BATCH_SIZE][OUTPUT_NODES]
    double *batch_target,    // [BATCH_SIZE][OUTPUT_NODES]
    int *predictions,        // [BATCH_SIZE] - saída com predições corretas
    int batch_size
) {
    int sample = blockIdx.x * blockDim.x + threadIdx.x;

    if (sample < batch_size) {
        // Encontrar índice da classe predita
        int predicted = 0;
        double max_val = batch_output[sample * OUTPUT_NODES];

        for (int i = 1; i < OUTPUT_NODES; i++) {
            double val = batch_output[sample * OUTPUT_NODES + i];
            if (val > max_val) {
                max_val = val;
                predicted = i;
            }
        }

        // Encontrar índice da classe correta
        int actual = 0;
        for (int i = 0; i < OUTPUT_NODES; i++) {
            if (batch_target[sample * OUTPUT_NODES + i] > 0.5) {
                actual = i;
                break;
            }
        }

        // Marcar 1 se correto, 0 se incorreto
        predictions[sample] = (predicted == actual) ? 1 : 0;
    }
}

// Kernel otimizado para forward pass
__global__ void forward_pass_batch(
    double *batch_input,    // [BATCH_SIZE][INPUT_NODES]
    double *weight1,        // [INPUT_NODES][HIDDEN_NODES]
    double *bias1,          // [HIDDEN_NODES]
    double *batch_hidden,   // [BATCH_SIZE][HIDDEN_NODES]
    double *weight2,        // [HIDDEN_NODES][OUTPUT_NODES]
    double *bias2,          // [OUTPUT_NODES]
    double *batch_output,   // [BATCH_SIZE][OUTPUT_NODES]
    int batch_size
) {
    int sample = blockIdx.y;
    if (sample >= batch_size) return;

    // Primeira camada: Input -> Hidden
    int hid_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (hid_idx < HIDDEN_NODES) {
        double sum = bias1[hid_idx];

        // Loop desenrolado para melhor desempenho
        for (int i = 0; i < INPUT_NODES; i += 4) {
            if (i < INPUT_NODES)
                sum += batch_input[sample * INPUT_NODES + i] *
                       weight1[i * HIDDEN_NODES + hid_idx];
            if (i + 1 < INPUT_NODES)
                sum += batch_input[sample * INPUT_NODES + i + 1] *
                       weight1[(i + 1) * HIDDEN_NODES + hid_idx];
            if (i + 2 < INPUT_NODES)
                sum += batch_input[sample * INPUT_NODES + i + 2] *
                       weight1[(i + 2) * HIDDEN_NODES + hid_idx];
            if (i + 3 < INPUT_NODES)
                sum += batch_input[sample * INPUT_NODES + i + 3] *
                       weight1[(i + 3) * HIDDEN_NODES + hid_idx];
        }

        batch_hidden[sample * HIDDEN_NODES + hid_idx] = dev_relu(sum);
    }
}

// Kernel separado para segunda camada
__global__ void forward_pass_output(
    double *batch_hidden,   // [BATCH_SIZE][HIDDEN_NODES]
    double *weight2,        // [HIDDEN_NODES][OUTPUT_NODES]
    double *bias2,          // [OUTPUT_NODES]
    double *batch_output,   // [BATCH_SIZE][OUTPUT_NODES]
    int batch_size
) {
    int sample = blockIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (sample < batch_size && out_idx < OUTPUT_NODES) {
        double sum = bias2[out_idx];

        for (int i = 0; i < HIDDEN_NODES; i++) {
            sum += batch_hidden[sample * HIDDEN_NODES + i] *
                   weight2[i * OUTPUT_NODES + out_idx];
        }

        batch_output[sample * OUTPUT_NODES + out_idx] = dev_sigmoid(sum);
    }
}
// Kernel para calcular deltas de saída
__global__ void compute_output_deltas_batch(
    double *batch_output,
    double *batch_target,
    double *batch_delta_out,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sample = blockIdx.y;

    if (sample < batch_size && idx < OUTPUT_NODES) {
        int pos = sample * OUTPUT_NODES + idx;
        double out = batch_output[pos];
        // Derivada da função de custo com sigmoid
        batch_delta_out[pos] = (out - batch_target[pos]) * out * (1.0 - out);
    }
}

// Kernel para calcular deltas da camada oculta
__global__ void compute_hidden_deltas_batch(
    double *batch_delta_out,
    double *weight2,
    double *batch_hidden,
    double *batch_delta_hidden,
    int batch_size
) {
    int sample = blockIdx.y;
    int hid_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (sample < batch_size && hid_idx < HIDDEN_NODES) {
        double sum = 0.0;

        // Propagar erro da camada de saída
        for (int j = 0; j < OUTPUT_NODES; j++) {
            sum += batch_delta_out[sample * OUTPUT_NODES + j] *
                   weight2[hid_idx * OUTPUT_NODES + j];
        }

        // Multiplicar pela derivada da ReLU
        int pos = sample * HIDDEN_NODES + hid_idx;
        batch_delta_hidden[pos] = sum * dev_relu_derivative(batch_hidden[pos]);
    }
}

// Kernel otimizado para acumular gradientes (weight2)
__global__ void accumulate_gradients_weight2(
    double *batch_hidden,
    double *batch_delta_out,
    double *grad_weight2,
    int batch_size
) {
    int hid_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (hid_idx < HIDDEN_NODES && out_idx < OUTPUT_NODES) {
        double sum = 0.0;

        // Acumular gradientes de todas as amostras do batch
        for (int sample = 0; sample < batch_size; sample++) {
            sum += batch_hidden[sample * HIDDEN_NODES + hid_idx] *
                   batch_delta_out[sample * OUTPUT_NODES + out_idx];
        }

        grad_weight2[hid_idx * OUTPUT_NODES + out_idx] = sum;
    }
}

// Kernel similar para weight1
__global__ void accumulate_gradients_weight1(
    double *batch_input,
    double *batch_delta_hidden,
    double *grad_weight1,
    int batch_size
) {
    int inp_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int hid_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (inp_idx < INPUT_NODES && hid_idx < HIDDEN_NODES) {
        double sum = 0.0;

        for (int sample = 0; sample < batch_size; sample++) {
            sum += batch_input[sample * INPUT_NODES + inp_idx] *
                   batch_delta_hidden[sample * HIDDEN_NODES + hid_idx];
        }

        grad_weight1[inp_idx * HIDDEN_NODES + hid_idx] = sum;
    }
}

// Kernel para acumular gradientes de bias
__global__ void accumulate_gradients_bias(
    double *batch_delta_out,
    double *batch_delta_hidden,
    double *grad_bias2,
    double *grad_bias1,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Gradientes para bias2
    if (idx < OUTPUT_NODES) {
        double sum = 0.0;
        for (int sample = 0; sample < batch_size; sample++) {
            sum += batch_delta_out[sample * OUTPUT_NODES + idx];
        }
        grad_bias2[idx] = sum;
    }

    // Gradientes para bias1
    if (idx < HIDDEN_NODES) {
        double sum = 0.0;
        for (int sample = 0; sample < batch_size; sample++) {
            sum += batch_delta_hidden[sample * HIDDEN_NODES + idx];
        }
        grad_bias1[idx] = sum;
    }
}

// Kernel para atualizar pesos
__global__ void update_weights(
    double *weights,
    double *gradients,
    double learning_rate,
    int size,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Aplicar gradiente médio (dividir pelo tamanho do batch)
        weights[idx] -= learning_rate * (gradients[idx] / batch_size);
    }
}
// Função para carregar MNIST com tratamento de erros apropriado
void load_mnist() {
    // Abrir todos os arquivos e verificar se foram abertos corretamente
    FILE *train_img = fopen("mnist_train_images.bin", "rb");
    FILE *train_lbl = fopen("mnist_train_labels.bin", "rb");
    FILE *test_img = fopen("mnist_test_images.bin", "rb");
    FILE *test_lbl = fopen("mnist_test_labels.bin", "rb");

    if (!train_img || !train_lbl || !test_img || !test_lbl) {
        printf("Erro ao abrir arquivos MNIST\n");
        exit(1);
    }

    // O formato MNIST tem 8 bytes de cabeçalho que precisamos pular
    unsigned char t;
    for (int i = 0; i < 8; i++) {
        if (fread(&t, 1, 1, train_img) != 1 ||
            fread(&t, 1, 1, train_lbl) != 1 ||
            fread(&t, 1, 1, test_img) != 1 ||
            fread(&t, 1, 1, test_lbl) != 1) {
            printf("Erro ao ler cabeçalhos MNIST\n");
            exit(1);
        }
    }

    // Carregar imagens de treino - cada pixel é normalizado para [0,1]
    for (int i = 0; i < NUM_TRAINING_IMAGES; i++) {
        for (int j = 0; j < INPUT_NODES; j++) {
            unsigned char pixel;
            if (fread(&pixel, 1, 1, train_img) != 1) {
                printf("Erro lendo pixel de treino\n");
                exit(1);
            }
            h_training_images[i][j] = pixel / 255.0;
        }

        // Carregar labels e converter para one-hot encoding
        unsigned char label;
        if (fread(&label, 1, 1, train_lbl) != 1) {
            printf("Erro lendo label de treino\n");
            exit(1);
        }
        for (int j = 0; j < OUTPUT_NODES; j++) {
            h_training_labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }

    // Carregar imagens de teste com o mesmo processo
    for (int i = 0; i < NUM_TEST_IMAGES; i++) {
        for (int j = 0; j < INPUT_NODES; j++) {
            unsigned char pixel;
            if (fread(&pixel, 1, 1, test_img) != 1) {
                printf("Erro lendo pixel de teste\n");
                exit(1);
            }
            h_test_images[i][j] = pixel / 255.0;
        }

        unsigned char label;
        if (fread(&label, 1, 1, test_lbl) != 1) {
            printf("Erro lendo label de teste\n");
            exit(1);
        }
        for (int j = 0; j < OUTPUT_NODES; j++) {
            h_test_labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }

    fclose(train_img);
    fclose(train_lbl);
    fclose(test_img);
    fclose(test_lbl);
    printf("Dataset MNIST carregado com sucesso\n");
}
int main() {
    printf("CUDA Otimizado - Processamento em Mini-batches\n");
    printf("Compilado para GT 1030 (arquitetura Pascal sm_61)\n");

    // Verificar propriedades da GPU
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("Nenhuma GPU CUDA encontrada!\n");
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU detectada: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Multiprocessadores: %d\n", prop.multiProcessorCount);
    printf("\n");

    // Carregar dados MNIST
    load_mnist();

    // Inicializar pesos e bias com valores pequenos aleatórios
    // Isso ajuda a quebrar simetria e permite aprendizado efetivo
    srand(42);
    for (int i = 0; i < INPUT_NODES; i++) {
        for (int j = 0; j < HIDDEN_NODES; j++) {
            h_weight1[i][j] = (double)rand() / RAND_MAX * 0.1 - 0.05;
        }
    }
    for (int i = 0; i < HIDDEN_NODES; i++) {
        h_bias1[i] = (double)rand() / RAND_MAX * 0.1 - 0.05;
        for (int j = 0; j < OUTPUT_NODES; j++) {
            h_weight2[i][j] = (double)rand() / RAND_MAX * 0.1 - 0.05;
        }
    }
    for (int i = 0; i < OUTPUT_NODES; i++) {
        h_bias2[i] = (double)rand() / RAND_MAX * 0.1 - 0.05;
    }

    // Calcular tamanhos de memória necessários
    size_t img_size = NUM_TRAINING_IMAGES * INPUT_NODES * sizeof(double);
    size_t lbl_size = NUM_TRAINING_IMAGES * OUTPUT_NODES * sizeof(double);
    size_t w1_size = INPUT_NODES * HIDDEN_NODES * sizeof(double);
    size_t w2_size = HIDDEN_NODES * OUTPUT_NODES * sizeof(double);
    size_t b1_size = HIDDEN_NODES * sizeof(double);
    size_t b2_size = OUTPUT_NODES * sizeof(double);

    // Alocar memória na GPU para dados completos
    CUDA_CHECK(cudaMalloc(&d_training_images, img_size));
    CUDA_CHECK(cudaMalloc(&d_training_labels, lbl_size));
    CUDA_CHECK(cudaMalloc(&d_weight1, w1_size));
    CUDA_CHECK(cudaMalloc(&d_weight2, w2_size));
    CUDA_CHECK(cudaMalloc(&d_bias1, b1_size));
    CUDA_CHECK(cudaMalloc(&d_bias2, b2_size));

    // Alocar memória para processamento em batch
    // Processar em batches reduz conflitos de memória e melhora eficiência
    size_t batch_img_size = BATCH_SIZE * INPUT_NODES * sizeof(double);
    size_t batch_lbl_size = BATCH_SIZE * OUTPUT_NODES * sizeof(double);
    size_t batch_hid_size = BATCH_SIZE * HIDDEN_NODES * sizeof(double);
    size_t batch_out_size = BATCH_SIZE * OUTPUT_NODES * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_batch_input, batch_img_size));
    CUDA_CHECK(cudaMalloc(&d_batch_target, batch_lbl_size));
    CUDA_CHECK(cudaMalloc(&d_batch_hidden, batch_hid_size));
    CUDA_CHECK(cudaMalloc(&d_batch_output, batch_out_size));
    CUDA_CHECK(cudaMalloc(&d_batch_delta_out, batch_out_size));
    CUDA_CHECK(cudaMalloc(&d_batch_delta_hidden, batch_hid_size));
    CUDA_CHECK(cudaMalloc(&d_batch_predictions, BATCH_SIZE * sizeof(int)));

    // Alocar memória para gradientes acumulados
    CUDA_CHECK(cudaMalloc(&d_grad_weight1, w1_size));
    CUDA_CHECK(cudaMalloc(&d_grad_weight2, w2_size));
    CUDA_CHECK(cudaMalloc(&d_grad_bias1, b1_size));
    CUDA_CHECK(cudaMalloc(&d_grad_bias2, b2_size));

    // Copiar dados iniciais para GPU
    CUDA_CHECK(cudaMemcpy(d_training_images, h_training_images, img_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_training_labels, h_training_labels, lbl_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight1, h_weight1, w1_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight2, h_weight2, w2_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias1, h_bias1, b1_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias2, h_bias2, b2_size, cudaMemcpyHostToDevice));

    // Configurar dimensões de grid para diferentes kernels
    dim3 hidden_grid((HIDDEN_NODES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, BATCH_SIZE);
    dim3 output_grid((OUTPUT_NODES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, BATCH_SIZE);
    dim3 weight1_grid((HIDDEN_NODES + 15) / 16, (INPUT_NODES + 15) / 16);
    dim3 weight2_grid((OUTPUT_NODES + 15) / 16, (HIDDEN_NODES + 15) / 16);
    dim3 block_16x16(16, 16);

    printf("Iniciando treinamento...\n");
    clock_t start_time = clock();
    double learning_rate = 0.1;

    // Array temporário para armazenar predições do host
    int h_batch_predictions[BATCH_SIZE];
    // Loop principal de treinamento
    for (int epoch = 0; epoch < NUMBER_OF_EPOCHS; epoch++) {
        int num_batches = NUM_TRAINING_IMAGES / BATCH_SIZE;
        int epoch_correct = 0;  // Contador de predições corretas na época

        // Processar cada batch de imagens
        for (int batch = 0; batch < num_batches; batch++) {
            int batch_start = batch * BATCH_SIZE;

            // Copiar o batch atual para GPU
            // Fazemos isso por batch para economizar memória e melhorar cache
            CUDA_CHECK(cudaMemcpy(d_batch_input,
                                  &h_training_images[batch_start][0],
                                  batch_img_size, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_batch_target,
                                  &h_training_labels[batch_start][0],
                                  batch_lbl_size, cudaMemcpyHostToDevice));

            // Forward pass - primeira camada
            forward_pass_batch<<<hidden_grid, THREADS_PER_BLOCK>>>(
                d_batch_input, d_weight1, d_bias1, d_batch_hidden,
                d_weight2, d_bias2, d_batch_output, BATCH_SIZE
            );

            // Forward pass - segunda camada
            forward_pass_output<<<output_grid, THREADS_PER_BLOCK>>>(
                d_batch_hidden, d_weight2, d_bias2, d_batch_output, BATCH_SIZE
            );

            // Calcular acurácia do batch
            dim3 acc_grid((BATCH_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
            calculate_batch_accuracy<<<acc_grid, THREADS_PER_BLOCK>>>(
                d_batch_output, d_batch_target, d_batch_predictions, BATCH_SIZE
            );

            // Copiar predições de volta para contar acertos
            CUDA_CHECK(cudaMemcpy(h_batch_predictions, d_batch_predictions,
                                  BATCH_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

            // Somar predições corretas do batch
            for (int i = 0; i < BATCH_SIZE; i++) {
                epoch_correct += h_batch_predictions[i];
            }

            // Backward pass - calcular deltas
            compute_output_deltas_batch<<<output_grid, THREADS_PER_BLOCK>>>(
                d_batch_output, d_batch_target, d_batch_delta_out, BATCH_SIZE
            );

            compute_hidden_deltas_batch<<<hidden_grid, THREADS_PER_BLOCK>>>(
                d_batch_delta_out, d_weight2, d_batch_hidden,
                d_batch_delta_hidden, BATCH_SIZE
            );

            // Acumular gradientes para atualização dos pesos
            accumulate_gradients_weight2<<<weight2_grid, block_16x16>>>(
                d_batch_hidden, d_batch_delta_out, d_grad_weight2, BATCH_SIZE
            );

            accumulate_gradients_weight1<<<weight1_grid, block_16x16>>>(
                d_batch_input, d_batch_delta_hidden, d_grad_weight1, BATCH_SIZE
            );

            dim3 bias_grid((fmax((float)HIDDEN_NODES, (float)OUTPUT_NODES) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
            accumulate_gradients_bias<<<bias_grid, THREADS_PER_BLOCK>>>(
                d_batch_delta_out, d_batch_delta_hidden,
                d_grad_bias2, d_grad_bias1, BATCH_SIZE
            );

            // Atualizar pesos usando os gradientes acumulados
            int w1_blocks = (INPUT_NODES * HIDDEN_NODES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            update_weights<<<w1_blocks, THREADS_PER_BLOCK>>>(
                d_weight1, d_grad_weight1, learning_rate,
                INPUT_NODES * HIDDEN_NODES, BATCH_SIZE
            );

            int w2_blocks = (HIDDEN_NODES * OUTPUT_NODES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            update_weights<<<w2_blocks, THREADS_PER_BLOCK>>>(
                d_weight2, d_grad_weight2, learning_rate,
                HIDDEN_NODES * OUTPUT_NODES, BATCH_SIZE
            );

            int b1_blocks = (HIDDEN_NODES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            update_weights<<<b1_blocks, THREADS_PER_BLOCK>>>(
                d_bias1, d_grad_bias1, learning_rate, HIDDEN_NODES, BATCH_SIZE
            );

            int b2_blocks = (OUTPUT_NODES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            update_weights<<<b2_blocks, THREADS_PER_BLOCK>>>(
                d_bias2, d_grad_bias2, learning_rate, OUTPUT_NODES, BATCH_SIZE
            );
        }

        // Sincronizar para garantir que todos os cálculos terminaram
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copiar um peso de exemplo para mostrar progresso
        double example_weight;
        CUDA_CHECK(cudaMemcpy(&example_weight, d_weight1, sizeof(double), cudaMemcpyDeviceToHost));

        // Mostrar acurácia da época (como no código original)
        double epoch_accuracy = (double)epoch_correct / NUM_TRAINING_IMAGES;
        printf("Epoch %d: Training Accuracy: %lf\n", epoch, epoch_accuracy);
        printf("Example weight: %lf\n", example_weight);

        // Ajustar taxa de aprendizado a cada 5 épocas
        if (epoch % 5 == 0 && epoch > 0) {
            learning_rate *= 0.5;
            printf("Taxa de aprendizado ajustada para %f\n", learning_rate);
        }
    }

    clock_t end_time = clock();
    printf("Tempo de treinamento: %.2f segundos\n",
           (double)(end_time - start_time) / CLOCKS_PER_SEC);
    // Fase de teste - avaliar o modelo treinado
    printf("\nIniciando teste...\n");
    start_time = clock();

    // Alocar memória para dados de teste na GPU
    size_t test_img_size = NUM_TEST_IMAGES * INPUT_NODES * sizeof(double);
    size_t test_lbl_size = NUM_TEST_IMAGES * OUTPUT_NODES * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_test_images, test_img_size));
    CUDA_CHECK(cudaMalloc(&d_test_labels, test_lbl_size));

    // Copiar dados de teste para GPU
    CUDA_CHECK(cudaMemcpy(d_test_images, h_test_images, test_img_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_test_labels, h_test_labels, test_lbl_size, cudaMemcpyHostToDevice));

    int total_correct = 0;
    int test_batches = (NUM_TEST_IMAGES + BATCH_SIZE - 1) / BATCH_SIZE;

    // Processar imagens de teste em batches
    for (int batch = 0; batch < test_batches; batch++) {
        int batch_start = batch * BATCH_SIZE;
        int current_batch_size = fmin((float)BATCH_SIZE, (float)NUM_TEST_IMAGES - batch_start);

        // Para o último batch que pode ser menor
        if (current_batch_size < BATCH_SIZE) {
            // Precisamos ajustar as dimensões do grid
            hidden_grid.y = current_batch_size;
            output_grid.y = current_batch_size;
        }

        // Copiar batch de teste para memória temporária
        CUDA_CHECK(cudaMemcpy(d_batch_input,
                              &d_test_images[batch_start * INPUT_NODES],
                              current_batch_size * INPUT_NODES * sizeof(double),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_batch_target,
                              &d_test_labels[batch_start * OUTPUT_NODES],
                              current_batch_size * OUTPUT_NODES * sizeof(double),
                              cudaMemcpyDeviceToDevice));

        // Forward pass apenas (sem backpropagation no teste)
        forward_pass_batch<<<hidden_grid, THREADS_PER_BLOCK>>>(
            d_batch_input, d_weight1, d_bias1, d_batch_hidden,
            d_weight2, d_bias2, d_batch_output, current_batch_size
        );

        forward_pass_output<<<output_grid, THREADS_PER_BLOCK>>>(
            d_batch_hidden, d_weight2, d_bias2, d_batch_output, current_batch_size
        );

        // Calcular acurácia
        dim3 acc_grid((current_batch_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        calculate_batch_accuracy<<<acc_grid, THREADS_PER_BLOCK>>>(
            d_batch_output, d_batch_target, d_batch_predictions, current_batch_size
        );

        // Copiar resultados e contar acertos
        CUDA_CHECK(cudaMemcpy(h_batch_predictions, d_batch_predictions,
                              current_batch_size * sizeof(int), cudaMemcpyDeviceToHost));

        for (int i = 0; i < current_batch_size; i++) {
            total_correct += h_batch_predictions[i];
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    end_time = clock();
    printf("Tempo de teste: %.2f segundos\n",
           (double)(end_time - start_time) / CLOCKS_PER_SEC);
    printf("Testing Accuracy: %f\n", (double)total_correct / NUM_TEST_IMAGES);

    // Liberar toda a memória alocada na GPU
    // É importante liberar a memória para evitar vazamentos
    printf("\nLiberando memória GPU...\n");

    cudaFree(d_training_images);
    cudaFree(d_training_labels);
    cudaFree(d_test_images);
    cudaFree(d_test_labels);
    cudaFree(d_weight1);
    cudaFree(d_weight2);
    cudaFree(d_bias1);
    cudaFree(d_bias2);
    cudaFree(d_batch_input);
    cudaFree(d_batch_target);
    cudaFree(d_batch_hidden);
    cudaFree(d_batch_output);
    cudaFree(d_batch_delta_out);
    cudaFree(d_batch_delta_hidden);
    cudaFree(d_batch_predictions);
    cudaFree(d_grad_weight1);
    cudaFree(d_grad_weight2);
    cudaFree(d_grad_bias1);
    cudaFree(d_grad_bias2);

    printf("Execução concluída com sucesso!\n");

    return 0;
}
