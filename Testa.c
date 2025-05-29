/*
Paralelização GPU com OpenMP
Alunos:
    Arthur C.
    Leonardo B.
    Vinícius T.
    Wanderson P.

Para compilar com suporte GPU:
    gcc -fopenmp -foffload=nvptx-none -O3 -lm neural_network_gpu.c -o neural_network_gpu
    ou
    clang -fopenmp -fopenmp-targets=nvptx64 -O3 -lm neural_network_gpu.c -o neural_network_gpu
*/

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define INPUT_NODES 784  // 28*28 pixels
#define HIDDEN_NODES 256 // Number of hidden nodes
#define OUTPUT_NODES 10  // 10 digits (0-9)

#define NUM_TRAINING_IMAGES 8000
#define NUM_TEST_IMAGES 1000

#define NUMBER_OF_EPOCHS 10

// Arrays globais para dados, pesos e bias
double training_images[NUM_TRAINING_IMAGES][INPUT_NODES];
double training_labels[NUM_TRAINING_IMAGES][OUTPUT_NODES];
double test_images[NUM_TEST_IMAGES][INPUT_NODES];
double test_labels[NUM_TEST_IMAGES][OUTPUT_NODES];

// Pesos e bias - serão transferidos para GPU
double weight1[INPUT_NODES][HIDDEN_NODES];
double weight2[HIDDEN_NODES][OUTPUT_NODES];
double bias1[HIDDEN_NODES];
double bias2[OUTPUT_NODES];

// Variáveis para contagem
int correct_predictions;
int forward_prob_output;

void load_mnist() {
  // Open the training images file
  FILE *training_images_file = fopen("mnist_train_images.bin", "rb");
  if (training_images_file == NULL) {
    printf("Error opening training images file\n");
    exit(1);
  }

  // Open the training labels file
  FILE *training_labels_file = fopen("mnist_train_labels.bin", "rb");
  if (training_labels_file == NULL) {
    printf("Error opening training labels file\n");
    exit(1);
  }

  // Open the test images file
  FILE *test_images_file = fopen("mnist_test_images.bin", "rb");
  if (test_images_file == NULL) {
    printf("Error opening test images file\n");
    exit(1);
  }

  // Open the test labels file
  FILE *test_labels_file = fopen("mnist_test_labels.bin", "rb");
  if (test_labels_file == NULL) {
    printf("Error opening test labels file\n");
    exit(1);
  }

  unsigned char t;
  for (int i = 0; i < 8; i++) {
    fread(&t, sizeof(unsigned char), 1, training_images_file);
  }

  // Read the training images
  for (int i = 0; i < NUM_TRAINING_IMAGES; i++) {
    for (int j = 0; j < INPUT_NODES; j++) {
      unsigned char pixel;
      fread(&pixel, sizeof(unsigned char), 1, training_images_file);
      training_images[i][j] = (double)pixel / 255.0;
    }
  }
  for (int i = 0; i < 8; i++) {
    fread(&t, sizeof(unsigned char), 1, training_labels_file);
  }

  // Read the training labels
  for (int i = 0; i < NUM_TRAINING_IMAGES; i++) {
    unsigned char label;
    fread(&label, sizeof(unsigned char), 1, training_labels_file);
    for (int j = 0; j < OUTPUT_NODES; j++) {
      if (j == label) {
        training_labels[i][j] = 1;
      } else {
        training_labels[i][j] = 0;
      }
    }
  }

  // Read the test images
  for (int i = 0; i < 8; i++) {
    fread(&t, sizeof(unsigned char), 1, test_images_file);
  }

  for (int i = 0; i < NUM_TEST_IMAGES; i++) {
    for (int j = 0; j < INPUT_NODES; j++) {
      unsigned char pixel;
      fread(&pixel, sizeof(unsigned char), 1, test_images_file);
      test_images[i][j] = (double)pixel / 255.0;
    }
  }

  // Read the test labels
  for (int i = 0; i < 8; i++) {
    fread(&t, sizeof(unsigned char), 1, test_labels_file);
  }

  for (int i = 0; i < NUM_TEST_IMAGES; i++) {
    unsigned char label;
    fread(&label, sizeof(unsigned char), 1, test_labels_file);
    for (int j = 0; j < OUTPUT_NODES; j++) {
      if (j == label) {
        test_labels[i][j] = 1;
      } else {
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

// Funções inline que funcionam na GPU
#pragma omp declare target
static inline double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

static inline double reLU(double x) { return x > 0.0 ? x : 0.0; }
#pragma omp end declare target

int max_index(double arr[], int size) {
  int max_i = 0;
  for (int i = 1; i < size; i++) {
    if (arr[i] > arr[max_i]) {
      max_i = i;
    }
  }
  return max_i;
}

// Versão GPU da função de treino
void train_gpu_batch(double training_images[NUM_TRAINING_IMAGES][INPUT_NODES],
                     double training_labels[NUM_TRAINING_IMAGES][OUTPUT_NODES],
                     double weight1[INPUT_NODES][HIDDEN_NODES],
                     double weight2[HIDDEN_NODES][OUTPUT_NODES],
                     double bias1[HIDDEN_NODES], 
                     double bias2[OUTPUT_NODES],
                     int *predictions, int epoch, int batch_start, int batch_size) {
    
    double lr = 0.1;
    if (epoch % 5 == 0 && epoch > 0) {
        lr *= 0.5;
    }

    // Transferir dados para GPU e executar computação
    #pragma omp target teams distribute parallel for \
        map(to: training_images[batch_start:batch_size][:INPUT_NODES], \
                training_labels[batch_start:batch_size][:OUTPUT_NODES], \
                lr, epoch) \
        map(tofrom: weight1[:INPUT_NODES][:HIDDEN_NODES], \
                    weight2[:HIDDEN_NODES][:OUTPUT_NODES], \
                    bias1[:HIDDEN_NODES], bias2[:OUTPUT_NODES]) \
        map(from: predictions[0:batch_size]) \
        thread_limit(1024)
    for (int img = 0; img < batch_size; img++) {
        // Arrays locais para cada thread
        double hidden[HIDDEN_NODES];
        double output_layer[OUTPUT_NODES];
        double delta_out[OUTPUT_NODES];
        double delta_hidden[HIDDEN_NODES];
        
        int actual_img = batch_start + img;
        if (actual_img >= NUM_TRAINING_IMAGES) continue;
        
        // Forward pass - Primeira camada (Input -> Hidden)
        for (int i = 0; i < HIDDEN_NODES; i++) {
            double sum = bias1[i];
            for (int j = 0; j < INPUT_NODES; j++) {
                sum += training_images[actual_img][j] * weight1[j][i];
            }
            hidden[i] = reLU(sum);
        }
        
        // Forward pass - Segunda camada (Hidden -> Output)
        for (int i = 0; i < OUTPUT_NODES; i++) {
            double sum = bias2[i];
            for (int j = 0; j < HIDDEN_NODES; j++) {
                sum += hidden[j] * weight2[j][i];
            }
            output_layer[i] = sigmoid(sum);
        }
        
        // Encontrar predição e label correto
        int prediction = 0;
        int correct_label = 0;
        double max_output = output_layer[0];
        double max_label = training_labels[actual_img][0];
        
        for (int i = 1; i < OUTPUT_NODES; i++) {
            if (output_layer[i] > max_output) {
                max_output = output_layer[i];
                prediction = i;
            }
            if (training_labels[actual_img][i] > max_label) {
                max_label = training_labels[actual_img][i];
                correct_label = i;
            }
        }
        
        predictions[img] = (prediction == correct_label) ? 1 : 0;
        
        // Backward pass - Calcular deltas de saída
        for (int i = 0; i < OUTPUT_NODES; i++) {
            double a = output_layer[i];
            delta_out[i] = (a - training_labels[actual_img][i]) * a * (1.0 - a);
        }
        
        // Backward pass - Calcular deltas da camada oculta
        for (int i = 0; i < HIDDEN_NODES; i++) {
            double sum = 0.0;
            for (int j = 0; j < OUTPUT_NODES; j++) {
                sum += delta_out[j] * weight2[i][j];
            }
            delta_hidden[i] = sum * (hidden[i] > 0.0 ? 1.0 : 0.0);
        }
        
        // Atualização de pesos - Weight2 (Hidden -> Output)
        for (int i = 0; i < HIDDEN_NODES; i++) {
            for (int j = 0; j < OUTPUT_NODES; j++) {
                #pragma omp atomic update
                weight2[i][j] -= lr * delta_out[j] * hidden[i];
            }
        }
        
        // Atualização de bias2
        for (int j = 0; j < OUTPUT_NODES; j++) {
            #pragma omp atomic update
            bias2[j] -= lr * delta_out[j];
        }
        
        // Atualização de pesos - Weight1 (Input -> Hidden)
        for (int i = 0; i < INPUT_NODES; i++) {
            for (int j = 0; j < HIDDEN_NODES; j++) {
                #pragma omp atomic update
                weight1[i][j] -= lr * delta_hidden[j] * training_images[actual_img][i];
            }
        }
        
        // Atualização de bias1
        for (int j = 0; j < HIDDEN_NODES; j++) {
            #pragma omp atomic update
            bias1[j] -= lr * delta_hidden[j];
        }
    }
}

// Versão GPU da função de teste
void test_gpu_batch(double test_images[NUM_TEST_IMAGES][INPUT_NODES],
                    double test_labels[NUM_TEST_IMAGES][OUTPUT_NODES],
                    double weight1[INPUT_NODES][HIDDEN_NODES],
                    double weight2[HIDDEN_NODES][OUTPUT_NODES],
                    double bias1[HIDDEN_NODES], 
                    double bias2[OUTPUT_NODES],
                    int *predictions, int batch_start, int batch_size) {
    
    #pragma omp target teams distribute parallel for \
        map(to: test_images[batch_start:batch_size][:INPUT_NODES], \
                test_labels[batch_start:batch_size][:OUTPUT_NODES], \
                weight1[:INPUT_NODES][:HIDDEN_NODES], \
                weight2[:HIDDEN_NODES][:OUTPUT_NODES], \
                bias1[:HIDDEN_NODES], bias2[:OUTPUT_NODES]) \
        map(from: predictions[0:batch_size]) \
        thread_limit(1024)
    for (int img = 0; img < batch_size; img++) {
        double hidden[HIDDEN_NODES];
        double output_layer[OUTPUT_NODES];
        
        int actual_img = batch_start + img;
        if (actual_img >= NUM_TEST_IMAGES) continue;
        
        // Forward pass - Primeira camada
        for (int i = 0; i < HIDDEN_NODES; i++) {
            double sum = bias1[i];
            for (int j = 0; j < INPUT_NODES; j++) {
                sum += test_images[actual_img][j] * weight1[j][i];
            }
            hidden[i] = reLU(sum);
        }
        
        // Forward pass - Segunda camada
        for (int i = 0; i < OUTPUT_NODES; i++) {
            double sum = bias2[i];
            for (int j = 0; j < HIDDEN_NODES; j++) {
                sum += hidden[j] * weight2[j][i];
            }
            output_layer[i] = sigmoid(sum);
        }
        
        // Encontrar predição e label correto
        int prediction = 0;
        int correct_label = 0;
        double max_output = output_layer[0];
        double max_label = test_labels[actual_img][0];
        
        for (int i = 1; i < OUTPUT_NODES; i++) {
            if (output_layer[i] > max_output) {
                max_output = output_layer[i];
                prediction = i;
            }
            if (test_labels[actual_img][i] > max_label) {
                max_label = test_labels[actual_img][i];
                correct_label = i;
            }
        }
        
        predictions[img] = (prediction == correct_label) ? 1 : 0;
    }
}

void save_weights_biases(char *file_name) {
  FILE *file = fopen(file_name, "wb");
  if (file == NULL) {
    printf("Error opening file\n");
    exit(1);
  }
  fwrite(weight1, sizeof(double), HIDDEN_NODES * INPUT_NODES, file);
  fwrite(weight2, sizeof(double), HIDDEN_NODES * OUTPUT_NODES, file);
  fwrite(bias1, sizeof(double), HIDDEN_NODES, file);
  fwrite(bias2, sizeof(double), OUTPUT_NODES, file);
  fclose(file);
}

void load_weights_biases(char *file_name) {
  FILE *file = fopen(file_name, "rb");
  if (file == NULL) {
    printf("Error opening file\n");
    exit(1);
  }
  fread(weight1, sizeof(double), HIDDEN_NODES * INPUT_NODES, file);
  fread(weight2, sizeof(double), HIDDEN_NODES * OUTPUT_NODES, file);
  fread(bias1, sizeof(double), HIDDEN_NODES, file);
  fread(bias2, sizeof(double), OUTPUT_NODES, file);
  fclose(file);
}

int main(int argc, char *argv[]) {
    printf("Verificando suporte GPU...\n");
    
    // Verificar se GPU está disponível
    int num_devices = omp_get_num_devices();
    printf("Número de dispositivos GPU disponíveis: %d\n", num_devices);
    
    if (num_devices == 0) {
        printf("Nenhuma GPU encontrada. Executando em CPU com OpenMP.\n");
        // Fallback para CPU seria implementado aqui
    } else {
        printf("GPU encontrada. Executando computação na GPU.\n");
    }

    // Configurar número de threads para operações CPU
    int num_threads = (argc > 1) ? atoi(argv[1]) : 4;
    omp_set_num_threads(num_threads);
    printf("Usando %d threads CPU para operações auxiliares\n", num_threads);

    // Seed para geração de números aleatórios
    srand(time(NULL));

    // Inicializar pesos e bias
    for (int i = 0; i < INPUT_NODES; i++) {
        for (int j = 0; j < HIDDEN_NODES; j++) {
            weight1[i][j] = ((double)rand() / RAND_MAX * 0.1) - 0.05;
        }
    }

    for (int i = 0; i < HIDDEN_NODES; i++) {
        bias1[i] = ((double)rand() / RAND_MAX * 0.1) - 0.05;
        for (int j = 0; j < OUTPUT_NODES; j++) {
            weight2[i][j] = ((double)rand() / RAND_MAX * 0.1) - 0.05;
        }
    }

    for (int i = 0; i < OUTPUT_NODES; i++) {
        bias2[i] = ((double)rand() / RAND_MAX * 0.1) - 0.05;
    }

    // Carregar o dataset MNIST
    printf("Carregando dataset MNIST...\n");
    load_mnist();

    // TREINAMENTO
    printf("Iniciando treinamento na GPU...\n");
    
    const int BATCH_SIZE = 100; // Tamanho do batch para processamento GPU
    int *batch_predictions = (int*)malloc(BATCH_SIZE * sizeof(int));

    for (int epoch = 0; epoch < NUMBER_OF_EPOCHS; epoch++) {
        forward_prob_output = 0;
        
        // Processar em batches para otimizar transferências GPU
        for (int batch_start = 0; batch_start < NUM_TRAINING_IMAGES; batch_start += BATCH_SIZE) {
            int current_batch_size = (batch_start + BATCH_SIZE > NUM_TRAINING_IMAGES) ? 
                                   NUM_TRAINING_IMAGES - batch_start : BATCH_SIZE;
            
            // Executar batch na GPU
            train_gpu_batch(training_images, training_labels, weight1, weight2,
                           bias1, bias2, batch_predictions, epoch, batch_start, current_batch_size);
            
            // Contar predições corretas
            for (int i = 0; i < current_batch_size; i++) {
                forward_prob_output += batch_predictions[i];
            }
        }

        printf("Epoch %d: Training Accuracy: %lf\n", epoch,
               (double)forward_prob_output / NUM_TRAINING_IMAGES);
        printf("Example weight: %lf\n", weight1[0][0]);
    }

    // Salvar modelo
    save_weights_biases("model_gpu_optimized.bin");

    // TESTE
    printf("Iniciando teste na GPU...\n");
    correct_predictions = 0;

    // Processar teste em batches
    for (int batch_start = 0; batch_start < NUM_TEST_IMAGES; batch_start += BATCH_SIZE) {
        int current_batch_size = (batch_start + BATCH_SIZE > NUM_TEST_IMAGES) ? 
                               NUM_TEST_IMAGES - batch_start : BATCH_SIZE;
        
        test_gpu_batch(test_images, test_labels, weight1, weight2,
                      bias1, bias2, batch_predictions, batch_start, current_batch_size);
        
        // Contar predições corretas
        for (int i = 0; i < current_batch_size; i++) {
            correct_predictions += batch_predictions[i];
        }
    }

    printf("Testing Accuracy: %f\n", (double)correct_predictions / NUM_TEST_IMAGES);

    free(batch_predictions);
    return 0;
}