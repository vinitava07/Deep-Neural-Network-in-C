#include <math.h>
#include <mpi.h>
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

// Arrays globais sem atributo aligned
double training_images[NUM_TRAINING_IMAGES][INPUT_NODES];
double training_labels[NUM_TRAINING_IMAGES][OUTPUT_NODES];
double test_images[NUM_TEST_IMAGES][INPUT_NODES];
double test_labels[NUM_TEST_IMAGES][OUTPUT_NODES];

// Pesos e bias
double weight1[INPUT_NODES][HIDDEN_NODES];
double weight2[HIDDEN_NODES][OUTPUT_NODES];
double bias1[HIDDEN_NODES];
double bias2[OUTPUT_NODES];

// Buffers para redução MPI
double weight1_buffer[INPUT_NODES][HIDDEN_NODES];
double weight2_buffer[HIDDEN_NODES][OUTPUT_NODES];
double bias1_buffer[HIDDEN_NODES];
double bias2_buffer[OUTPUT_NODES];

// Variáveis para contagem
int correct_predictions = 0;
int forward_prob_output = 0;
int local_correct_predictions = 0;
int local_forward_prob_output = 0;

// Variáveis MPI
int mpi_rank, mpi_size;
int images_per_process;
int test_images_per_process;

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

// Utilizar funções inline para operações simples
static inline double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

static inline double reLU(double x) { return x > 0.0 ? x : 0.0; }

int max_index(double arr[], int size) {
  int max_i = 0;
  for (int i = 1; i < size; i++) {
    if (arr[i] > arr[max_i]) {
      max_i = i;
    }
  }
  return max_i;
}

// Versão otimizada da função de treino com MPI e OpenMP
void train(double input[INPUT_NODES], double target[OUTPUT_NODES],
           double weight1[INPUT_NODES][HIDDEN_NODES],
           double weight2[HIDDEN_NODES][OUTPUT_NODES],
           double bias1[HIDDEN_NODES], double bias2[OUTPUT_NODES],
           double weight1_local[INPUT_NODES][HIDDEN_NODES],
           double weight2_local[HIDDEN_NODES][OUTPUT_NODES],
           double bias1_local[HIDDEN_NODES], double bias2_local[OUTPUT_NODES],
           int correct_label, int epoch) {
  // Alocação na pilha em vez do heap para melhor desempenho de cache
  double hidden[HIDDEN_NODES];
  double output_layer[OUTPUT_NODES];
  double delta_out[OUTPUT_NODES];
  double delta_hidden[HIDDEN_NODES];

  // Calcular taxa de aprendizado uma única vez
  double lr = 0.1;
  if (epoch % 5 == 0 && epoch > 0) {
    lr *= 0.5;
  }

// Fase forward - paralelizar apenas blocos computacionalmente intensivos
// Usando uma única região paralela para reduzir overhead de criação de threads
#pragma omp parallel
  {
// Primeira camada
#pragma omp for schedule(guided, 16) nowait
    for (int i = 0; i < HIDDEN_NODES; i++) {
      double sum = bias1[i];
      // Loop interno desenrolado para melhor uso de pipelines
      for (int j = 0; j < INPUT_NODES; j += 4) {
        sum += input[j] * weight1[j][i];
        if (j + 1 < INPUT_NODES)
          sum += input[j + 1] * weight1[j + 1][i];
        if (j + 2 < INPUT_NODES)
          sum += input[j + 2] * weight1[j + 2][i];
        if (j + 3 < INPUT_NODES)
          sum += input[j + 3] * weight1[j + 3][i];
      }
      hidden[i] = reLU(sum);
    }

// Segunda camada
#pragma omp for schedule(guided, 4) nowait
    for (int i = 0; i < OUTPUT_NODES; i++) {
      double sum = bias2[i];
      // Loop similar desenrolado, mas menos, já que HIDDEN_NODES não é tão grande
      for (int j = 0; j < HIDDEN_NODES; j += 2) {
        sum += hidden[j] * weight2[j][i];
        if (j + 1 < HIDDEN_NODES)
          sum += hidden[j + 1] * weight2[j + 1][i];
      }
      output_layer[i] = sigmoid(sum);
    }

    // Sem barreira no final da região paralela para permitir sobreposição
  }

  // Parte sequencial (baixo custo computacional, não compensa paralelizar)
  int prediction = max_index(output_layer, OUTPUT_NODES);
  if (prediction == correct_label)
    local_forward_prob_output++;

// Backpropagation deltas - paralelizando apenas onde necessário
#pragma omp parallel
  {
// Calcular deltas de saída
#pragma omp for schedule(static) nowait
    for (int i = 0; i < OUTPUT_NODES; i++) {
      double a = output_layer[i];
      delta_out[i] = (a - target[i]) * a * (1.0 - a);
    }

// Calcular deltas da camada oculta - uso mais intensivo
#pragma omp for schedule(guided, 16)
    for (int i = 0; i < HIDDEN_NODES; i++) {
      double sum = 0.0;
      for (int j = 0; j < OUTPUT_NODES; j++) {
        sum += delta_out[j] * weight2[i][j];
      }
      delta_hidden[i] = sum * (hidden[i] > 0.0 ? 1.0 : 0.0);
    }
  }

// Atualização de pesos - parte mais intensiva computacionalmente
// Paralelizamos em blocos separados para otimizar o equilíbrio de carga

// Atualização de pesos hidden -> output (parte menor, mais leve)
#pragma omp parallel for schedule(guided, 8)
  for (int i = 0; i < HIDDEN_NODES; i++) {
    for (int j = 0; j < OUTPUT_NODES; j++) {
      weight2_local[i][j] -= lr * delta_out[j] * hidden[i];
    }
  }

  // Atualização de bias de saída (muito pequeno, baixa granularidade)
  for (int j = 0; j < OUTPUT_NODES; j++) {
    bias2_local[j] -= lr * delta_out[j];
  }

// Atualização de pesos input -> hidden (parte maior, mais pesada)
#pragma omp parallel for schedule(guided, 32)
  for (int i = 0; i < INPUT_NODES; i++) {
    for (int j = 0; j < HIDDEN_NODES; j++) {
      weight1_local[i][j] -= lr * delta_hidden[j] * input[i];
    }
  }

// Atualização de bias da camada oculta
#pragma omp parallel for schedule(static)
  for (int j = 0; j < HIDDEN_NODES; j++) {
    bias1_local[j] -= lr * delta_hidden[j];
  }
}

void test(double input[INPUT_NODES], double weight1[INPUT_NODES][HIDDEN_NODES],
          double weight2[HIDDEN_NODES][OUTPUT_NODES],
          double bias1[HIDDEN_NODES], double bias2[OUTPUT_NODES],
          int correct_label) {
  double hidden[HIDDEN_NODES];
  double output_layer[OUTPUT_NODES];

  // Feedforward sem paralelização, pois é chamado em um loop já paralelizado
  for (int i = 0; i < HIDDEN_NODES; i++) {
    double sum = bias1[i];
    for (int j = 0; j < INPUT_NODES; j++) {
      sum += input[j] * weight1[j][i];
    }
    hidden[i] = reLU(sum);
  }

  for (int i = 0; i < OUTPUT_NODES; i++) {
    double sum = bias2[i];
    for (int j = 0; j < HIDDEN_NODES; j++) {
      sum += hidden[j] * weight2[j][i];
    }
    output_layer[i] = sigmoid(sum);
  }

  int index = max_index(output_layer, OUTPUT_NODES);
  if (index == correct_label) {
#pragma omp atomic
    local_correct_predictions++;
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

// Inicializar os pesos com mesma semente em todos os processos para garantir consistência
void initialize_weights() {
  // Usamos o processo 0 para gerar os pesos iniciais e depois broadcast
  if (mpi_rank == 0) {
    // Seed para geração de números aleatórios
    srand(time(NULL));

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
  }

  // Broadcast dos pesos e biases para todos os processos
  MPI_Bcast(weight1, INPUT_NODES * HIDDEN_NODES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(weight2, HIDDEN_NODES * OUTPUT_NODES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(bias1, HIDDEN_NODES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(bias2, OUTPUT_NODES, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Inicializar cópias locais dos pesos para acumulação de gradientes
  for (int i = 0; i < INPUT_NODES; i++) {
    for (int j = 0; j < HIDDEN_NODES; j++) {
      weight1_buffer[i][j] = weight1[i][j];
    }
  }

  for (int i = 0; i < HIDDEN_NODES; i++) {
    bias1_buffer[i] = bias1[i];
    for (int j = 0; j < OUTPUT_NODES; j++) {
      weight2_buffer[i][j] = weight2[i][j];
    }
  }

  for (int i = 0; i < OUTPUT_NODES; i++) {
    bias2_buffer[i] = bias2[i];
  }
}

int main(int argc, char *argv[]) {
  // Inicializar MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // Definir número de threads OpenMP
  int tNumber = 2; // Valor padrão
  
  if (argc > 1) {
    tNumber = atoi(argv[1]);
    if (mpi_rank == 0) {
      printf("Usando número de threads OpenMP fornecido: %d\n", tNumber);
    }
  } else if (mpi_rank == 0) {
    printf("Nenhum número de threads fornecido, usando padrão: %d\n", tNumber);
  }
  
  omp_set_num_threads(tNumber);

  // Configurar afinidade de threads para melhor desempenho
#ifdef __linux__
  putenv("OMP_PROC_BIND=close");
  putenv("OMP_PLACES=cores");
#endif

  if (mpi_rank == 0) {
    printf("Executando com %d processos MPI e %d threads OpenMP por processo\n", 
           mpi_size, tNumber);
  }

  // Calcular número de imagens por processo
  images_per_process = NUM_TRAINING_IMAGES / mpi_size;
  test_images_per_process = NUM_TEST_IMAGES / mpi_size;
  
  if (mpi_rank == 0) {
    printf("Cada processo processará aproximadamente %d imagens de treino e %d imagens de teste\n", 
           images_per_process, test_images_per_process);
  }

  // Carregar dataset MNIST (todos os processos carregam o dataset completo)
  load_mnist();
  
  // Inicializar pesos e bias com mesma semente
  initialize_weights();
  
  // TREINAMENTO
  if (mpi_rank == 0) {
    printf("Iniciando treinamento...\n");
  }

  double start_time = MPI_Wtime();
  
  for (int epoch = 0; epoch < NUMBER_OF_EPOCHS; epoch++) {
    local_forward_prob_output = 0;
    
    // Cada processo processa seu subconjunto de imagens
    int start_idx = mpi_rank * images_per_process;
    int end_idx = (mpi_rank == mpi_size - 1) ? NUM_TRAINING_IMAGES : start_idx + images_per_process;
    
    // Zerar acumuladores locais no início de cada época
    memcpy(weight1_buffer, weight1, sizeof(double) * INPUT_NODES * HIDDEN_NODES);
    memcpy(weight2_buffer, weight2, sizeof(double) * HIDDEN_NODES * OUTPUT_NODES);
    memcpy(bias1_buffer, bias1, sizeof(double) * HIDDEN_NODES);
    memcpy(bias2_buffer, bias2, sizeof(double) * OUTPUT_NODES);

    // Cada processo treina com seu subconjunto de dados
#pragma omp parallel for schedule(dynamic, 20)
    for (int i = start_idx; i < end_idx; i++) {
      int correct_label = max_index(training_labels[i], OUTPUT_NODES);
      train(training_images[i], training_labels[i], weight1, weight2, bias1,
            bias2, weight1_buffer, weight2_buffer, bias1_buffer, bias2_buffer,
            correct_label, epoch);
    }
    
    // Reduzir (somar) os pesos de todos os processos
    MPI_Allreduce(MPI_IN_PLACE, weight1_buffer, INPUT_NODES * HIDDEN_NODES, 
                 MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, weight2_buffer, HIDDEN_NODES * OUTPUT_NODES, 
                 MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, bias1_buffer, HIDDEN_NODES, 
                 MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, bias2_buffer, OUTPUT_NODES, 
                 MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    // Reduzir forward_prob_output para estatísticas de acurácia
    MPI_Allreduce(&local_forward_prob_output, &forward_prob_output, 1, 
                 MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    // Calcular médias dos pesos
    for (int i = 0; i < INPUT_NODES; i++) {
      for (int j = 0; j < HIDDEN_NODES; j++) {
        weight1[i][j] = weight1_buffer[i][j] / mpi_size;
      }
    }

    for (int i = 0; i < HIDDEN_NODES; i++) {
      bias1[i] = bias1_buffer[i] / mpi_size;
      for (int j = 0; j < OUTPUT_NODES; j++) {
        weight2[i][j] = weight2_buffer[i][j] / mpi_size;
      }
    }

    for (int i = 0; i < OUTPUT_NODES; i++) {
      bias2[i] = bias2_buffer[i] / mpi_size;
    }
    
    // Apenas o processo 0 imprime estatísticas
    if (mpi_rank == 0) {
      printf("Epoch %d: Training Accuracy: %lf\n", epoch,
             (double)forward_prob_output / NUM_TRAINING_IMAGES);
      printf("Example weight: %lf\n", weight1[0][0]);
    }
  }
  
  double end_time = MPI_Wtime();
  
  if (mpi_rank == 0) {
    printf("Tempo total de treinamento: %lf segundos\n", end_time - start_time);
    
    // Salvar modelo final
    save_weights_biases("model_mpi_optimized.bin");
  }

  // TESTE
  if (mpi_rank == 0) {
    printf("Iniciando teste...\n");
  }
  
  local_correct_predictions = 0;
  start_time = MPI_Wtime();

  // Cada processo testa seu subconjunto de imagens
  int start_test_idx = mpi_rank * test_images_per_process;
  int end_test_idx = (mpi_rank == mpi_size - 1) ? NUM_TEST_IMAGES : start_test_idx + test_images_per_process;

#pragma omp parallel for schedule(dynamic, 20)
  for (int i = start_test_idx; i < end_test_idx; i++) {
    int correct_label = max_index(test_labels[i], OUTPUT_NODES);
    test(test_images[i], weight1, weight2, bias1, bias2, correct_label);
  }
  
  // Reduzir contagem de predições corretas
  MPI_Reduce(&local_correct_predictions, &correct_predictions, 1, 
             MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  
  end_time = MPI_Wtime();
  
  if (mpi_rank == 0) {
    printf("Tempo total de teste: %lf segundos\n", end_time - start_time);
    printf("Testing Accuracy: %f\n",
           (double)correct_predictions / NUM_TEST_IMAGES);
  }

  MPI_Finalize();
  return 0;
}
