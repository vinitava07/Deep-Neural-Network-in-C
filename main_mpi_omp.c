#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>    // Para suporte MPI
#include <omp.h>    // Para suporte OpenMP

#define INPUT_NODES 784  // 28*28 pixels
#define HIDDEN_NODES 256 // Number of hidden nodes
#define OUTPUT_NODES 10  // 10 digits (0-9)

#define NUM_TRAINING_IMAGES 8000
#define NUM_TEST_IMAGES 1000

#define NUMBER_OF_EPOCHS 10

// Matrizes para armazenar os dados
double training_images[NUM_TRAINING_IMAGES][INPUT_NODES];
double training_labels[NUM_TRAINING_IMAGES][OUTPUT_NODES];
double test_images[NUM_TEST_IMAGES][INPUT_NODES];
double test_labels[NUM_TEST_IMAGES][OUTPUT_NODES];

// Pesos e bias
double weight1[INPUT_NODES][HIDDEN_NODES];
double weight2[HIDDEN_NODES][OUTPUT_NODES];
double bias1[HIDDEN_NODES];
double bias2[OUTPUT_NODES];

// Buffers para acumulação e redução de pesos e bias em MPI
double weight1_delta[INPUT_NODES][HIDDEN_NODES];
double weight2_delta[HIDDEN_NODES][OUTPUT_NODES];
double bias1_delta[HIDDEN_NODES];
double bias2_delta[OUTPUT_NODES];

// Contadores
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

// Função para acumular as alterações nos pesos e bias em vez de aplicá-las imediatamente
void train_batch(
    double input[INPUT_NODES],
    double target[OUTPUT_NODES],
    double weight1[INPUT_NODES][HIDDEN_NODES],
    double weight2[HIDDEN_NODES][OUTPUT_NODES],
    double bias1[HIDDEN_NODES],
    double bias2[OUTPUT_NODES],
    double weight1_delta[INPUT_NODES][HIDDEN_NODES],
    double weight2_delta[HIDDEN_NODES][OUTPUT_NODES],
    double bias1_delta[HIDDEN_NODES],
    double bias2_delta[OUTPUT_NODES],
    int correct_label,
    int epoch,
    double lr)
{
    double hidden[HIDDEN_NODES];
    double output_layer[OUTPUT_NODES];
    double delta_out[OUTPUT_NODES];
    double delta_hidden[HIDDEN_NODES];

    // 1) Feedforward - Paralelizando com OpenMP
    #pragma omp parallel
    {
        // Primeira camada: entrada -> camada oculta
        #pragma omp for
        for (int i = 0; i < HIDDEN_NODES; i++)
        {
            double sum = bias1[i];
            for (int j = 0; j < INPUT_NODES; j++)
                sum += input[j] * weight1[j][i];
            hidden[i] = reLU(sum);
        }

        // Segunda camada: camada oculta -> saída
        #pragma omp for
        for (int i = 0; i < OUTPUT_NODES; i++)
        {
            double sum = bias2[i];
            for (int j = 0; j < HIDDEN_NODES; j++)
                sum += hidden[j] * weight2[j][i];
            output_layer[i] = sigmoid(sum);
        }
    }

    // Controle de acurácia
    int prediction = max_index(output_layer, OUTPUT_NODES);
    #pragma omp atomic
    forward_prob_output += (prediction == correct_label);

    // 2) Backpropagation deltas
    #pragma omp parallel
    {
        // Cálculo dos deltas de saída
        #pragma omp for
        for (int i = 0; i < OUTPUT_NODES; i++)
        {
            double a = output_layer[i];
            delta_out[i] = (a - target[i]) * a * (1.0 - a);
        }

        // Cálculo dos deltas da camada oculta
        #pragma omp for
        for (int i = 0; i < HIDDEN_NODES; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < OUTPUT_NODES; j++)
                sum += delta_out[j] * weight2[i][j];
            delta_hidden[i] = sum * (hidden[i] > 0.0 ? 1.0 : 0.0);
        }
    }

    // 3) Acumular as alterações de pesos e bias para posterior redução MPI
    #pragma omp parallel
    {
        // (a) Hidden → Output weights & biases
        #pragma omp for collapse(2)
        for (int i = 0; i < HIDDEN_NODES; i++)
        {
            for (int j = 0; j < OUTPUT_NODES; j++)
            {
                weight2_delta[i][j] += lr * delta_out[j] * hidden[i];
            }
        }

        #pragma omp for
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            bias2_delta[j] += lr * delta_out[j];
        }

        // (b) Input → Hidden weights & biases
        #pragma omp for collapse(2)
        for (int i = 0; i < INPUT_NODES; i++)
        {
            for (int j = 0; j < HIDDEN_NODES; j++)
            {
                weight1_delta[i][j] += lr * delta_hidden[j] * input[i];
            }
        }

        #pragma omp for
        for (int j = 0; j < HIDDEN_NODES; j++)
        {
            bias1_delta[j] += lr * delta_hidden[j];
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
        #pragma omp atomic
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

int main(int argc, char *argv[])
{
    int rank, size;
    double start_time, end_time;
    int local_correct = 0;
    int global_correct = 0;
    
    // Inicializar MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Definir número de threads OpenMP
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    
    if (rank == 0) {
        printf("Executando com %d processos MPI e %d threads OpenMP por processo\n", 
               size, num_threads);
    }
    
    // Inicializar seed para números aleatórios
    srand(time(NULL) + rank);
    
    // Inicializar pesos e bias com valores aleatórios pequenos
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < INPUT_NODES; i++)
    {
        for (int j = 0; j < HIDDEN_NODES; j++)
        {
            weight1[i][j] = (double)rand() / RAND_MAX * 0.1 - 0.05;
        }
    }

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < HIDDEN_NODES; i++)
        {
            bias1[i] = (double)rand() / RAND_MAX * 0.1 - 0.05;
        }
        
        #pragma omp for collapse(2)
        for (int i = 0; i < HIDDEN_NODES; i++)
        {
            for (int j = 0; j < OUTPUT_NODES; j++)
            {
                weight2[i][j] = (double)rand() / RAND_MAX * 0.1 - 0.05;
            }
        }
        
        #pragma omp for
        for (int i = 0; i < OUTPUT_NODES; i++)
        {
            bias2[i] = (double)rand() / RAND_MAX * 0.1 - 0.05;
        }
    }
    
    // Todos os processos carregam os dados do MNIST
    load_mnist();
    
    // Garantir que todos os processos tenham os mesmos pesos iniciais
    MPI_Bcast(weight1, INPUT_NODES * HIDDEN_NODES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(weight2, HIDDEN_NODES * OUTPUT_NODES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(bias1, HIDDEN_NODES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(bias2, OUTPUT_NODES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Calcular quantas imagens cada processo irá processar
    int images_per_process = NUM_TRAINING_IMAGES / size;
    int start_idx = rank * images_per_process;
    int end_idx = (rank == size - 1) ? NUM_TRAINING_IMAGES : start_idx + images_per_process;
    
    // Iniciar o cronômetro
    start_time = MPI_Wtime();
    
    // Loop principal de treinamento
    for (int epoch = 0; epoch < NUMBER_OF_EPOCHS; epoch++)
    {
        forward_prob_output = 0;
        local_correct = 0;
        
        // Resetar os buffers de delta para esta época
        #pragma omp parallel
        {
            #pragma omp for collapse(2)
            for (int i = 0; i < INPUT_NODES; i++)
            {
                for (int j = 0; j < HIDDEN_NODES; j++)
                {
                    weight1_delta[i][j] = 0.0;
                }
            }
            
            #pragma omp for collapse(2)
            for (int i = 0; i < HIDDEN_NODES; i++)
            {
                for (int j = 0; j < OUTPUT_NODES; j++)
                {
                    weight2_delta[i][j] = 0.0;
                }
            }
            
            #pragma omp for
            for (int i = 0; i < HIDDEN_NODES; i++)
            {
                bias1_delta[i] = 0.0;
            }
            
            #pragma omp for
            for (int i = 0; i < OUTPUT_NODES; i++)
            {
                bias2_delta[i] = 0.0;
            }
        }
        
        // Calcular taxa de aprendizado para esta época
        double lr = 0.1;
        if (epoch % 5 == 0 && epoch > 0)
        {
            lr *= 0.5;
        }
        
        // Cada processo processa seu conjunto de imagens
        #pragma omp parallel for schedule(dynamic)
        for (int i = start_idx; i < end_idx; i++)
        {
            int correct_label = max_index(training_labels[i], OUTPUT_NODES);
            train_batch(training_images[i], training_labels[i], 
                        weight1, weight2, bias1, bias2,
                        weight1_delta, weight2_delta, bias1_delta, bias2_delta,
                        correct_label, epoch, lr);
        }
        
        // Comunicação MPI: Reduzir os deltas de todos os processos
        MPI_Allreduce(MPI_IN_PLACE, weight1_delta, INPUT_NODES * HIDDEN_NODES, 
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, weight2_delta, HIDDEN_NODES * OUTPUT_NODES, 
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, bias1_delta, HIDDEN_NODES, 
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, bias2_delta, OUTPUT_NODES, 
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        // Comunicação MPI: Reduzir e distribuir o contador de acurácia
        local_correct = forward_prob_output;
        MPI_Allreduce(&local_correct, &global_correct, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        
        // Atualizar os pesos com os deltas acumulados
        #pragma omp parallel
        {
            #pragma omp for collapse(2)
            for (int i = 0; i < INPUT_NODES; i++)
            {
                for (int j = 0; j < HIDDEN_NODES; j++)
                {
                    weight1[i][j] -= weight1_delta[i][j] / size;
                }
            }
            
            #pragma omp for collapse(2)
            for (int i = 0; i < HIDDEN_NODES; i++)
            {
                for (int j = 0; j < OUTPUT_NODES; j++)
                {
                    weight2[i][j] -= weight2_delta[i][j] / size;
                }
            }
            
            #pragma omp for
            for (int i = 0; i < HIDDEN_NODES; i++)
            {
                bias1[i] -= bias1_delta[i] / size;
            }
            
            #pragma omp for
            for (int i = 0; i < OUTPUT_NODES; i++)
            {
                bias2[i] -= bias2_delta[i] / size;
            }
        }
        
        // Apenas o processo 0 imprime informações da época
        if (rank == 0)
        {
            printf("Epoch %d : Training Accuracy: %lf\n", 
                   epoch, (double)global_correct / NUM_TRAINING_IMAGES);
            printf("Example weight: %lf\n", weight1[0][0]);
        }
    }
    
    // Parar o cronômetro
    end_time = MPI_Wtime();
    
    // Processo 0 salva os pesos e bias e executa o teste
    if (rank == 0)
    {
        printf("Time to train: %f seconds\n", end_time - start_time);
        save_weights_biases("model_mpi_omp.bin");
        
        // Testar a rede
        start_time = MPI_Wtime();
        correct_predictions = 0;
        
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < NUM_TEST_IMAGES; i++)
        {
            int correct_label = max_index(test_labels[i], OUTPUT_NODES);
            test(test_images[i], weight1, weight2, bias1, bias2, correct_label);
        }
        
        end_time = MPI_Wtime();
        printf("Time to test: %f seconds\n", end_time - start_time);
        printf("Testing Accuracy: %f\n", (double)correct_predictions / NUM_TEST_IMAGES);
    }
    
    MPI_Finalize();
    return 0;
}