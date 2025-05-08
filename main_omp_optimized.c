#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <string.h>

#define INPUT_NODES 784  // 28*28 pixels
#define HIDDEN_NODES 256 // Number of hidden nodes
#define OUTPUT_NODES 10  // 10 digits (0-9)

#define NUM_TRAINING_IMAGES 8000
#define NUM_TEST_IMAGES 1000

#define NUMBER_OF_EPOCHS 10

// Alinhamento de memória para melhor desempenho de cache
__attribute__((aligned(64))) double training_images[NUM_TRAINING_IMAGES][INPUT_NODES];
__attribute__((aligned(64))) double training_labels[NUM_TRAINING_IMAGES][OUTPUT_NODES];
__attribute__((aligned(64))) double test_images[NUM_TEST_IMAGES][INPUT_NODES];
__attribute__((aligned(64))) double test_labels[NUM_TEST_IMAGES][OUTPUT_NODES];

// Pesos e bias alinhados para melhor desempenho
__attribute__((aligned(64))) double weight1[INPUT_NODES][HIDDEN_NODES];
__attribute__((aligned(64))) double weight2[HIDDEN_NODES][OUTPUT_NODES];
__attribute__((aligned(64))) double bias1[HIDDEN_NODES];
__attribute__((aligned(64))) double bias2[OUTPUT_NODES];

// Variáveis para contagem
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

// Utilizar funções inline para operações simples
static inline double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

static inline double reLU(double x)
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

// Versão otimizada da função de treino
void train(
    double input[INPUT_NODES],
    double target[OUTPUT_NODES],
    double weight1[INPUT_NODES][HIDDEN_NODES],
    double weight2[HIDDEN_NODES][OUTPUT_NODES],
    double bias1[HIDDEN_NODES],
    double bias2[OUTPUT_NODES],
    int correct_label,
    int epoch)
{
    // Alocação na pilha em vez do heap para melhor desempenho de cache
    double hidden[HIDDEN_NODES] __attribute__((aligned(64)));
    double output_layer[OUTPUT_NODES] __attribute__((aligned(64)));
    double delta_out[OUTPUT_NODES] __attribute__((aligned(64)));
    double delta_hidden[HIDDEN_NODES] __attribute__((aligned(64)));

    // Calcular taxa de aprendizado uma única vez
    double lr = 0.1;
    if (epoch % 5 == 0 && epoch > 0)
    {
        lr *= 0.5;
    }

    // Fase forward - paralelizar apenas blocos computacionalmente intensivos
    // Usando uma única região paralela para reduzir overhead de criação de threads
    #pragma omp parallel
    {
        // Primeira camada
        #pragma omp for schedule(guided, 16) nowait
        for (int i = 0; i < HIDDEN_NODES; i++)
        {
            double sum = bias1[i];
            // Loop interno desenrolado para melhor uso de pipelines
            for (int j = 0; j < INPUT_NODES; j += 4)
            {
                sum += input[j] * weight1[j][i];
                if (j+1 < INPUT_NODES) sum += input[j+1] * weight1[j+1][i];
                if (j+2 < INPUT_NODES) sum += input[j+2] * weight1[j+2][i];
                if (j+3 < INPUT_NODES) sum += input[j+3] * weight1[j+3][i];
            }
            hidden[i] = reLU(sum);
        }

        // Segunda camada
        #pragma omp for schedule(guided, 4) nowait
        for (int i = 0; i < OUTPUT_NODES; i++)
        {
            double sum = bias2[i];
            // Loop similar desenrolado, mas menos, já que HIDDEN_NODES não é tão grande
            for (int j = 0; j < HIDDEN_NODES; j += 2)
            {
                sum += hidden[j] * weight2[j][i];
                if (j+1 < HIDDEN_NODES) sum += hidden[j+1] * weight2[j+1][i];
            }
            output_layer[i] = sigmoid(sum);
        }

        // Sem barreira no final da região paralela para permitir sobreposição
    }

    // Parte sequencial (baixo custo computacional, não compensa paralelizar)
    int prediction = max_index(output_layer, OUTPUT_NODES);
    if (prediction == correct_label)
        forward_prob_output++;

    // Backpropagation deltas - paralelizando apenas onde necessário
    #pragma omp parallel
    {
        // Calcular deltas de saída
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < OUTPUT_NODES; i++)
        {
            double a = output_layer[i];
            delta_out[i] = (a - target[i]) * a * (1.0 - a);
        }

        // Calcular deltas da camada oculta - uso mais intensivo
        #pragma omp for schedule(guided, 16)
        for (int i = 0; i < HIDDEN_NODES; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < OUTPUT_NODES; j++)
            {
                sum += delta_out[j] * weight2[i][j];
            }
            delta_hidden[i] = sum * (hidden[i] > 0.0 ? 1.0 : 0.0);
        }
    }

    // Atualização de pesos - parte mais intensiva computacionalmente
    // Paralelizamos em blocos separados para otimizar o equilíbrio de carga

    // Atualização de pesos hidden -> output (parte menor, mais leve)
    #pragma omp parallel for schedule(guided, 8)
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            weight2[i][j] -= lr * delta_out[j] * hidden[i];
        }
    }

    // Atualização de bias de saída (muito pequeno, baixa granularidade)
    for (int j = 0; j < OUTPUT_NODES; j++)
    {
        bias2[j] -= lr * delta_out[j];
    }

    // Atualização de pesos input -> hidden (parte maior, mais pesada)
    #pragma omp parallel for schedule(guided, 32)
    for (int i = 0; i < INPUT_NODES; i++)
    {
        for (int j = 0; j < HIDDEN_NODES; j++)
        {
            weight1[i][j] -= lr * delta_hidden[j] * input[i];
        }
    }

    // Atualização de bias da camada oculta
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < HIDDEN_NODES; j++)
    {
        bias1[j] -= lr * delta_hidden[j];
    }
}

void test(double input[INPUT_NODES], double weight1[INPUT_NODES][HIDDEN_NODES], double weight2[HIDDEN_NODES][OUTPUT_NODES], double bias1[HIDDEN_NODES], double bias2[OUTPUT_NODES], int correct_label)
{
    double hidden[HIDDEN_NODES] __attribute__((aligned(64)));
    double output_layer[OUTPUT_NODES] __attribute__((aligned(64)));

    // Feedforward sem paralelização, pois é chamado em um loop já paralelizado
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

int main()
{
    clock_t start, end;
    double seconds;

    for(int tNumber = 2; tNumber <= 8; tNumber*=2){
        // Configure afinidade de processos/threads - importante para processadores de geração antiga
        #ifdef _OPENMP
        // Definir número fixo de threads para evitar overhead de criação/destruição
        omp_set_num_threads(tNumber);

        // Definir afinidade de threads para processador Q6600
        #ifdef __linux__
        putenv("OMP_PROC_BIND=close");
        putenv("OMP_PLACES=cores");
        #endif

        printf("Usando OpenMP com %d threads\n", tNumber);
        #endif

        // Seed para geração de números aleatórios
        srand(time(NULL));

        // Inicializar pesos e bias
        // Não paralelizamos essa parte, pois só é executada uma vez e o overhead supera o ganho
        for (int i = 0; i < INPUT_NODES; i++)
        {
            for (int j = 0; j < HIDDEN_NODES; j++)
            {
                weight1[i][j] = ((double)rand() / RAND_MAX * 0.1) - 0.05;
            }
        }

        for (int i = 0; i < HIDDEN_NODES; i++)
        {
            bias1[i] = ((double)rand() / RAND_MAX * 0.1) - 0.05;
            for (int j = 0; j < OUTPUT_NODES; j++)
            {
                weight2[i][j] = ((double)rand() / RAND_MAX * 0.1) - 0.05;
            }
        }

        for (int i = 0; i < OUTPUT_NODES; i++)
        {
            bias2[i] = ((double)rand() / RAND_MAX * 0.1) - 0.05;
        }

        // Carregar o dataset MNIST
        load_mnist();

        // TREINAMENTO
        printf("Iniciando treinamento...\n");
        start = clock();

        for (int epoch = 0; epoch < NUMBER_OF_EPOCHS; epoch++)
        {
            forward_prob_output = 0;

            // Importante: usar schedule(dynamic, batch_size) para melhor balanceamento
            // O tamanho do batch (20) foi escolhido para balancear granularidade e overhead
            #pragma omp parallel for schedule(dynamic, 20)
            for (int i = 0; i < NUM_TRAINING_IMAGES; i++)
            {
                int correct_label = max_index(training_labels[i], OUTPUT_NODES);
                train(training_images[i], training_labels[i], weight1, weight2, bias1, bias2, correct_label, epoch);
            }

            printf("Epoch %d: Training Accuracy: %lf\n", epoch, (double)forward_prob_output / NUM_TRAINING_IMAGES);
            printf("Example weight: %lf\n", weight1[0][0]);
        }

        end = clock();
        seconds = ((double)(end - start) / CLOCKS_PER_SEC) / tNumber;
        printf("Time to train: %f s\n", seconds);

        // Salvar modelo
        save_weights_biases("model_omp_optimized.bin");

        // TESTE
        printf("Iniciando teste...\n");
        start = clock();
        correct_predictions = 0;

        #pragma omp parallel for schedule(dynamic, 20)
        for (int i = 0; i < NUM_TEST_IMAGES; i++)
        {
            int correct_label = max_index(test_labels[i], OUTPUT_NODES);
            test(test_images[i], weight1, weight2, bias1, bias2, correct_label);
        }

        end = clock();
        seconds = ((double)(end - start) / CLOCKS_PER_SEC) / tNumber;
        printf("Time to test: %f s\n", seconds);
        printf("Testing Accuracy: %f\n", (double)correct_predictions / NUM_TEST_IMAGES);
    }
    return 0;
}
