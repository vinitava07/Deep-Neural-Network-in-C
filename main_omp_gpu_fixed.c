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
#define BATCH_SIZE 64

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

int correct_predictions;
int forward_prob_output;

// Versão simples das funções de ativação que funcionam em CPU e GPU
double sigmoid_simple(double x) {
    if (x > 10.0) return 1.0;
    if (x < -10.0) return 0.0;
    return 0.5 + x / (4.0 + (x > 0 ? x : -x));
}

double relu(double x) {
    return x > 0.0 ? x : 0.0;
}

// Função para carregar dados MNIST
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
    
    // Pular cabeçalho
    for (int i = 0; i < 8; i++) {
        fread(&t, 1, 1, training_images_file);
    }
    for (int i = 0; i < 8; i++) {
        fread(&t, 1, 1, training_labels_file);
    }
    for (int i = 0; i < 8; i++) {
        fread(&t, 1, 1, test_images_file);
    }
    for (int i = 0; i < 8; i++) {
        fread(&t, 1, 1, test_labels_file);
    }

    // Carregar dados de treino
    for (int i = 0; i < NUM_TRAINING_IMAGES; i++) {
        for (int j = 0; j < INPUT_NODES; j++) {
            unsigned char pixel;
            fread(&pixel, 1, 1, training_images_file);
            training_images[i][j] = (double)pixel / 255.0;
        }
    }

    for (int i = 0; i < NUM_TRAINING_IMAGES; i++) {
        unsigned char label;
        fread(&label, 1, 1, training_labels_file);
        for (int j = 0; j < OUTPUT_NODES; j++) {
            training_labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }

    // Carregar dados de teste
    for (int i = 0; i < NUM_TEST_IMAGES; i++) {
        for (int j = 0; j < INPUT_NODES; j++) {
            unsigned char pixel;
            fread(&pixel, 1, 1, test_images_file);
            test_images[i][j] = (double)pixel / 255.0;
        }
    }

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
// Versão simplificada e com debug do treinamento
void train_simple_debug(int epoch) {
    double lr = 0.1;
    if (epoch % 5 == 0 && epoch > 0) {
        lr *= 0.5;
    }
    
    forward_prob_output = 0;
    printf("[DEBUG] Iniciando época %d com learning rate %.3f\n", epoch, lr);
    
    // Processar uma imagem por vez para evitar problemas de memória
    for (int img = 0; img < NUM_TRAINING_IMAGES; img++) {
        // Debug a cada 100 imagens
        if (img % 100 == 0) {
            printf("[DEBUG] Processando imagem %d/%d\n", img, NUM_TRAINING_IMAGES);
        }
        
        // Arrays locais
        double hidden[HIDDEN_NODES];
        double output_layer[OUTPUT_NODES];
        
        // Forward pass - CPU primeiro para testar
        // Camada oculta
        for (int i = 0; i < HIDDEN_NODES; i++) {
            double sum = bias1[i];
            for (int j = 0; j < INPUT_NODES; j++) {
                sum += training_images[img][j] * weight1[j][i];
            }
            hidden[i] = relu(sum);
        }
        
        // Camada de saída
        for (int i = 0; i < OUTPUT_NODES; i++) {
            double sum = bias2[i];
            for (int j = 0; j < HIDDEN_NODES; j++) {
                sum += hidden[j] * weight2[j][i];
            }
            output_layer[i] = sigmoid_simple(sum);
        }
        
        // Verificar predição
        int prediction = 0;
        int correct_label = 0;
        double max_output = output_layer[0];
        
        for (int j = 1; j < OUTPUT_NODES; j++) {
            if (output_layer[j] > max_output) {
                max_output = output_layer[j];
                prediction = j;
            }
        }
        
        for (int j = 0; j < OUTPUT_NODES; j++) {
            if (training_labels[img][j] == 1.0) {
                correct_label = j;
                break;
            }
        }
        
        if (prediction == correct_label) {
            forward_prob_output++;
        }
        
        // Backpropagation
        double delta_out[OUTPUT_NODES];
        double delta_hidden[HIDDEN_NODES];
        
        // Calcular deltas de saída
        for (int i = 0; i < OUTPUT_NODES; i++) {
            double a = output_layer[i];
            delta_out[i] = (a - training_labels[img][i]) * a * (1.0 - a);
        }
        
        // Calcular deltas da camada oculta
        for (int i = 0; i < HIDDEN_NODES; i++) {
            double sum = 0.0;
            for (int j = 0; j < OUTPUT_NODES; j++) {
                sum += delta_out[j] * weight2[i][j];
            }
            delta_hidden[i] = sum * (hidden[i] > 0.0 ? 1.0 : 0.0);
        }
        
        // Atualizar pesos
        // Weight2 e bias2
        for (int i = 0; i < HIDDEN_NODES; i++) {
            for (int j = 0; j < OUTPUT_NODES; j++) {
                weight2[i][j] -= lr * delta_out[j] * hidden[i];
            }
        }
        
        for (int j = 0; j < OUTPUT_NODES; j++) {
            bias2[j] -= lr * delta_out[j];
        }
        
        // Weight1 e bias1
        for (int i = 0; i < INPUT_NODES; i++) {
            for (int j = 0; j < HIDDEN_NODES; j++) {
                weight1[i][j] -= lr * delta_hidden[j] * training_images[img][i];
            }
        }
        
        for (int j = 0; j < HIDDEN_NODES; j++) {
            bias1[j] -= lr * delta_hidden[j];
        }
    }
    
    printf("[DEBUG] Época %d concluída\n", epoch);
}

// Versão GPU com offload gradual
void train_gpu_gradual(int epoch) {
    double lr = 0.1;
    if (epoch % 5 == 0 && epoch > 0) {
        lr *= 0.5;
    }
    
    forward_prob_output = 0;
    printf("[DEBUG GPU] Iniciando época %d\n", epoch);
    
    // Testar primeiro com uma operação simples na GPU
    printf("[DEBUG GPU] Testando operação simples na GPU...\n");
    
    double test_array[100];
    for (int i = 0; i < 100; i++) test_array[i] = i;
    
    #pragma omp target teams distribute parallel for map(tofrom: test_array[0:100])
    for (int i = 0; i < 100; i++) {
        test_array[i] = test_array[i] * 2.0;
    }
    
    printf("[DEBUG GPU] Teste simples concluído. test_array[50] = %.1f\n", test_array[50]);
    
    // Agora processar as imagens
    for (int img = 0; img < NUM_TRAINING_IMAGES; img++) {
        if (img % 1000 == 0) {
            printf("[DEBUG GPU] Processando imagem %d\n", img);
        }
        
        // Arrays locais para esta imagem
        double hidden[HIDDEN_NODES] = {0};
        double output_layer[OUTPUT_NODES] = {0};
        double delta_out[OUTPUT_NODES] = {0};
        double delta_hidden[HIDDEN_NODES] = {0};
        
        // Copiar dados da imagem atual para array local
        double current_image[INPUT_NODES];
        double current_label[OUTPUT_NODES];
        
        for (int i = 0; i < INPUT_NODES; i++) {
            current_image[i] = training_images[img][i];
        }
        for (int i = 0; i < OUTPUT_NODES; i++) {
            current_label[i] = training_labels[img][i];
        }
        
        // Forward pass - camada oculta (GPU com dados locais)
        #pragma omp target teams distribute parallel for \
        map(to: current_image[0:INPUT_NODES], weight1[0:INPUT_NODES][0:HIDDEN_NODES], bias1[0:HIDDEN_NODES]) \
        map(from: hidden[0:HIDDEN_NODES])
        for (int i = 0; i < HIDDEN_NODES; i++) {
            double sum = bias1[i];
            for (int j = 0; j < INPUT_NODES; j++) {
                sum += current_image[j] * weight1[j][i];
            }
            hidden[i] = sum > 0.0 ? sum : 0.0; // ReLU inline
        }
        
        // Forward pass - camada de saída (CPU por enquanto)
        for (int i = 0; i < OUTPUT_NODES; i++) {
            double sum = bias2[i];
            for (int j = 0; j < HIDDEN_NODES; j++) {
                sum += hidden[j] * weight2[j][i];
            }
            output_layer[i] = sigmoid_simple(sum);
        }
        
        // Verificação e backprop (CPU)
        int prediction = 0;
        double max_output = output_layer[0];
        
        for (int j = 1; j < OUTPUT_NODES; j++) {
            if (output_layer[j] > max_output) {
                max_output = output_layer[j];
                prediction = j;
            }
        }
        
        int correct_label = 0;
        for (int j = 0; j < OUTPUT_NODES; j++) {
            if (current_label[j] == 1.0) {
                correct_label = j;
                break;
            }
        }
        
        if (prediction == correct_label) {
            forward_prob_output++;
        }
        
        // Backpropagation (CPU por simplicidade)
        for (int i = 0; i < OUTPUT_NODES; i++) {
            double a = output_layer[i];
            delta_out[i] = (a - current_label[i]) * a * (1.0 - a);
        }
        
        for (int i = 0; i < HIDDEN_NODES; i++) {
            double sum = 0.0;
            for (int j = 0; j < OUTPUT_NODES; j++) {
                sum += delta_out[j] * weight2[i][j];
            }
            delta_hidden[i] = sum * (hidden[i] > 0.0 ? 1.0 : 0.0);
        }
        
        // Atualização de pesos (CPU)
        for (int i = 0; i < HIDDEN_NODES; i++) {
            for (int j = 0; j < OUTPUT_NODES; j++) {
                weight2[i][j] -= lr * delta_out[j] * hidden[i];
            }
        }
        
        for (int j = 0; j < OUTPUT_NODES; j++) {
            bias2[j] -= lr * delta_out[j];
        }
        
        for (int i = 0; i < INPUT_NODES; i++) {
            for (int j = 0; j < HIDDEN_NODES; j++) {
                weight1[i][j] -= lr * delta_hidden[j] * current_image[i];
            }
        }
        
        for (int j = 0; j < HIDDEN_NODES; j++) {
            bias1[j] -= lr * delta_hidden[j];
        }
    }
    
    printf("[DEBUG GPU] Época concluída\n");
}
// Função de teste simples (CPU)
void test_simple() {
    correct_predictions = 0;
    
    printf("[DEBUG] Iniciando teste...\n");
    
    for (int img = 0; img < NUM_TEST_IMAGES; img++) {
        double hidden[HIDDEN_NODES];
        double output_layer[OUTPUT_NODES];
        
        // Forward pass
        for (int i = 0; i < HIDDEN_NODES; i++) {
            double sum = bias1[i];
            for (int j = 0; j < INPUT_NODES; j++) {
                sum += test_images[img][j] * weight1[j][i];
            }
            hidden[i] = relu(sum);
        }
        
        for (int i = 0; i < OUTPUT_NODES; i++) {
            double sum = bias2[i];
            for (int j = 0; j < HIDDEN_NODES; j++) {
                sum += hidden[j] * weight2[j][i];
            }
            output_layer[i] = sigmoid_simple(sum);
        }
        
        // Verificar predição
        int prediction = 0;
        double max_output = output_layer[0];
        
        for (int j = 1; j < OUTPUT_NODES; j++) {
            if (output_layer[j] > max_output) {
                max_output = output_layer[j];
                prediction = j;
            }
        }
        
        int correct_label = 0;
        for (int j = 0; j < OUTPUT_NODES; j++) {
            if (test_labels[img][j] == 1.0) {
                correct_label = j;
                break;
            }
        }
        
        if (prediction == correct_label) {
            correct_predictions++;
        }
    }
    
    printf("[DEBUG] Teste concluído\n");
}

int main(int argc, char *argv[]) {
    printf("=== OpenMP GPU - Versão Debug ===\n");
    printf("Rede Neural: %d -> %d -> %d\n", INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES);
    
    // Verificar argumentos
    int mode = 0; // 0 = CPU, 1 = GPU simples, 2 = GPU completa
    if (argc > 1) {
        mode = atoi(argv[1]);
    }
    
    printf("\nModo de execução: ");
    switch(mode) {
        case 0: printf("CPU (debug)\n"); break;
        case 1: printf("GPU Gradual\n"); break;
        case 2: printf("GPU Completa\n"); break;
        default: printf("CPU (padrão)\n"); mode = 0;
    }
    
    // Verificar suporte GPU
    int num_devices = omp_get_num_devices();
    printf("\nInformações do Sistema:\n");
    printf("- OpenMP version: %d\n", _OPENMP);
    printf("- Número de dispositivos GPU: %d\n", num_devices);
    
    if (num_devices == 0) {
        printf("\n⚠️  AVISO: Nenhuma GPU detectada!\n");
        if (mode > 0) {
            printf("Forçando modo CPU...\n");
            mode = 0;
        }
    } else {
        printf("- Dispositivo padrão: %d\n", omp_get_default_device());
        
        // Testar se conseguimos executar na GPU
        printf("\nTestando execução na GPU...\n");
        double test_val = 0.0;
        
        #pragma omp target map(tofrom: test_val)
        {
            test_val = 42.0;
        }
        
        if (test_val == 42.0) {
            printf("✅ GPU funcionando corretamente!\n");
        } else {
            printf("❌ Problema ao executar na GPU\n");
            if (mode > 0) {
                printf("Forçando modo CPU...\n");
                mode = 0;
            }
        }
    }
    
    // Inicializar pesos
    printf("\nInicializando pesos da rede...\n");
    srand(42);
    
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
    
    // Carregar dados
    printf("Carregando dataset MNIST...\n");
    load_mnist();
    printf("✅ Dados carregados: %d imagens de treino, %d de teste\n\n", 
           NUM_TRAINING_IMAGES, NUM_TEST_IMAGES);
    
    // Escolher função de treinamento baseada no modo
    void (*train_func)(int) = NULL;
    switch(mode) {
        case 0: train_func = train_simple_debug; break;
        case 1: train_func = train_gpu_gradual; break;
        default: train_func = train_simple_debug;
    }
    
    // Treinar
    printf("Iniciando treinamento...\n");
    printf("Learning rate inicial: 0.1 (reduz 50%% a cada 5 épocas)\n\n");
    
    clock_t start = clock();
    
    // Treinar apenas 1 época primeiro para teste
    printf("*** MODO DEBUG: Treinando apenas 1 época para teste ***\n");
    forward_prob_output = 0;
    train_func(0);
    
    clock_t end = clock();
    double train_time = (double)(end - start) / CLOCKS_PER_SEC;
    
    double accuracy = (double)forward_prob_output / NUM_TRAINING_IMAGES * 100.0;
    printf("\nResultado da primeira época:\n");
    printf("- Acurácia: %.2f%%\n", accuracy);
    printf("- Tempo: %.2f segundos\n", train_time);
    
    // Perguntar se deve continuar
    printf("\nContinuar com o treinamento completo? (s/n): ");
    char resp;
    scanf(" %c", &resp);
    
    if (resp == 's' || resp == 'S') {
        // Treinar as épocas restantes
        for (int epoch = 1; epoch < NUMBER_OF_EPOCHS; epoch++) {
            forward_prob_output = 0;
            
            clock_t epoch_start = clock();
            train_func(epoch);
            clock_t epoch_end = clock();
            
            double epoch_time = (double)(epoch_end - epoch_start) / CLOCKS_PER_SEC;
            accuracy = (double)forward_prob_output / NUM_TRAINING_IMAGES * 100.0;
            
            printf("Época %2d: Acurácia = %.2f%% | Tempo = %.2f seg\n", 
                   epoch + 1, accuracy, epoch_time);
        }
    }
    
    // Testar
    printf("\nIniciando teste...\n");
    start = clock();
    
    test_simple();
    
    end = clock();
    double test_time = (double)(end - start) / CLOCKS_PER_SEC;
    
    double test_accuracy = (double)correct_predictions / NUM_TEST_IMAGES * 100.0;
    printf("\n📊 Resultados Finais:\n");
    printf("- Acurácia de teste: %.2f%%\n", test_accuracy);
    printf("- Tempo de teste: %.2f segundos\n", test_time);
    
    return 0;
}