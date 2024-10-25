#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>  // OpenMP ���̺귯��
#pragma warning(disable:4996)

#define EPSILON 1e-14  // �α� ��� �� �ʹ� ���� �� ����

// Pr(rlen'=s|rlen=x) ��� �Լ�
double pr_rlen_s_given_rlen_x(int s, int x, int s_max) {
    if (s == s_max) {
        return 1.0;
    }
    else {
        return 0.0;
    }
}

// Pr(rlen=x) ��� �Լ�
double pr_rlen_x(int* keyword_num_documents, int num_keywords, int x) {
    int count = 0;
    for (int i = 0; i < num_keywords; i++) {
        if (keyword_num_documents[i] == x) {
            count++;
        }
    }
    return (count > 0) ? (double)count / (double)num_keywords : 0.0;
}

// Pr(rlen'=s)�� �̸� ����ϴ� �Լ�
void precompute_pr_rlen_prime_s(double* pr_rlen_prime_s, int* keyword_num_documents, int num_keywords, int s_min, int s_max) {
    int total_iterations = s_max - s_min + 1;
    int completed_iterations = 0;

#pragma omp parallel for
    for (int s = s_min; s <= s_max; s++) {
        pr_rlen_prime_s[s - s_min] = 0.0;
        for (int x = s_min; x <= s_max; x++) {
            pr_rlen_prime_s[s - s_min] += pr_rlen_x(keyword_num_documents, num_keywords, x) * pr_rlen_s_given_rlen_x(s, x, s_max);
        }

        // ���� ��Ȳ ������Ʈ
#pragma omp atomic
        completed_iterations++;

        // ���� ��Ȳ�� ��� (0.1% ������)
        if (completed_iterations % (total_iterations / 1000) == 0) {
#pragma omp critical
            {
                printf("Precompute Progress: %.1f%% (s=%d)\n", (double)completed_iterations / total_iterations * 100, s);
            }
        }
    }
}

// Pr(key=w|rlen'=s) ��� �Լ�
double pr_key_w_given_rlen_s(int keyword_num_documents, int s, int num_keywords, double pr_rlen_prime_s_val, int s_max) {
    double pr_kw_w = 1.0 / num_keywords;  // Pr(kw=w)
    double pr_rlen_s_given_kw_w = pr_rlen_s_given_rlen_x(s, keyword_num_documents, s_max);  // Pr(rlen'=s|kw=w)

    if (pr_rlen_prime_s_val < EPSILON) {
        return 0.0;  // Pr(rlen'=s)�� �ʹ� ������ 0 ��ȯ
    }

    return (pr_rlen_s_given_kw_w * pr_kw_w) / pr_rlen_prime_s_val;
}

// H(s) ��� �Լ�
double H_s(int* keyword_num_documents, int num_keywords, int s, double* pr_rlen_prime_s_arr, int s_min, int s_max) {
    double H = 0.0;

    // �ε��� ���
    int index = s - s_min; // �ε��� ���
    if (index < 0 || index >= (s_max - s_min + 1)) {  // s_min �� s_max�� ������ Ȯ��
        printf("Invalid index for pr_rlen_prime_s_arr: %d\n", index);
        return H;  // �Ǵ� ���� ó��
    }

    double pr_rlen_prime_s_val = pr_rlen_prime_s_arr[index];  // �̹� ���� Pr(rlen'=s) ���

    for (int i = 0; i < num_keywords; i++) {
        double pr_key_w_rlen_s = pr_key_w_given_rlen_s(keyword_num_documents[i], s, num_keywords, pr_rlen_prime_s_val, s_max);
        if (pr_key_w_rlen_s > EPSILON) {
            H -= pr_key_w_rlen_s * log(pr_key_w_rlen_s);
        }
    }

    return H;
}

// RLO ��� �Լ� (����ȭ ���� �� ���� ��Ȳ ���)
double RLO_uniform_nopad(int* keyword_num_documents, int num_keywords, int s_min, int s_max) {
    double RLO = 0.0;
    int total_iterations = s_max - s_min + 1;
    int completed_iterations = 0;

    // Pr(rlen'=s) ���� �̸� ����Ͽ� ������ �迭
    double* pr_rlen_prime_s_arr = (double*)malloc((s_max - s_min + 1) * sizeof(double));
    if (!pr_rlen_prime_s_arr) {
        printf("Memory allocation failed for pr_rlen_prime_s_arr\n");
        exit(EXIT_FAILURE);
    }

    // ��� s�� ���� Pr(rlen'=s)�� �̸� ���
    precompute_pr_rlen_prime_s(pr_rlen_prime_s_arr, keyword_num_documents, num_keywords, s_min, s_max);

    // RLO ��� (����ȭ ����)
#pragma omp parallel for reduction(+:RLO)
    for (int s = s_min; s <= s_max; s++) {
        double pr_rlen_prime_s_val = pr_rlen_prime_s_arr[s - s_min];  // �̸� ���� �� ���
        double H_value = H_s(keyword_num_documents, num_keywords, s, pr_rlen_prime_s_arr, s_min, s_max);  // H(s) ���

        RLO += pr_rlen_prime_s_val * H_value;  // RLO ���

        // ���� ��Ȳ ������Ʈ
#pragma omp atomic
        completed_iterations++;

        // ���� ��Ȳ�� ��� (0.1% ������)
        if (completed_iterations % (total_iterations / 1000) == 0) {
#pragma omp critical
            {
                printf("RLO Calculation Progress: %.1f%% (s=%d)\n", (double)completed_iterations / total_iterations * 100, s);
            }
        }
    }

    free(pr_rlen_prime_s_arr);  // �޸� ����
    return RLO;
}

// CSV ���� �б� �Լ�
int read_csv(const char* filename, int* keyword_num_documents, int W) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Could not open file %s\n", filename);
        return -1;
    }

    char line[256];
    int i = 0;

    // ù ��° �� (���) �ǳʶٱ�
    fgets(line, sizeof(line), file);

    // ������ �б�
    while (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ",");
        token = strtok(NULL, ",");  // keyword_num_documents �� ����

        if (token != NULL) {
            if (i < W) {  // �迭�� ��踦 �˻�
                keyword_num_documents[i++] = atoi(token);
            }
            else {
                break;  // �ִ� ������ �����ϸ� ���� ����
            }
        }
    }

    fclose(file);
    return 0;
}

// �� RLO ���� ����ϰ� ���Ͽ� �����ϴ� �Լ�
void calculate_and_save_rlo(const char* input_file, const char* output_file, int W) {
    // keyword_num_documents �迭 ����
    int* keyword_num_documents = (int*)malloc(W * sizeof(int));
    if (!keyword_num_documents) {
        printf("Memory allocation failed for keyword_num_documents\n");
        exit(EXIT_FAILURE);
    }

    // CSV ���� �б�
    if (read_csv(input_file, keyword_num_documents, W) != 0) {
        free(keyword_num_documents);
        exit(EXIT_FAILURE);  // ������ ���� ���ϸ� ����
    }

    int num_keywords = W;  // Ű������ �� ����
    int s_min = keyword_num_documents[W - 1];  // �ּ� s ��
    int s_max = keyword_num_documents[0];      // �ִ� s ��

    // RLO �� ���
    double rlo_value = RLO_uniform_nopad(keyword_num_documents, num_keywords, s_min, s_max);

    // ��� ���
    printf("%s: %lf\n", output_file, rlo_value);

    // ��� ���Ͽ� ����
    FILE* output_fp = fopen(output_file, "w");
    if (output_fp != NULL) {
        fprintf(output_fp, "%s: %lf\n", output_file, rlo_value);
        fclose(output_fp);
        printf("%s value saved to %s\n", output_file, output_file);
    }
    else {
        printf("Error: Unable to open output file %s\n", output_file);
    }

    free(keyword_num_documents);
}

int main() {
    // RLO ��� �� ���� ����
    calculate_and_save_rlo("C:/dataset/picked_pairs(500).csv", "RLO_uniform_max_500.txt", 500);

    return 0;
}