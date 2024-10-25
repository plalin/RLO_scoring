#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>  // OpenMP ���̺귯��
#pragma warning(disable:4996)

#define W 1000  // Ű������ �� (w�� ����)
#define EPSILON 1e-14  // �α� ��� �� �ʹ� ���� �� ����

// Pr(rlen'=s|rlen=x) ��� �Լ�
double pr_rlen_s_given_rlen_x(int s, int x, int s_max) {
    if (s >= x) {
        return 1.0 / (s_max - x + 1);
    }
    else {
        return 0.0;  // s < x�� ��� 0
    }
}

// Pr(rlen=x) ��� �Լ�
double pr_rlen_x(int* keyword_num_documents, double* zipf_probs, int num_keywords, int x) {
    double sum_zipf_probs = 0.0;
    for (int i = 0; i < num_keywords; i++) {
        if (keyword_num_documents[i] == x) {
            sum_zipf_probs += zipf_probs[i];
        }
    }
    return sum_zipf_probs > 0.0 ? sum_zipf_probs : 0.0;
}

// Pr(key=w|rlen'=s) ��� �Լ�
double pr_key_w_given_rlen_s(int keyword_num_documents, int s, double pr_rlen_prime_s_val, double zipf_prob, int s_max) {
    double pr_kw_w = zipf_prob;  // ���� Ű���忡 �����Ǵ� zipf_prob ��
    double pr_rlen_s_given_kw_w = pr_rlen_s_given_rlen_x(s, keyword_num_documents, s_max);  // Pr(rlen'=s|kw=w)�� ���

    if (pr_rlen_prime_s_val < EPSILON) {
        return 0.0;  // Pr(rlen'=s)�� �ʹ� ������ 0 ��ȯ
    }

    return (pr_rlen_s_given_kw_w * pr_kw_w) / pr_rlen_prime_s_val;
}

// Pr(rlen'=s)�� �̸� ����ϴ� �Լ�
void precompute_pr_rlen_prime_s(double* pr_rlen_prime_s, int* keyword_num_documents, double* zipf_probs, int num_keywords, int s_min, int s_max) {
    int total_iterations = s_max - s_min + 1;
    int completed_iterations = 0;

#pragma omp parallel for
    for (int s = s_min; s <= s_max; s++) {
        pr_rlen_prime_s[s - s_min] = 0.0;
        for (int x = s_min; x <= s_max; x++) {
            pr_rlen_prime_s[s - s_min] += pr_rlen_x(keyword_num_documents, zipf_probs, num_keywords, x) * pr_rlen_s_given_rlen_x(s, x, s_max);
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

// H(s) ��� �Լ�
double H_s(int* keyword_num_documents, double* zipf_probs, int num_keywords, int s, double* pr_rlen_prime_s_arr, int s_min, int s_max) {
    double H = 0.0;

    // �ε��� ���
    int index = s - s_min; // �ε��� ���
    if (index < 0 || index >= (s_max - s_min + 1)) {  // s_min �� s_max�� ������ Ȯ��
        printf("Invalid index for pr_rlen_prime_s_arr: %d\n", index);
        return H;  // ���� ó��
    }

    double pr_rlen_prime_s_val = pr_rlen_prime_s_arr[index];  // �̹� ���� Pr(rlen'=s) ���

    for (int i = 0; i < num_keywords; i++) {
        double pr_key_w_rlen_s = pr_key_w_given_rlen_s(keyword_num_documents[i], s, pr_rlen_prime_s_val, zipf_probs[i], s_max);
        if (pr_key_w_rlen_s > EPSILON) {
            H -= pr_key_w_rlen_s * log(pr_key_w_rlen_s);
        }
    }

    return H;
}

// RLO ��� �Լ� (����ȭ ���� �� ���� ��Ȳ ���)
double RLO_uniform_nopad(int* keyword_num_documents, double* zipf_probs, int num_keywords, int s_min, int s_max) {
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
    precompute_pr_rlen_prime_s(pr_rlen_prime_s_arr, keyword_num_documents, zipf_probs, num_keywords, s_min, s_max);

    // RLO ��� (����ȭ ����)
#pragma omp parallel for reduction(+:RLO)
    for (int s = s_min; s <= s_max; s++) {
        double pr_rlen_prime_s_val = pr_rlen_prime_s_arr[s - s_min];  // �̸� ���� �� ���
        double H_value = H_s(keyword_num_documents, zipf_probs, num_keywords, s, pr_rlen_prime_s_arr, s_min, s_max);  // H(s) ���

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
int read_csv(const char* filename, int* keyword_num_documents, double* zipf_probs) {
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
        if (token != NULL && i < W) {
            keyword_num_documents[i] = atoi(token);

            // zipf_prob �� ����
            token = strtok(NULL, ",");
            if (token != NULL) {
                zipf_probs[i] = atof(token);
                i++;
            }
        }
    }

    fclose(file);
    return 0;
}

int main() {
    // keyword_num_documents�� zipf_probs �迭 ����
    int keyword_num_documents[W] = { 0 };
    double zipf_probs[W] = { 0.0 };

    // CSV ���� �б�
    const char* filename = "C:/dataset/picked_pairs_zipf(1000).csv";
    if (read_csv(filename, keyword_num_documents, zipf_probs) != 0) {
        return 1;  // ������ ���� ���ϸ� ����
    }

    int num_keywords = W;  // Ű������ �� ����
    int s_min = keyword_num_documents[W - 1];  // �ּ� s ��
    int s_max = keyword_num_documents[0];      // �ִ� s ��

    // RLO �� ���
    double rlo_value = RLO_uniform_nopad(keyword_num_documents, zipf_probs, num_keywords, s_min, s_max);

    // ��� ���
    printf("RLO_zipf_rp: %lf\n", rlo_value);

    // ��� ���Ͽ� ����
    FILE* output_file = fopen("RLO_zipf_rp_1000.txt", "w");
    if (output_file != NULL) {
        fprintf(output_file, "RLO_zipf_rp_1000: %lf\n", rlo_value);
        fclose(output_file);
        printf("RLO_zipf_rp value saved to RLO_zipf_rp_1000.txt\n");
    }
    else {
        printf("Error: Unable to open output file.\n");
    }

    return 0;
}