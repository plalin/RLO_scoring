#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>  // OpenMP ���̺귯��
#pragma warning(disable:4996)

#define W 2000  // Ű������ �� (w�� ����)
#define EPSILON 1e-14  // �α� ��� �� �ʹ� ���� �� ����

// Pr(rlen'=s|rlen=x) ��� �Լ�
// s >= x �̸� 1/(s_max - x + 1), s < x �̸� 0
double pr_rlen_s_given_rlen_x(int s, int x, int s_max) {
    if (s >= x) {
        return 1.0 / (s_max - x + 1);
    }
    else {
        return 0.0;  // s < x�� ��� 0
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

// Pr(key=w|rlen'=s) ��� �Լ�
// s < keyword_num_documents�̸� 0, s >= keyword_num_documents�̸� 1/(s_max - keyword_num_documents + 1)
double pr_key_w_given_rlen_s(int keyword_num_documents, int s, int s_max) {
    if (s < keyword_num_documents) {
        return 0.0;  // s�� keyword_num_documents���� ������ 0
    }
    else {
        return 1.0 / (s_max - keyword_num_documents + 1);  // s >= keyword_num_documents�� ���
    }
}

// H(s) ��� �Լ�
double H_s(int* keyword_num_documents, int num_keywords, int s, int s_max) {
    double H = 0.0;
    for (int i = 0; i < num_keywords; i++) {
        double pr_key_w_rlen_s = pr_key_w_given_rlen_s(keyword_num_documents[i], s, s_max);
        if (pr_key_w_rlen_s > EPSILON) {
            H -= pr_key_w_rlen_s * log(pr_key_w_rlen_s);
        }
    }
    return H;
}

// RLO ��� �Լ� (����ȭ)
double RLO_uniform_nopad(int* keyword_num_documents, int num_keywords, int s_min, int s_max) {
    double RLO = 0.0;
    int total_iterations = s_max - s_min + 1;
    int completed_iterations = 0;

    // OpenMP�� ����Ͽ� s�� ���� �ܺ� ���� ����ȭ
#pragma omp parallel for reduction(+:RLO)
    for (int s = s_min; s <= s_max; s++) {
        double pr_rlen_prime_s = 0.0;

        // x�� ���� ������ ����ȭ
        for (int x = s_min; x <= s_max; x++) {
            pr_rlen_prime_s += pr_rlen_x(keyword_num_documents, num_keywords, x) * pr_rlen_s_given_rlen_x(s, x, s_max);
        }

        // H(s) ��� �� RLO�� �߰�
        double H_value = H_s(keyword_num_documents, num_keywords, s, s_max);
        RLO += pr_rlen_prime_s * H_value;

        // ���� ��Ȳ ������Ʈ
#pragma omp atomic
        completed_iterations++;

        // ���� ��Ȳ�� ��� (1% ������)
        if (completed_iterations % (total_iterations / 100) == 0) {
#pragma omp critical
            {
                printf("Progress: %d/%d (%.2f%%)\n", completed_iterations, total_iterations,
                    (double)completed_iterations / total_iterations * 100);
            }
        }
    }
    return RLO;
}

// CSV ���� �б� �Լ�
int read_csv(const char* filename, int* keyword_num_documents) {
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
            keyword_num_documents[i++] = atoi(token);
        }

        if (i >= W) {  // 500�������� ����
            break;
        }
    }

    fclose(file);
    return 0;
}

int main() {
    // keyword_num_documents �迭 ����
    int keyword_num_documents[W] = { 0 };

    // CSV ���� �б�
    const char* filename = "C:/dataset/picked_pairs(2000).csv";
    if (read_csv(filename, keyword_num_documents) != 0) {
        return 1;  // ������ ���� ���ϸ� ����
    }

    int num_keywords = W;  // Ű������ �� ����
    int s_min = 100;         // �ּ� s ��
    int s_max = 107620;    // �ִ� s ��

    // RLO �� ���
    double rlo_value = RLO_uniform_nopad(keyword_num_documents, num_keywords, s_min, s_max);

    // ��� ���
    printf("RLO_uniform_nopad: %lf\n", rlo_value);

    // ��� ���Ͽ� ����
    FILE* output_file = fopen("RLO_uniform_RP_2000'.txt", "w");
    if (output_file != NULL) {
        fprintf(output_file, "RLO_uniform_RP: %lf\n", rlo_value);
        fclose(output_file);
        printf("RLO_uniform_nopad value saved to RLO_uniform_RP_2000'.txt\n");
    }
    else {
        printf("Error: Unable to open output file.\n");
    }

    return 0;
}
