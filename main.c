#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>  // OpenMP 라이브러리
#pragma warning(disable:4996)

#define W 2000  // 키워드의 수 (w의 범위)
#define EPSILON 1e-14  // 로그 계산 시 너무 작은 값 방지

// Pr(rlen'=s|rlen=x) 계산 함수
// s >= x 이면 1/(s_max - x + 1), s < x 이면 0
double pr_rlen_s_given_rlen_x(int s, int x, int s_max) {
    if (s >= x) {
        return 1.0 / (s_max - x + 1);
    }
    else {
        return 0.0;  // s < x일 경우 0
    }
}

// Pr(rlen=x) 계산 함수
double pr_rlen_x(int* keyword_num_documents, int num_keywords, int x) {
    int count = 0;
    for (int i = 0; i < num_keywords; i++) {
        if (keyword_num_documents[i] == x) {
            count++;
        }
    }
    return (count > 0) ? (double)count / (double)num_keywords : 0.0;
}

// Pr(key=w|rlen'=s) 계산 함수
// s < keyword_num_documents이면 0, s >= keyword_num_documents이면 1/(s_max - keyword_num_documents + 1)
double pr_key_w_given_rlen_s(int keyword_num_documents, int s, int s_max) {
    if (s < keyword_num_documents) {
        return 0.0;  // s가 keyword_num_documents보다 작으면 0
    }
    else {
        return 1.0 / (s_max - keyword_num_documents + 1);  // s >= keyword_num_documents인 경우
    }
}

// H(s) 계산 함수
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

// RLO 계산 함수 (병렬화)
double RLO_uniform_nopad(int* keyword_num_documents, int num_keywords, int s_min, int s_max) {
    double RLO = 0.0;
    int total_iterations = s_max - s_min + 1;
    int completed_iterations = 0;

    // OpenMP를 사용하여 s에 대한 외부 루프 병렬화
#pragma omp parallel for reduction(+:RLO)
    for (int s = s_min; s <= s_max; s++) {
        double pr_rlen_prime_s = 0.0;

        // x에 대한 루프를 병렬화
        for (int x = s_min; x <= s_max; x++) {
            pr_rlen_prime_s += pr_rlen_x(keyword_num_documents, num_keywords, x) * pr_rlen_s_given_rlen_x(s, x, s_max);
        }

        // H(s) 계산 및 RLO에 추가
        double H_value = H_s(keyword_num_documents, num_keywords, s, s_max);
        RLO += pr_rlen_prime_s * H_value;

        // 진행 상황 업데이트
#pragma omp atomic
        completed_iterations++;

        // 진행 상황을 출력 (1% 단위로)
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

// CSV 파일 읽기 함수
int read_csv(const char* filename, int* keyword_num_documents) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Could not open file %s\n", filename);
        return -1;
    }

    char line[256];
    int i = 0;

    // 첫 번째 줄 (헤더) 건너뛰기
    fgets(line, sizeof(line), file);

    // 데이터 읽기
    while (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ",");
        token = strtok(NULL, ",");  // keyword_num_documents 값 추출

        if (token != NULL) {
            keyword_num_documents[i++] = atoi(token);
        }

        if (i >= W) {  // 500개까지만 읽음
            break;
        }
    }

    fclose(file);
    return 0;
}

int main() {
    // keyword_num_documents 배열 선언
    int keyword_num_documents[W] = { 0 };

    // CSV 파일 읽기
    const char* filename = "C:/dataset/picked_pairs(2000).csv";
    if (read_csv(filename, keyword_num_documents) != 0) {
        return 1;  // 파일을 읽지 못하면 종료
    }

    int num_keywords = W;  // 키워드의 총 개수
    int s_min = 100;         // 최소 s 값
    int s_max = 107620;    // 최대 s 값

    // RLO 값 계산
    double rlo_value = RLO_uniform_nopad(keyword_num_documents, num_keywords, s_min, s_max);

    // 결과 출력
    printf("RLO_uniform_nopad: %lf\n", rlo_value);

    // 결과 파일에 저장
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
