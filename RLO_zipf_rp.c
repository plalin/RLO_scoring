#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>  // OpenMP 라이브러리
#pragma warning(disable:4996)

#define W 1000  // 키워드의 수 (w의 범위)
#define EPSILON 1e-14  // 로그 계산 시 너무 작은 값 방지

// Pr(rlen'=s|rlen=x) 계산 함수
double pr_rlen_s_given_rlen_x(int s, int x, int s_max) {
    if (s >= x) {
        return 1.0 / (s_max - x + 1);
    }
    else {
        return 0.0;  // s < x일 경우 0
    }
}

// Pr(rlen=x) 계산 함수
double pr_rlen_x(int* keyword_num_documents, double* zipf_probs, int num_keywords, int x) {
    double sum_zipf_probs = 0.0;
    for (int i = 0; i < num_keywords; i++) {
        if (keyword_num_documents[i] == x) {
            sum_zipf_probs += zipf_probs[i];
        }
    }
    return sum_zipf_probs > 0.0 ? sum_zipf_probs : 0.0;
}

// Pr(key=w|rlen'=s) 계산 함수
double pr_key_w_given_rlen_s(int keyword_num_documents, int s, double pr_rlen_prime_s_val, double zipf_prob, int s_max) {
    double pr_kw_w = zipf_prob;  // 현재 키워드에 대응되는 zipf_prob 값
    double pr_rlen_s_given_kw_w = pr_rlen_s_given_rlen_x(s, keyword_num_documents, s_max);  // Pr(rlen'=s|kw=w)를 계산

    if (pr_rlen_prime_s_val < EPSILON) {
        return 0.0;  // Pr(rlen'=s)가 너무 작으면 0 반환
    }

    return (pr_rlen_s_given_kw_w * pr_kw_w) / pr_rlen_prime_s_val;
}

// Pr(rlen'=s)를 미리 계산하는 함수
void precompute_pr_rlen_prime_s(double* pr_rlen_prime_s, int* keyword_num_documents, double* zipf_probs, int num_keywords, int s_min, int s_max) {
    int total_iterations = s_max - s_min + 1;
    int completed_iterations = 0;

#pragma omp parallel for
    for (int s = s_min; s <= s_max; s++) {
        pr_rlen_prime_s[s - s_min] = 0.0;
        for (int x = s_min; x <= s_max; x++) {
            pr_rlen_prime_s[s - s_min] += pr_rlen_x(keyword_num_documents, zipf_probs, num_keywords, x) * pr_rlen_s_given_rlen_x(s, x, s_max);
        }

        // 진행 상황 업데이트
#pragma omp atomic
        completed_iterations++;

        // 진행 상황을 출력 (0.1% 단위로)
        if (completed_iterations % (total_iterations / 1000) == 0) {
#pragma omp critical
            {
                printf("Precompute Progress: %.1f%% (s=%d)\n", (double)completed_iterations / total_iterations * 100, s);
            }
        }
    }
}

// H(s) 계산 함수
double H_s(int* keyword_num_documents, double* zipf_probs, int num_keywords, int s, double* pr_rlen_prime_s_arr, int s_min, int s_max) {
    double H = 0.0;

    // 인덱스 계산
    int index = s - s_min; // 인덱스 계산
    if (index < 0 || index >= (s_max - s_min + 1)) {  // s_min 및 s_max로 범위를 확인
        printf("Invalid index for pr_rlen_prime_s_arr: %d\n", index);
        return H;  // 에러 처리
    }

    double pr_rlen_prime_s_val = pr_rlen_prime_s_arr[index];  // 이미 계산된 Pr(rlen'=s) 사용

    for (int i = 0; i < num_keywords; i++) {
        double pr_key_w_rlen_s = pr_key_w_given_rlen_s(keyword_num_documents[i], s, pr_rlen_prime_s_val, zipf_probs[i], s_max);
        if (pr_key_w_rlen_s > EPSILON) {
            H -= pr_key_w_rlen_s * log(pr_key_w_rlen_s);
        }
    }

    return H;
}

// RLO 계산 함수 (병렬화 적용 및 진행 상황 출력)
double RLO_uniform_nopad(int* keyword_num_documents, double* zipf_probs, int num_keywords, int s_min, int s_max) {
    double RLO = 0.0;
    int total_iterations = s_max - s_min + 1;
    int completed_iterations = 0;

    // Pr(rlen'=s) 값을 미리 계산하여 저장할 배열
    double* pr_rlen_prime_s_arr = (double*)malloc((s_max - s_min + 1) * sizeof(double));
    if (!pr_rlen_prime_s_arr) {
        printf("Memory allocation failed for pr_rlen_prime_s_arr\n");
        exit(EXIT_FAILURE);
    }

    // 모든 s에 대한 Pr(rlen'=s)를 미리 계산
    precompute_pr_rlen_prime_s(pr_rlen_prime_s_arr, keyword_num_documents, zipf_probs, num_keywords, s_min, s_max);

    // RLO 계산 (병렬화 적용)
#pragma omp parallel for reduction(+:RLO)
    for (int s = s_min; s <= s_max; s++) {
        double pr_rlen_prime_s_val = pr_rlen_prime_s_arr[s - s_min];  // 미리 계산된 값 사용
        double H_value = H_s(keyword_num_documents, zipf_probs, num_keywords, s, pr_rlen_prime_s_arr, s_min, s_max);  // H(s) 계산

        RLO += pr_rlen_prime_s_val * H_value;  // RLO 계산

        // 진행 상황 업데이트
#pragma omp atomic
        completed_iterations++;

        // 진행 상황을 출력 (0.1% 단위로)
        if (completed_iterations % (total_iterations / 1000) == 0) {
#pragma omp critical
            {
                printf("RLO Calculation Progress: %.1f%% (s=%d)\n", (double)completed_iterations / total_iterations * 100, s);
            }
        }
    }

    free(pr_rlen_prime_s_arr);  // 메모리 해제
    return RLO;
}

// CSV 파일 읽기 함수
int read_csv(const char* filename, int* keyword_num_documents, double* zipf_probs) {
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
        if (token != NULL && i < W) {
            keyword_num_documents[i] = atoi(token);

            // zipf_prob 값 추출
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
    // keyword_num_documents와 zipf_probs 배열 선언
    int keyword_num_documents[W] = { 0 };
    double zipf_probs[W] = { 0.0 };

    // CSV 파일 읽기
    const char* filename = "C:/dataset/picked_pairs_zipf(1000).csv";
    if (read_csv(filename, keyword_num_documents, zipf_probs) != 0) {
        return 1;  // 파일을 읽지 못하면 종료
    }

    int num_keywords = W;  // 키워드의 총 개수
    int s_min = keyword_num_documents[W - 1];  // 최소 s 값
    int s_max = keyword_num_documents[0];      // 최대 s 값

    // RLO 값 계산
    double rlo_value = RLO_uniform_nopad(keyword_num_documents, zipf_probs, num_keywords, s_min, s_max);

    // 결과 출력
    printf("RLO_zipf_rp: %lf\n", rlo_value);

    // 결과 파일에 저장
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