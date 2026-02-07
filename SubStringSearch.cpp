%%writefile search_phonebook.cu
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>

using namespace std;

// ১. প্রি-প্রসেসিং ফাংশন
string preprocess(string s) {
    if (s.empty()) return "";
    s.erase(remove(s.begin(), s.end(), '\"'), s.end());
    transform(s.begin(), s.end(), s.begin(), ::tolower);
    size_t first = s.find_first_not_of(" \t\r\n");
    if (string::npos == first) return "";
    size_t last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, (last - first + 1));
}

// ২. CUDA Kernel (LCS Logic)
__global__ void lcs_kernel(char* d_data, int* d_offsets, int* d_lengths, int num_lines,
                           char* d_search_term, int search_len, int* d_scores, int* d_match_pos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_lines) return;

    char* line = d_data + d_offsets[idx];
    int line_len = d_lengths[idx];

    int max_len = 0;
    int end_idx = 0;

    // GPU-র লোকাল মেমোরিতে DP ক্যালকুলেশন (Max search term length: 512)
    int curr[513] = {0};
    int prev[513] = {0};

    for (int i = 0; i < line_len; i++) {
        for (int j = 0; j < search_len; j++) {
            if (line[i] == d_search_term[j]) {
                curr[j + 1] = prev[j] + 1;
                if (curr[j + 1] > max_len) {
                    max_len = curr[j + 1];
                    end_idx = i;
                }
            } else {
                curr[j + 1] = 0;
            }
        }
        for (int j = 0; j <= search_len; j++) {
            prev[j] = curr[j];
        }
    }
    d_scores[idx] = max_len;
    d_match_pos[idx] = (max_len > 0) ? (end_idx - max_len + 1) : 0;
}

struct FinalRes { int score; string line, part; };

int main(int argc, char** argv) {
    // ==========================================
    // Fixed Configuration
    // ==========================================
    string file_path = "/content/sample_data/phonebook1.txt";
    int threshold = 3;
    // ==========================================

    if (argc < 3) {
        cerr << "Usage: ./search_phonebook <search_term> <threads_per_block>" << endl;
        return 1;
    }

    string search_word = argv[1];
    int threadsPerBlock = stoi(argv[2]); // টার্মিনাল থেকে আসা ১০০ বা অন্য সংখ্যা

    string search_term = preprocess(search_word);

    ifstream f(file_path);
    if (!f.is_open()) {
        cerr << "Error: File not found at " << file_path << endl;
        return 1;
    }

    vector<string> original_lines, clean_lines;
    string raw_line, all_data_flat = "";
    vector<int> offsets, lengths;

    while (getline(f, raw_line)) {
        if (raw_line.empty()) continue;
        original_lines.push_back(raw_line);
        string cleaned = preprocess(raw_line);
        clean_lines.push_back(cleaned);

        offsets.push_back(all_data_flat.size());
        lengths.push_back(cleaned.size());
        all_data_flat += cleaned;
    }
    f.close();

    int num_lines = clean_lines.size();
    int search_len = search_term.size();

    char *d_data, *d_search_term;
    int *d_offsets, *d_lengths, *d_scores, *d_match_pos;

    cudaMalloc(&d_data, all_data_flat.size());
    cudaMalloc(&d_offsets, num_lines * sizeof(int));
    cudaMalloc(&d_lengths, num_lines * sizeof(int));
    cudaMalloc(&d_search_term, search_len);
    cudaMalloc(&d_scores, num_lines * sizeof(int));
    cudaMalloc(&d_match_pos, num_lines * sizeof(int));

    cudaMemcpy(d_data, all_data_flat.c_str(), all_data_flat.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets.data(), num_lines * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, lengths.data(), num_lines * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_search_term, search_term.c_str(), search_len, cudaMemcpyHostToDevice);

    // টার্মিনাল থেকে পাওয়া থ্রেড সংখ্যা ব্যবহার করে গ্রিড সাইজ নির্ধারণ
    int blocksPerGrid = (num_lines + threadsPerBlock - 1) / threadsPerBlock;

    lcs_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_offsets, d_lengths, num_lines, d_search_term, search_len, d_scores, d_match_pos);
    cudaDeviceSynchronize();

    vector<int> h_scores(num_lines), h_pos(num_lines);
    cudaMemcpy(h_scores.data(), d_scores, num_lines * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pos.data(), d_match_pos, num_lines * sizeof(int), cudaMemcpyDeviceToHost);

    vector<FinalRes> results;
    for (int i = 0; i < num_lines; i++) {
        if (h_scores[i] >= threshold) {
            results.push_back({h_scores[i], original_lines[i], clean_lines[i].substr(h_pos[i], h_scores[i])});
        }
    }

    sort(results.begin(), results.end(), [](FinalRes a, FinalRes b) { return a.score > b.score; });

    ofstream fout("output.txt");
    for (auto& r : results) {
        fout << "[Score: " << r.score << "] " << r.line << " (Match: " << r.part << ")" << endl;
    }
    fout.close();

    cout << "GPU Search Complete. Threads used per block: " << threadsPerBlock << endl;
    cout << "Total Matches Found: " << results.size() << " (Threshold: " << threshold << ")" << endl;

    cudaFree(d_data); cudaFree(d_offsets); cudaFree(d_lengths);
    cudaFree(d_search_term); cudaFree(d_scores); cudaFree(d_match_pos);

    return 0;
}
