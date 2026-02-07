 #include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>

using namespace std;

#define MAX_NAME 50
#define MAX_NUM  50

//////////////////////////////////////////////////////////
// Contact structure
//////////////////////////////////////////////////////////
struct Contact {
    char name[MAX_NAME];
    char number[MAX_NUM];
};

//////////////////////////////////////////////////////////
// Device helpers
//////////////////////////////////////////////////////////
__device__ char lower(char c) {
    if (c >= 'A' && c <= 'Z') return c + 32;
    return c;
}

// case-insensitive substring match
__device__ bool contains(const char* text, const char* pattern, int plen) {

    for (int i = 0; text[i] != '\0'; i++) {

        int j = 0;

        while (text[i + j] != '\0' &&
               pattern[j] != '\0' &&
               lower(text[i + j]) == lower(pattern[j])) {
            j++;
        }

        if (j == plen)
            return true;
    }
    return false;
}

//////////////////////////////////////////////////////////
// GPU kernel
//////////////////////////////////////////////////////////
__global__ void searchKernel(
    Contact* phonebook,
    int n,
    char* search_name,
    int name_len,
    int* results
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        if (contains(phonebook[idx].name, search_name, name_len))
            results[idx] = 1;
        else
            results[idx] = 0;
    }
}

//////////////////////////////////////////////////////////
// Main
//////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {

    if (argc != 2) {
        cout << "Usage: ./search_phonebook <search_name>\n";
        return 1;
    }

    string search_name = argv[1];

    //////////////////////////////////////////////////////
    // Open phonebook
    //////////////////////////////////////////////////////
    ifstream file("phonebook.txt");

    if (!file.is_open()) {
        cout << "Cannot open phonebook.txt\n";
        return 1;
    }

    vector<Contact> phonebook;
    string line;

    //////////////////////////////////////////////////////
    // SAFE parsing: "NAME","NUMBER"
    //////////////////////////////////////////////////////
    while (getline(file, line)) {

        Contact c;

        size_t q1 = line.find('"');
        size_t q2 = line.find('"', q1 + 1);
        size_t q3 = line.find('"', q2 + 1);
        size_t q4 = line.find('"', q3 + 1);

        if (q1==string::npos || q2==string::npos ||
            q3==string::npos || q4==string::npos)
            continue;

        string name   = line.substr(q1 + 1, q2 - q1 - 1);
        string number = line.substr(q3 + 1, q4 - q3 - 1);

        strcpy(c.name, name.c_str());
        strcpy(c.number, number.c_str());

        phonebook.push_back(c);
    }

    file.close();

    int n = phonebook.size();

    cout << "Total contacts loaded: " << n << endl;

    if (n == 0) return 0;

    //////////////////////////////////////////////////////
    // GPU memory
    //////////////////////////////////////////////////////
    Contact* d_phonebook;
    char* d_search;
    int* d_results;

    cudaMalloc(&d_phonebook, sizeof(Contact) * n);
    cudaMalloc(&d_search, search_name.size() + 1);
    cudaMalloc(&d_results, sizeof(int) * n);

    cudaMemcpy(d_phonebook, phonebook.data(),
               sizeof(Contact) * n,
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_search, search_name.c_str(),
               search_name.size() + 1,
               cudaMemcpyHostToDevice);

    //////////////////////////////////////////////////////
    // Launch kernel (auto sizing)
    //////////////////////////////////////////////////////
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    searchKernel<<<gridSize, blockSize>>>(
        d_phonebook,
        n,
        d_search,
        search_name.size(),
        d_results
    );

    cudaDeviceSynchronize();

    //////////////////////////////////////////////////////
    // Copy results back
    //////////////////////////////////////////////////////
    vector<int> results(n);

    cudaMemcpy(results.data(), d_results,
               sizeof(int) * n,
               cudaMemcpyDeviceToHost);

    //////////////////////////////////////////////////////
    // Write output file
    //////////////////////////////////////////////////////
    ofstream out("result.txt");

    int found = 0;

    for (int i = 0; i < n; i++) {
        if (results[i]) {
            out << "FOUND: "
                << phonebook[i].name
                << " -> "
                << phonebook[i].number
                << "\n";
            found++;
        }
    }

    if (found == 0)
        out << "No matches found\n";

    out.close();

    cout << "Matches: " << found << endl;
    cout << "Results saved to result.txt\n";

    //////////////////////////////////////////////////////
    // Cleanup
    //////////////////////////////////////////////////////
    cudaFree(d_phonebook);
    cudaFree(d_search);
    cudaFree(d_results);

    return 0;
}
