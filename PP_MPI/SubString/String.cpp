#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

// ১. স্ট্রিং ক্লিনিং এবং লোয়ারকেস ফাংশন
string clean(string s) {
    if (s.empty()) return s;
    s.erase(remove(s.begin(), s.end(), '\"'), s.end());
    size_t first = s.find_first_not_of(" \t\r\n");
    if (string::npos == first) return "";
    size_t last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, (last - first + 1));
}

string low(string s) {
    transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

// ২. LCS লজিক (SORTED_M)
struct LCSResult { int len; string part; };
LCSResult getLCS(string line, string term) {
    string cl = clean(line), l = low(cl), t = low(clean(term));
    int n = l.length(), m = t.length();
    if (n == 0 || m == 0) return {0, ""};

    vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));
    int maxL = 0, endI = 0;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            if (l[i - 1] == t[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
                if (dp[i][j] > maxL) { maxL = dp[i][j]; endI = i; }
            }
        }
    }
    return {maxL, (maxL > 0 ? cl.substr(endI - maxL, maxL) : "")};
}

// ৩. MPI স্ট্রিং আদান-প্রদান ফাংশন
void send_str(string s, int target) {
    int len = s.size() + 1;
    MPI_Send(&len, 1, MPI_INT, target, 0, MPI_COMM_WORLD);
    MPI_Send(s.c_str(), len, MPI_CHAR, target, 0, MPI_COMM_WORLD);
}

string recv_str(int source) {
    int len;
    MPI_Recv(&len, 1, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    char* buf = new char[len];
    MPI_Recv(buf, len, MPI_CHAR, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    string res(buf); delete[] buf;
    return res;
}

struct FinalRes { int score; string line, part; };

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) cerr << "Usage: mpirun -np <p> ./p <file> <optional_threshold> <term...>" << endl;
        MPI_Finalize(); return 0;
    }

    int MIN_MATCH_LENGTH = 3; // ডিফল্ট থ্রেশহোল্ড
    int term_start_idx = 2;

    // আর্গুমেন্ট চেক: argv[2] কি সংখ্যা নাকি শব্দ? (stoi এরর এড়াতে)
    if (argc > 3 && isdigit(argv[2][0])) {
        MIN_MATCH_LENGTH = stoi(argv[2]);
        term_start_idx = 3;
    }

    string search_term = "";
    for (int i = term_start_idx; i < argc; i++)
        search_term += (string)argv[i] + (i == argc - 1 ? "" : " ");

    // সময় গণনা শুরু
    double start_time = MPI_Wtime();

    if (rank == 0) {
        // --- MASTER (Rank 0) ---
        ifstream f(argv[1]);
        if (!f.is_open()) { cerr << "File Error: " << argv[1] << endl; MPI_Abort(MPI_COMM_WORLD, 1); }

        vector<string> lines; string l;
        while (getline(f, l)) if (!l.empty()) lines.push_back(l);
        f.close();

        int total_lines = lines.size();
        int chunk = (total_lines + size - 1) / size;

        // ওয়ার্কারদের ডেটা ডিস্ট্রিবিউশন
        for (int i = 1; i < size; i++) {
            string s = "";
            int start = i * chunk, end = min(total_lines, (i + 1) * chunk);
            for (int j = start; j < end; j++) s += lines[j] + "\n";
            send_str(s, i);
        }

        vector<FinalRes> all_results;
        // মাস্টারের নিজের অংশের কাজ
        for (int i = 0; i < min(chunk, total_lines); i++) {
            LCSResult r = getLCS(lines[i], search_term);
            if (r.len >= MIN_MATCH_LENGTH) all_results.push_back({r.len, lines[i], r.part});
        }

        // ওয়ার্কারদের থেকে রেজাল্ট সংগ্রহ
        for (int i = 1; i < size; i++) {
            string data = recv_str(i);
            if (data == "") continue;
            stringstream ss(data); string row;
            while (getline(ss, row)) {
                size_t p1 = row.find('|'), p2 = row.find('|', p1 + 1);
                if (p1 != string::npos && p2 != string::npos)
                    all_results.push_back({stoi(row.substr(0, p1)), row.substr(p2 + 1), row.substr(p1 + 1, p2 - p1 - 1)});
            }
        }

        // সর্টিং
        sort(all_results.begin(), all_results.end(), [](FinalRes a, FinalRes b) { return a.score > b.score; });

        // আউটপুট সেভ
        ofstream fout("output.txt");
        for (auto& r : all_results)
            fout << "[Length: " << r.score << "] " << r.line << " (Match: " << r.part << ")" << endl;
        fout.close();

        double end_time = MPI_Wtime();
        cout << "Search Complete. Found " << all_results.size() << " matches." << endl;
        printf("Process Rank %d (Master) finished in: %f seconds\n", rank, end_time - start_time);
    }
    else {
        // --- WORKER (Rank 1 to N) ---
        string data = recv_str(0);
        stringstream ss(data); string line_str, response = "";
        while (getline(ss, line_str)) {
            LCSResult r = getLCS(line_str, search_term);
            if (r.len >= MIN_MATCH_LENGTH)
                response += to_string(r.len) + "|" + r.part + "|" + line_str + "\n";
        }
        send_str(response, 0);

        double end_time = MPI_Wtime();
        printf("Process Rank %d (Worker) finished in: %f seconds\n", rank, end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
//mpic++ max1.cpp -o p
//mpirun -np 2 ./p phonebook1.txt PANGAMA
