// test/test_att_tiered.cu
//  Correctness + benchmark driver for att_aat_tiered (C = A * A^T), self-contained.
#include "att_tiered.cuh"

#include <cuda_runtime.h>
#include <vector>
#include <tuple>
#include <random>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <string>

using std::vector;
using std::string;

// small CSR builder (host)
struct CSR { int m=0, n=0; vector<int> rp, ci; vector<float> val; int64_t nnz() const { return rp.empty()?0:(int64_t)rp.back(); } };

static CSR build_csr(int m, int n, vector<std::tuple<int,int,float>> trips) {
    // drop explicit zeros, dedup not required (we generate clean inputs)
    vector<int> cnt(m, 0);
    for (auto& t : trips) cnt[std::get<0>(t)]++;
    CSR A; A.m = m; A.n = n; A.rp.assign(m + 1, 0);
    for (int i = 0; i < m; ++i) A.rp[i + 1] = A.rp[i] + cnt[i];
    A.ci.resize(trips.size()); A.val.resize(trips.size());
    vector<int> cur(m); for (int i = 0; i < m; ++i) cur[i] = A.rp[i];
    for (auto& t : trips) { int i = std::get<0>(t); int p = cur[i]++; A.ci[p] = std::get<1>(t); A.val[p] = std::get<2>(t); }
    for (int i = 0; i < m; ++i) {           // sort within row by col
        int s = A.rp[i], e = A.rp[i + 1];
        vector<int> idx(e - s); for (int x = 0; x < e - s; ++x) idx[x] = x;
        std::sort(idx.begin(), idx.end(), [&](int a, int b){ return A.ci[s + a] < A.ci[s + b]; });
        vector<int> c2(e - s); vector<float> v2(e - s);
        for (int x = 0; x < e - s; ++x) { c2[x] = A.ci[s + idx[x]]; v2[x] = A.val[s + idx[x]]; }
        for (int x = 0; x < e - s; ++x) { A.ci[s + x] = c2[x]; A.val[s + x] = v2[x]; }
    }
    return A;
}

// CPU reference C = A*A^T (dense, upper then mirror)
static vector<vector<float>> ref_aat(const CSR& A) {
    int m = A.m;
    vector<vector<float>> C(m, vector<float>(m, 0.f));
    vector<vector<std::pair<int,float>>> col(A.n);
    for (int i = 0; i < m; ++i)
        for (int p = A.rp[i]; p < A.rp[i + 1]; ++p) col[A.ci[p]].push_back({i, A.val[p]});
    for (int k = 0; k < A.n; ++k) {
        auto& L = col[k];
        for (int a = 0; a < (int)L.size(); ++a)
            for (int b = a; b < (int)L.size(); ++b) {
                int i = L[a].first, j = L[b].first; if (i > j) std::swap(i, j);
                C[i][j] += L[a].second * L[b].second;
            }
    }
    for (int i = 0; i < m; ++i)
        for (int j = i + 1; j < m; ++j) C[j][i] = C[i][j];
    return C;
}

// upload + run + verify
static bool run_case(const string& name, const CSR& A, float tol, bool verbose, AttStats* stats_out=nullptr) {
    int m = A.m;
    int *d_rp, *d_ci; float *d_val;
    cudaMalloc(&d_rp,  (m + 1) * sizeof(int));
    cudaMalloc(&d_ci,  A.nnz() * sizeof(int));
    cudaMalloc(&d_val, A.nnz() * sizeof(float));
    cudaMemcpy(d_rp,  A.rp.data(),  (m + 1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_ci,  A.ci.data(),  A.nnz() * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, A.val.data(), A.nnz() * sizeof(float), cudaMemcpyHostToDevice);

    int *C_rp=nullptr, *C_ci=nullptr; float *C_val=nullptr; int64_t C_nnz=0;
    AttTiming T{}; AttStats S{};
    AttTieredConfig cfg;
    AttStatus st = att_aat_tiered(d_rp, d_ci, d_val, m, A.n, A.nnz(),
                                  &C_rp, &C_ci, &C_val, &C_nnz, cfg, &T, &S);
    if (stats_out) *stats_out = S;

    bool ok = true;
    if (st != AttStatus::OK) {
        printf("[FAIL] %-22s status=%d\n", name.c_str(), (int)st);
        cudaFree(d_rp); cudaFree(d_ci); cudaFree(d_val);
        if (C_rp) cudaFree(C_rp); if (C_ci) cudaFree(C_ci); if (C_val) cudaFree(C_val);
        return false;
    }

    // download
    vector<int> hrp(m + 1);
    cudaMemcpy(hrp.data(), C_rp, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    vector<int> hci(C_nnz); vector<float> hval(C_nnz);
    if (C_nnz > 0) {
        cudaMemcpy(hci.data(),  C_ci,  C_nnz * sizeof(int),   cudaMemcpyDeviceToHost);
        cudaMemcpy(hval.data(), C_val, C_nnz * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // structural checks
    auto fail = [&](const char* why){
        printf("[FAIL] %-22s %s\n", name.c_str(), why); ok = false;
    };
    if (hrp[0] != 0) fail("C_row_ptr[0]!=0");
    for (int i = 0; i < m && ok; ++i) {
        if (hrp[i + 1] < hrp[i]) { fail("row_ptr not monotone"); break; }
        for (int p = hrp[i]; p < hrp[i + 1] - 1; ++p) {
            if (hci[p] >= hci[p + 1]) { fail("cols not strictly increasing (dup or unsorted)"); break; }
        }
        if (!ok) break;
    }
    if (ok && hrp[m] != (int)C_nnz) fail("C_row_ptr[m]!=C_nnz");

    // numeric + symmetry vs reference
    if (ok) {
        vector<vector<float>> ref = ref_aat(A);
        // every GPU entry must match reference
        double maxabs = 0.0;
        for (int i = 0; i < m && ok; ++i)
            for (int p = hrp[i]; p < hrp[i + 1]; ++p) {
                int j = hci[p];
                float r = ref[i][j];
                float d = std::fabs((double)hval[p] - (double)r);
                double allow = tol + 1e-4 * std::fabs((double)r);
                if (d > allow) {
                    if (verbose || d > 1e-3)
                        printf("    mismatch C[%d][%d]=%g ref=%g d=%g\n", i, j, hval[p], r, d);
                    if (d > 1e-2) { fail("numeric mismatch vs reference"); break; }
                }
                if (d > maxabs) maxabs = d;
                // symmetry: find C[j][i]
                bool found = false;
                for (int q = hrp[j]; q < hrp[j + 1]; ++q) if (hci[q] == i) {
                    found = true; if (std::fabs((double)hval[q] - (double)hval[p]) > allow) { fail("C!=C^T value"); }
                    break;
                }
                if (!found) fail("C!=C^T missing mirror");
                if (!ok) break;
            }
        // every reference nonzero must be present in GPU
        if (ok) {
            for (int i = 0; i < m && ok; ++i)
                for (int j = 0; j < m; ++j) {
                    float r = ref[i][j];
                    bool struct_present = std::fabs((double)r) > 0.f;   // structural nonzero in reference
                    if (struct_present) {
                        bool in_gpu = false;
                        for (int p = hrp[i]; p < hrp[i + 1]; ++p) if (hci[p] == j) { in_gpu = true; break; }
                        if (!in_gpu) { fail("reference nonzero missing in GPU"); break; }
                    }
                }
        }
        if (verbose) printf("    maxabs_err=%.3g tol=%.3g\n", maxabs, tol);
    }

    if (ok) printf("[ OK ] %-22s nnz(A)=%-7lld nnz(C)=%-9lld L/M/H=%d/%d/%d heavyTasks=%lld Fup=%lld\n",
                   name.c_str(), (long long)A.nnz(), (long long)C_nnz,
                   S.n_light, S.n_medium, S.n_heavy, (long long)S.n_heavy_tasks, (long long)S.F_upper);

    cudaFree(d_rp); cudaFree(d_ci); cudaFree(d_val);
    if (C_rp) cudaFree(C_rp); if (C_ci) cudaFree(C_ci); if (C_val) cudaFree(C_val);
    return ok;
}

// matrix generators
static std::mt19937 rng(12345);

static CSR gen_random(int m, int n, double density, double vlo=-2.0, double vhi=2.0) {
    std::uniform_real_distribution<double> u01(0,1), uv(vlo,vhi);
    vector<std::tuple<int,int,float>> t;
    for (int i=0;i<m;i++) for (int k=0;k<n;k++) if (u01(rng) < density)
        t.push_back({i,k,(float)uv(rng)});
    return build_csr(m,n,t);
}
// power-law columns: a few hub columns have huge degree
static CSR gen_powerlaw(int m, int n, int hubs, double leaf_density) {
    std::uniform_real_distribution<double> u01(0,1), uv(-2,2);
    vector<std::tuple<int,int,float>> t;
    for (int i=0;i<m;i++) for (int k=0;k<n;k++) {
        double p = (k < hubs) ? 0.9 : leaf_density;     // hub columns near-dense
        if (u01(rng) < p) t.push_back({i,k,(float)uv(rng)});
    }
    return build_csr(m,n,t);
}
// a single super-heavy row: row 0 spans many high-degree columns
static CSR gen_heavy_row(int m, int n, int heavy_cols) {
    std::uniform_real_distribution<double> uv(-2,2);
    vector<std::tuple<int,int,float>> t;
    // every row connects to all hub columns -> high column degrees
    for (int i=0;i<m;i++) for (int k=0;k<heavy_cols;k++) t.push_back({i,k,(float)uv(rng)});
    // sprinkle a few extra so it is not fully uniform
    for (int i=0;i<m;i++) for (int k=heavy_cols;k<n;k++) if ((i*7+k*3)%11==0) t.push_back({i,k,(float)uv(rng)});
    return build_csr(m,n,t);
}
// cancellation: contributions +v/-v to same (i,j) cancel to ~0 (structural nonzero).
static CSR gen_cancellation() {
    // pick A so two contributions to C[0][1] are +1 and -1 -> cancel to 0 (structural).
    vector<std::tuple<int,int,float>> t = {{0,0,1.f},{1,0,1.f},{0,1,1.f},{1,1,-1.f}};
    return build_csr(2,2,t);
}
// multiple k merge into one C[i][j]
static CSR gen_multimerge() {
    // rows 0,1 share cols 0,1,2 -> C[0][1] = sum of 3 products
    vector<std::tuple<int,int,float>> t;
    for (int k=0;k<3;k++){ t.push_back({0,k,(float)(k+1)}); t.push_back({1,k,(float)(k+1)}); }
    return build_csr(2,3,t);
}

// ---- Matrix Market reader (self-contained) for benchmtx mode ----
// coordinate format -> CSR; mirrors off-diagonal entries when banner says "symmetric".
static CSR read_mtx(const char* path) {
    FILE* f = std::fopen(path, "r");
    if (!f) { std::fprintf(stderr, "[benchmtx] cannot open %s\n", path); std::exit(1); }
    char line[2048];
    if (!std::fgets(line, sizeof(line), f)) { std::fclose(f); std::exit(1); }
    bool symmetric = std::string(line).find("symmetric") != std::string::npos;
    while (std::fgets(line, sizeof(line), f)) { if (line[0] != '%') break; }
    int nr=0, nc=0;
    std::sscanf(line, "%d %d", &nr, &nc);
    vector<std::tuple<int,int,float>> trips;
    int r,c; double v;
    while (std::fgets(line, sizeof(line), f)) {
        if (line[0]=='%') continue;
        if (std::sscanf(line, "%d %d %lf", &r,&c,&v)==3) {
            trips.push_back({r-1,c-1,(float)v});
            if (symmetric && r!=c) trips.push_back({c-1,r-1,(float)v});
        } else if (std::sscanf(line, "%d %d", &r,&c)==2) {   // pattern (no value)
            trips.push_back({r-1,c-1,1.0f});
            if (symmetric && r!=c) trips.push_back({c-1,r-1,1.0f});
        }
    }
    std::fclose(f);
    return build_csr(nr, nc, trips);
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);   // unbuffered so hangs are localizable
    // benchmtx <file.mtx>: time att_aat_tiered on a REAL matrix (FP32). Produces the
    // measured-AAT timing used to cover the projected deliverable. Warmup + timed.
    if (argc > 2 && string(argv[1])=="benchmtx") {
        CSR A = read_mtx(argv[2]);
        int m=A.m;
        int *d_rp,*d_ci; float *d_val;
        cudaMalloc(&d_rp,(m+1)*sizeof(int)); cudaMalloc(&d_ci,A.nnz()*sizeof(int)); cudaMalloc(&d_val,A.nnz()*sizeof(float));
        cudaMemcpy(d_rp,A.rp.data(),(m+1)*sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(d_ci,A.ci.data(),A.nnz()*sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(d_val,A.val.data(),A.nnz()*sizeof(float),cudaMemcpyHostToDevice);
        int*cr=nullptr;int*cc=nullptr;float*cv=nullptr;int64_t cn=0;AttTiming T;AttStats S;
        att_aat_tiered(d_rp,d_ci,d_val,m,A.n,A.nnz(),&cr,&cc,&cv,&cn,{},&T,&S);   // warmup
        if(cr)cudaFree(cr);if(cc)cudaFree(cc);if(cv)cudaFree(cv);
        cr=cc=nullptr;cv=nullptr;cn=0;
        AttStatus st=att_aat_tiered(d_rp,d_ci,d_val,m,A.n,A.nnz(),&cr,&cc,&cv,&cn,{},&T,&S);
        std::string fn(argv[2]); size_t sl=fn.rfind('/'); std::string base=(sl==std::string::npos)?fn:fn.substr(sl+1);
        printf("AAT_MS %s total=%.4f nnzA=%lld nnzC=%lld L/M/H=%d/%d/%d status=%d ovf=%lld\n",
               base.c_str(), T.total, (long long)A.nnz(), (long long)cn,
               S.n_light,S.n_medium,S.n_heavy, (int)st, (long long)S.hash_overflow_fallbacks);
        if(cr)cudaFree(cr);if(cc)cudaFree(cc);if(cv)cudaFree(cv);
        cudaFree(d_rp);cudaFree(d_ci);cudaFree(d_val);
        return (st==AttStatus::OK)?0:2;
    }
    bool run_bench = (argc > 1 && string(argv[1])=="bench");
    int npass=0, nfail=0;
    auto CASE = [&](const string& nm, const CSR& A, float tol){
        if (run_case(nm, A, tol, false)) ++npass; else ++nfail; };

    // 1. empty matrix (m=0)
    { CSR A; A.m=0; A.n=0; A.rp={0};
      // run_case needs m>=1 path; handle empty inline
      int* d_rp; cudaMalloc(&d_rp,sizeof(int)); int z=0; cudaMemcpy(d_rp,&z,sizeof(int),cudaMemcpyHostToDevice);
      int*cr=nullptr;int*cc=nullptr;float*cv=nullptr;int64_t cn=0;AttTiming T;AttStats S;
      AttStatus st=att_aat_tiered(d_rp,nullptr,nullptr,0,0,0,&cr,&cc,&cv,&cn,{},&T,&S);
      printf("[ %s ] %-22s (empty m=0) cn=%lld\n", st==AttStatus::OK?"OK":"FAIL", "empty_matrix", (long long)cn);
      if(st==AttStatus::OK)++npass;else++nfail; cudaFree(d_rp); if(cr)cudaFree(cr);
    }
    // 2. matrix with empty rows
    { vector<std::tuple<int,int,float>> t={{0,0,1.f},{2,1,2.f}};
      CASE("empty_rows", build_csr(4,3,t), 1e-5); }
    // 3. identity
    { vector<std::tuple<int,int,float>> t; for(int i=0;i<6;i++)t.push_back({i,i,1.f});
      CASE("identity", build_csr(6,6,t), 1e-5); }
    // 4. diagonal (scalar matrix)
    { vector<std::tuple<int,int,float>> t; for(int i=0;i<5;i++)t.push_back({i,i,3.f});
      CASE("diagonal_scalar", build_csr(5,5,t), 1e-5); }
    // 5. rectangular A (m<n)
    CASE("rectangular_mxn", gen_random(3,7,0.5), 1e-4);
    // 6. random square small
    CASE("random_40x40", gen_random(40,40,0.15), 1e-4);
    // 7. multi-merge
    CASE("multi_merge", gen_multimerge(), 1e-5);
    // 8. cancellation (structural zero preserved, eps=0)
    CASE("cancellation", gen_cancellation(), 1e-5);
    // 9. single super-heavy row
    CASE("single_heavy_row", gen_heavy_row(60, 50, 45), 1e-3);
    // 10. power-law (skewed column degree)
    CASE("powerlaw", gen_powerlaw(150, 120, 6, 0.02), 1e-3);
    // 11. medium-path: a row with workload in (256, 8192]
    CASE("medium_path", gen_random(120, 120, 0.35), 1e-3);
    // 12. light-only tiny
    CASE("light_tiny", gen_random(20, 20, 0.1), 1e-4);

    printf("\n==== correctness: %d passed, %d failed ====\n\n", npass, nfail);

    // BENCHMARK
    if (run_bench) {
        printf("==== benchmark (CUDA-event compute-only phases, ms) ====\n");
        auto bench = [&](const string& nm, const CSR& A){
            int m=A.m; int *d_rp,*d_ci; float *d_val;
            cudaMalloc(&d_rp,(m+1)*sizeof(int)); cudaMalloc(&d_ci,A.nnz()*sizeof(int)); cudaMalloc(&d_val,A.nnz()*sizeof(float));
            cudaMemcpy(d_rp,A.rp.data(),(m+1)*sizeof(int),cudaMemcpyHostToDevice);
            cudaMemcpy(d_ci,A.ci.data(),A.nnz()*sizeof(int),cudaMemcpyHostToDevice);
            cudaMemcpy(d_val,A.val.data(),A.nnz()*sizeof(float),cudaMemcpyHostToDevice);
            // warmup
            int*cr=nullptr;int*cc=nullptr;float*cv=nullptr;int64_t cn=0;AttTiming T;AttStats S;
            att_aat_tiered(d_rp,d_ci,d_val,m,A.n,A.nnz(),&cr,&cc,&cv,&cn,{},&T,&S);
            if(cr)cudaFree(cr);if(cc)cudaFree(cc);if(cv)cudaFree(cv);
            // timed
            cr=cc=nullptr;cv=nullptr;cn=0;
            att_aat_tiered(d_rp,d_ci,d_val,m,A.n,A.nnz(),&cr,&cc,&cv,&cn,{},&T,&S);
            printf("--- %-16s m=%d n=%d nnz(A)=%lld nnz(C)=%lld ---\n",nm.c_str(),m,A.n,(long long)A.nnz(),(long long)cn);
            printf("    F_upper=%lld  light/medium/heavy=%d/%d/%d  heavy_tasks=%lld\n",
                   (long long)S.F_upper,S.n_light,S.n_medium,S.n_heavy,(long long)S.n_heavy_tasks);
            printf("    w: max=%lld avg=%.1f p99=%.0f | task_work: max=%lld avg=%.0f p99=%.0f | heavy_emit=%lld\n",
                   (long long)S.w_max,S.w_avg,S.w_p99,(long long)S.task_work_max,S.task_work_avg,S.task_work_p99,(long long)S.heavy_emit_pairs);
            printf("    csr2csc=%.3f workload=%.3f task_build=%.3f heavy_compute=%.3f heavy_reduce=%.3f\n",
                   T.csr2csc,T.workload,T.task_build,T.heavy_compute,T.heavy_reduce);
            printf("    sym_lightmed=%.3f assemble_ptr=%.3f num_lightmed=%.3f heavy_scatter=%.3f sym_csr=%.3f\n",
                   T.sym_lightmed,T.assemble_ptr,T.num_lightmed,T.heavy_scatter,T.sym_csr);
            printf("    TOTAL=%.3f ms\n", T.total);
            if(cr)cudaFree(cr);if(cc)cudaFree(cc);if(cv)cudaFree(cv);
            cudaFree(d_rp);cudaFree(d_ci);cudaFree(d_val);
        };
        bench("random_1k",   gen_random(1000,1000,0.01));
        bench("random_2k",   gen_random(2000,2000,0.005));
        bench("powerlaw_1k", gen_powerlaw(1000,1000,12,0.01));
        bench("heavy_120",   gen_heavy_row(120,120,110));

        // baseline comparison: tiered vs single-CTA-per-row, all-medium
        printf("\n==== baseline comparison (tiered vs single-CTA-per-row, all-medium) ====\n");
        auto run_cfg = [&](int* d_rp, int* d_ci, float* d_val, int m, int n, int64_t nnz,
                           const AttTieredConfig& cfg, double* total, int64_t* cnz, AttStats* S){
            int* cr=nullptr; int* cc=nullptr; float* cv=nullptr; int64_t c=0; AttTiming T; AttStats St;
            att_aat_tiered(d_rp,d_ci,d_val,m,n,nnz,&cr,&cc,&cv,&c,cfg,&T,&St);          // warmup
            if(cr)cudaFree(cr); if(cc)cudaFree(cc); if(cv)cudaFree(cv);
            cr=cc=nullptr; cv=nullptr; c=0;
            att_aat_tiered(d_rp,d_ci,d_val,m,n,nnz,&cr,&cc,&cv,&c,cfg,&T,&St);          // timed
            if(cr)cudaFree(cr); if(cc)cudaFree(cc); if(cv)cudaFree(cv);
            *total = T.total; *cnz = c; *S = St;
        };
        auto cmp = [&](const string& nm, const CSR& A){
            int m=A.m; int *d_rp,*d_ci; float* d_val;
            cudaMalloc(&d_rp,(m+1)*sizeof(int)); cudaMalloc(&d_ci,A.nnz()*sizeof(int)); cudaMalloc(&d_val,A.nnz()*sizeof(float));
            cudaMemcpy(d_rp,A.rp.data(),(m+1)*sizeof(int),cudaMemcpyHostToDevice);
            cudaMemcpy(d_ci,A.ci.data(),A.nnz()*sizeof(int),cudaMemcpyHostToDevice);
            cudaMemcpy(d_val,A.val.data(),A.nnz()*sizeof(float),cudaMemcpyHostToDevice);
            double tt=0, tb=0; int64_t ct=0, cb=0; AttStats St, Sb;
            run_cfg(d_rp,d_ci,d_val,m,A.n,A.nnz(),AttTieredConfig{},&tt,&ct,&St);
            AttTieredConfig base; base.light_work_threshold=0;
            base.heavy_work_threshold=(1LL<<62); base.medium_cap_override=16384;       // force all-medium
            run_cfg(d_rp,d_ci,d_val,m,A.n,A.nnz(),base,&tb,&cb,&Sb);
            printf("%-14s tiered=%.3fms (L/M/H=%d/%d/%d htasks=%lld) | baseline=%.3fms | speedup=%.2fx | nnz %lld vs %lld\n",
                   nm.c_str(), tt, St.n_light,St.n_medium,St.n_heavy,(long long)St.n_heavy_tasks,
                   tb, (tt>0?tb/tt:0.0), (long long)ct, (long long)cb);
            cudaFree(d_rp); cudaFree(d_ci); cudaFree(d_val);
        };
        cmp("heavy_120",   gen_heavy_row(120,120,110));
        cmp("powerlaw_1k", gen_powerlaw(1000,1000,12,0.01));
        cmp("random_1k",   gen_random(1000,1000,0.01));
    }
    return nfail ? 1 : 0;
}
