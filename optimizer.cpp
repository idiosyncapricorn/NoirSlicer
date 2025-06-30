// optimizer.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <map>
#include <string>
#include <random>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

// -----------------
// Fitness proxy: simple smoothness‐inversion
double fitness_proxy(const std::vector<double> &data, const std::map<std::string,double> &cfg) {
    // Here, we simulate a “smoothness” measure:
    // Use speed parameter to scale roughness
    double speed = cfg.at("speed");
    double factor = speed / 40.0;
    double sum_sq = 0.0;
    for (size_t i = 1; i < data.size(); ++i) {
        double d = factor * (data[i] - data[i-1]);
        sum_sq += d*d;
    }
    // Higher is better => invert
    return 1.0 / (sum_sq + 1e-6);
}

// Evaluate a single config over N trials
double evaluate_candidate(const std::vector<double> &data,
                          const std::map<std::string,double> &cfg,
                          int trials = 5)
{
    if (trials < 1)
        throw std::runtime_error("evaluate_candidate: trials must be ≥1");

    std::vector<double> scores;
    scores.reserve(trials);

    // Simple RNG for simulation jitter
    std::mt19937_64 rng(12345);
    std::normal_distribution<double> noise(0.0, 0.02);

    for (int t = 0; t < trials; ++t) {
        double base = fitness_proxy(data, cfg);
        double perturbed = base + noise(rng);
        scores.push_back(perturbed);
    }
    // Compute mean and stddev
    double sum = 0.0, sum2 = 0.0;
    for (double s : scores) {
        sum += s;
        sum2 += s*s;
    }
    double mean = sum / trials;
    double var  = (sum2 / trials) - (mean*mean);
    double stddev = var > 0.0 ? std::sqrt(var) : 0.0;

    // Return penalized score
    return mean - 0.5 * stddev;
}

// Optimize over a list of configs
py::list optimize(py::list cfg_list,
                  py::array_t<double, py::array::c_style | py::array::forcecast> arr,
                  int trials = 5,
                  int top_k = 3)
{
    // Extract data
    auto buf = arr.request();
    if (buf.ndim != 1)
        throw std::runtime_error("optimize: data must be 1-D array");
    size_t n = buf.shape[0];
    if (n < 2)
        throw std::runtime_error("optimize: need ≥2 data points");
    double *ptr = static_cast<double*>(buf.ptr);
    std::vector<double> data(ptr, ptr + n);

    // Convert cfg_list to C++ vector
    std::vector<std::map<std::string,double>> configs;
    for (auto item : cfg_list) {
        auto dict = item.cast<py::dict>();
        std::map<std::string,double> m;
        for (auto &kv : dict) {
            std::string key = kv.first.cast<std::string>();
            double val      = kv.second.cast<double>();
            m[key] = val;
        }
        configs.push_back(m);
    }

    // Evaluate
    std::vector<std::pair<double,int>> scored;  // (score, idx)
    for (size_t i = 0; i < configs.size(); ++i) {
        double sc = evaluate_candidate(data, configs[i], trials);
        scored.emplace_back(sc, i);
    }
    // Sort descending
    std::sort(scored.begin(), scored.end(),
              [](auto &a, auto &b){ return a.first > b.first; });

    // Return top_k as list of (score, cfg) tuples
    py::list out;
    for (int i = 0; i < std::min<int>(top_k, scored.size()); ++i) {
        auto [sc, idx] = scored[i];
        py::dict rec;
        rec["score"] = sc;
        rec["config"] = cfg_list[idx];
        out.append(rec);
    }
    return out;
}

PYBIND11_MODULE(optimizer, m) {
    m.doc() = "optimizer: trial‐based config optimization";
    m.def("optimize",
          &optimize,
          py::arg("configs"),
          py::arg("data"),
          py::arg("trials") = 5,
          py::arg("top_k")  = 3,
          "Return top_k configs after trial-based fitness evaluation");
}
