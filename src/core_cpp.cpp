#include <pybind11/pybind11.h>
namespace py = pybind11;

// A “heavy” compute routine
int heavy_compute(int x) {
    // pretend this is expensive
    return x * x + 42;
}

PYBIND11_MODULE(core_cpp, m) {
    m.doc() = "core_cpp: heavy C++ routines";
    m.def("heavy_compute", &heavy_compute, "An expensively trivial function");
}
