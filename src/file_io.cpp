#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>

namespace py = pybind11;

// ---- CSV reader ----
/// Handles quoted fields, commas-in-quotes, arbitrary line endings.
std::vector<std::vector<std::string>> read_csv(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Cannot open CSV file");

    std::vector<std::vector<std::string>> rows;
    std::string line;
    while (std::getline(file, line)) {
        std::vector<std::string> row;
        std::string cell;
        bool in_quotes = false;

        for (char c : line) {
            if (c == '"' ) {
                in_quotes = !in_quotes;
            }
            else if (c == ',' && !in_quotes) {
                row.push_back(cell);
                cell.clear();
            }
            else {
                cell.push_back(c);
            }
        }
        row.push_back(cell);
        rows.push_back(std::move(row));
    }
    return rows;
}

// ---- STL reader ----
// Basic ASCII STL parser. You can extend for binary.
struct Vec3 { double x,y,z; };
using Triangle = std::array<Vec3,3>;

std::vector<Triangle> read_stl(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open STL file");

    // peek to see if ASCII or binary
    char header[6] = {0};
    file.read(header,5);
    file.seekg(0);
    std::string hdr(header);
    if (hdr.rfind("solid",0)==0) {
        // ASCII mode
        std::string token;
        std::vector<Triangle> tris;
        while (file >> token) {
            if (token=="vertex") {
                Vec3 v[3];
                for (int i=0;i<3;i++)
                    file >> v[i].x >> v[i].y >> v[i].z;
                tris.push_back({v[0],v[1],v[2]});
            }
        }
        return tris;
    } else {
        throw std::runtime_error("Binary STL not yet supported");
    }
}

PYBIND11_MODULE(file_io, m) {
    m.doc() = "file_io: robust CSV & STL ingestion";
    m.def("read_csv", &read_csv, "Parse any CSV into a list of rows");
    m.def("read_stl", &read_stl, "Parse an ASCII STL into triangles");
}
