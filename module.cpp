#include <pybind11/pybind11.h>
#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <tuple>
#include <cinttypes>

namespace py = pybind11;
using namespace Eigen;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) {
    return i + j;
}

/*

                                for x in range(wallSize, size+wallSize):
            for y in range(wallSize, size+wallSize):
                features = np.empty((0,), dtype=int)
                if game._covered_board[x, y] == True:
                        for i in range(x-wallSize, x+1+wallSize):
                            for j in range(y-wallSize, y+1+wallSize):
                                one_hot = np.zeros(11)
                                if (i, j) == (x, y):
                                    continue
                                if(game._board[i, j] == -2):
                                    one_hot[10] = 1
                                    features = np.append(features, one_hot)
                                else:
                                    one_hot[int(game._board[i, j])] = 1
                                    features = np.append(features, one_hot)
                        dataPoints.append(features)
                        coords.append((x, y))
*/


void call_python(pybind11::object obj) {
    // py::scoped_interpreter guard{}; // start the interpreter and keep it alive
    py::object table_attr = obj.attr("table");
    // py::array table = table_attr.cast<py::array>();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            // std::cout << *(double*)(table.data() + j + i*3) << " ";
            double num = (table_attr.attr("__getitem__")(py::make_tuple(i, j))).cast<double>();
            std::cout << num << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}
/*
            for y in range(wallSize, 10+wallSize):
                for x in range(wallSize, 10+wallSize):
                        features = np.empty((0,), dtype=int)
                        if game._covered_board[x, y] == True:
                            for i in range(x-neighRange, x+1+neighRange):
                                for j in range(y-neighRange, y+1+neighRange):
                                    one_hot = np.zeros(10, dtype=int)
                                    if (i, j) == (x, y):
                                        continue
                                    if game._covered_board[i, j] == True:
                                        one_hot[9] = 1
                                        features = np.append(features, one_hot)
                                    #One hot with drop
                                    else:
                                        if(game._board[i, j] != -2):
                                            one_hot[int(game._board[i, j])] = 1
                                        features = np.append(features, one_hot)
                            if game._board[x, y] == -1:
                                features = np.append(features, 1)
                            else:
                                features = np.append(features, 0)
                            dataPoints.append(features)
                        */
std::vector<std::uint8_t> create_datapoint(py::EigenDRef<Matrix<bool, Dynamic, Dynamic>> covered_board, py::EigenDRef<Matrix<int, Dynamic, Dynamic>> board, int wallSize, int neighRange, int grid_size) {
    std::vector<std::uint8_t> data_point;
    data_point.reserve(99*((2*neighRange+1)*(2*neighRange+1) - 1)*10);
    std::vector<std::uint8_t> one_hot(10, 0);
    for (int y = wallSize; y < grid_size + wallSize; y++) {
        for (int x = wallSize; x < grid_size + wallSize; x++) {
            if (covered_board(x,y) == true) {
                for (int i = x - neighRange; i < x + 1 + neighRange; i++) {
                    for (int j = y - neighRange; j < y + 1 + neighRange; j++) {
                        one_hot = {0,0,0,0,0,0,0,0,0,0};
                        if (i == x && j ==  y) {
                            continue;
                        }
                        if (covered_board(i, j) == true) {
                            one_hot[9] = 1;
                        } else {
                            if (board(i,j) != -2) {
                                one_hot[board(i,j)] = 1;
                            }
                        }
                        data_point.insert(data_point.end(), one_hot.begin(), one_hot.end());
                    }
                }
                data_point.emplace_back(board(x,y) == -1 ? 1 : 0);
            }
        }
    }
    return data_point;
}

/*
            for y in range(wallSize, 10+wallSize):
                for x in range(wallSize, 10+wallSize):
                        features = np.empty((0,), dtype=int)
                        if game._covered_board[x, y] == True:
                            for i in range(x-neighRange, x+1+neighRange):
                                for j in range(y-neighRange, y+1+neighRange):
                                    one_hot = np.zeros(10, dtype=int)
                                    if (i, j) == (x, y):
                                        continue
                                    if game._covered_board[i, j] == True:
                                        one_hot[9] = 1
                                        features = np.append(features, one_hot)
                                    #One hot with drop
                                    else:
                                        if(game._board[i, j] != -2):
                                            one_hot[int(game._board[i, j])] = 1
                                        features = np.append(features, one_hot)
                            if game._board[x, y] == -1:
                                features = np.append(features, 1)
                            else:
                                features = np.append(features, 0)
                            dataPoints.append(features)


        for x in range(wallSize, size+wallSize):
            for y in range(wallSize, size+wallSize):
                features = np.empty((0,), dtype=int)
                if game._covered_board[x, y] == True:
                        for i in range(x-wallSize, x+1+wallSize):
                            for j in range(y-wallSize, y+1+wallSize):
                                one_hot = np.zeros(11)
                                if (i, j) == (x, y):
                                    continue
                                if(game._covered_board[i, j] == True):
                                    one_hot[9] = 1
                                    features = np.append(features, one_hot)
                                elif(game._board[i, j] == -2):
                                    one_hot[10] = 1
                                    features = np.append(features, one_hot)
                                else:
                                    one_hot[int(game._board[i, j])] = 1
                                    features = np.append(features, one_hot)
                        dataPoints.append(features)
                        coords.append((x, y))
*/
std::tuple<std::vector<std::uint8_t>, std::vector<py::tuple>> generate_predict_features(py::EigenDRef<Matrix<bool, Dynamic, Dynamic>> covered_board, py::EigenDRef<Matrix<int, Dynamic, Dynamic>> board, int wallSize, int neighRange, int grid_size) {
    std::vector<std::uint8_t> data_point;
    std::vector<py::tuple> coords;
    data_point.reserve(99*((2*neighRange+1)*(2*neighRange+1) - 1)*10);
    coords.reserve(size*size);
    std::vector<std::uint8_t> one_hot(10, 0);
    for (int y = wallSize; y < grid_size + wallSize; y++) {
        for (int x = wallSize; x < grid_size + wallSize; x++) {
            if (covered_board(x,y) == true) {
                for (int i = x - neighRange; i < x + 1 + neighRange; i++) {
                    for (int j = y - neighRange; j < y + 1 + neighRange; j++) {
                        one_hot = {0,0,0,0,0,0,0,0,0,0};
                        if (i == x && j ==  y) {
                            continue;
                        }
                        if (covered_board(i, j) == true) {
                            one_hot[9] = 1;
                        } else {
                            if (board(i,j) != -2) {
                                one_hot[board(i,j)] = 1;
                            }
                        }
                        data_point.insert(data_point.end(), one_hot.begin(), one_hot.end());
                    }
                }
                coords.emplace_back(py::make_tuple(x, y));
            }
        }
    }
    return std::make_tuple(data_point, coords);
}

std::vector<std::vector<int>> test_vec_in_vec() {
    std::vector<std::vector<int>> v;
    for (int i = 0; i < 5; i++)
    v.push_back(std::vector<int>(5, i));
    return v;
}

PYBIND11_MODULE(pybind_testing, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

    m.def("call_python", &call_python);
    m.def("create_datapoint", &create_datapoint);
    m.def("test_vec_in_vec", &test_vec_in_vec);
    m.def("generate_predict_features", &generate_predict_features);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
