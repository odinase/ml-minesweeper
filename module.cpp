#include <pybind11/pybind11.h>
#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <iostream>
#include <vector>

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
                                one_hot = np.zeros(11, dtype=int)
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
                        if game._board[x, y] == -1:
                            features = np.append(features, 1)
                        else:
                            features = np.append(features, 0)
                        dataPoints.append(features)
                        */
std::vector<int> gather_datapoints(py::EigenDRef<Matrix<bool, Dynamic, Dynamic>> covered_board, py::EigenDRef<Matrix<int, Dynamic, Dynamic>> board, int wallSize, int neighRange, int grid_size) {
    std::vector<int> dataPoints;
    for (int y = wallSize; y < grid_size + wallSize; y++) {
        for (int x = wallSize; x < grid_size + wallSize; x++) {
            std::vector<int> features;
            if (covered_board(x,y) == true) {
                for (int i = x - neighRange; i < x + 1 + neighRange; i++) {
                    for (int j = y - neighRange; j < y + 1 + neighRange; j++) {
                        std::vector<int> one_hot(11, 0);
                        if (i == x && j ==  y) {
                            continue;
                        }
                        if (covered_board(i, j) == true) {
                            one_hot[9] = 1;
                        } else if (board(i,j) == -2) {
                            one_hot[10] = 1;
                        } else {
                            if (board(i,j) < 0 || board(i,j) > 10)
                            std::cout << board(i,j) << std::endl;
                            one_hot[board(i,j)] = 1;
                        }
                        features.insert(features.end(), one_hot.begin(), one_hot.end());
                    }
                }
                features.emplace_back(board(x,y) == -1 ? 1 : 0);
                dataPoints.insert(dataPoints.end(), features.begin(), features.end());
            }
        }
    }
    return dataPoints;
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
    m.def("gather_datapoints", &gather_datapoints);
    m.def("test_vec_in_vec", &test_vec_in_vec);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
