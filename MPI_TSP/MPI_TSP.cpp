#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <limits>
#include <sstream> 
#include <iterator>

// Struktura do przechowywania współrzędnych miasta
struct City {
    double x, y;
};

// Funkcja do załadowania danych z pliku TSPLIB
std::vector<City> load_tsplib(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<City> cities;
    std::string line;
    bool node_section = false;

    while (std::getline(file, line)) {
        if (line == "NODE_COORD_SECTION") {
            node_section = true;
            continue;
        }
        if (node_section) {
            if (line == "EOF") {
                break;
            }
            double x, y;
            int index;
            std::istringstream iss(line);
            iss >> index >> x >> y;
            cities.push_back({ x, y });
        }
    }
    return cities;
}

// Funkcja do obliczania odległości euklidesowej
double calculate_distance(const City& a, const City& b) {
    return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
}

// Funkcja do tworzenia macierzy odległości
std::vector<std::vector<double>> create_distance_matrix(const std::vector<City>& cities) {
    int num_cities = cities.size();
    std::vector<std::vector<double>> distance_matrix(num_cities, std::vector<double>(num_cities));

    for (int i = 0; i < num_cities; ++i) {
        for (int j = 0; j < num_cities; ++j) {
            distance_matrix[i][j] = calculate_distance(cities[i], cities[j]);
        }
    }
    return distance_matrix;
}

// Funkcja oceny rozwiązania
double fitness(const std::vector<int>& solution, const std::vector<std::vector<double>>& distance_matrix) {
    double total_distance = 0.0;
    int num_cities = solution.size();
    for (int i = 0; i < num_cities - 1; ++i) {
        total_distance += distance_matrix[solution[i]][solution[i + 1]];
    }
    total_distance += distance_matrix[solution.back()][solution[0]];
    return total_distance;
}

// Funkcja mutacji
void mutate(std::vector<int>& solution) {
    int i = rand() % solution.size();
    int j = rand() % solution.size();
    std::swap(solution[i], solution[j]);
}

// Funkcja krzyżowania
std::vector<int> crossover(const std::vector<int>& parent1, const std::vector<int>& parent2) {
    int size = parent1.size();
    std::vector<int> child(size, -1);
    int start = rand() % size;
    int end = rand() % size;
    if (start > end) std::swap(start, end);
    for (int i = start; i <= end; ++i) {
        child[i] = parent1[i];
    }
    int current_index = 0;
    for (int i = 0; i < size; ++i) {
        if (std::find(child.begin(), child.end(), parent2[i]) == child.end()) {
            while (child[current_index] != -1) {
                ++current_index;
            }
            child[current_index] = parent2[i];
        }
    }
    return child;
}

// Funkcja do uruchomienia algorytmu genetycznego
std::pair<std::vector<int>, double> run_genetic_algorithm(const std::vector<std::vector<double>>& distance_matrix, int population_size, int generations, double mutation_rate) {
    int num_cities = distance_matrix.size();
    std::vector<std::vector<int>> population(population_size, std::vector<int>(num_cities));
    for (auto& individual : population) {
        for (int i = 0; i < num_cities; ++i) {
            individual[i] = i;
        }
        std::random_shuffle(individual.begin(), individual.end());
    }

    std::vector<int> best_solution = population[0];
    double best_fitness = fitness(best_solution, distance_matrix);

    for (int generation = 0; generation < generations; ++generation) {
        std::vector<std::vector<int>> new_population;
        for (int i = 0; i < population_size; ++i) {
            int parent1_idx = rand() % population_size;
            int parent2_idx = rand() % population_size;
            std::vector<int> child = crossover(population[parent1_idx], population[parent2_idx]);
            if ((rand() / double(RAND_MAX)) < mutation_rate) {
                mutate(child);
            }
            new_population.push_back(child);
        }
        population = new_population;

        for (const auto& individual : population) {
            double current_fitness = fitness(individual, distance_matrix);
            if (current_fitness < best_fitness) {
                best_solution = individual;
                best_fitness = current_fitness;
            }
        }
    }

    return { best_solution, best_fitness };
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string file_path = "berlin52.tsp";  // <-- Nazwa pliku wpisana bezpośrednio w kodzie

    std::vector<City> cities;
    std::vector<std::vector<double>> distance_matrix;

    if (rank == 0) {
        cities = load_tsplib(file_path);
        distance_matrix = create_distance_matrix(cities);
    }

    int num_cities;
    if (rank == 0) {
        num_cities = cities.size();
    }
    MPI_Bcast(&num_cities, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        distance_matrix.resize(num_cities, std::vector<double>(num_cities));
    }

    for (int i = 0; i < num_cities; ++i) {
        MPI_Bcast(distance_matrix[i].data(), num_cities, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int population_size = 100;
    int generations = 500;
    double mutation_rate = 0.01;

    srand(time(0) + rank); // Różne ziarno dla każdego procesu
    auto result = run_genetic_algorithm(distance_matrix, population_size, generations, mutation_rate);
    std::vector<int> best_solution = result.first;
    double best_fitness = result.second;

    std::vector<int> global_best_solution(num_cities);
    double global_best_fitness;

    struct {
        double fitness;
        int rank;
    } local_best{ best_fitness, rank }, global_best;

    MPI_Allreduce(&local_best, &global_best, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

    if (rank == global_best.rank) {
        global_best_solution = best_solution;
    }

    MPI_Bcast(global_best_solution.data(), num_cities, MPI_INT, global_best.rank, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Best solution: ";
        for (const auto& city : global_best_solution) {
            std::cout << city << " ";
        }
        std::cout << "\nShortest path: " << global_best.fitness << std::endl;
    }

    MPI_Finalize();
    return 0;
}