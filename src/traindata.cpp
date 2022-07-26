#include "traindata.hpp"

#include <fstream>
#include <iostream>
// 5rk1/1p3ppp/pq3b2/8/8/1P1Q1N2/P4PPP/3R2K1 w - - 2 27
std::vector<Puzzle> load_puzzles_csv(const std::string &file) {
    std::vector<Puzzle> ret;

    std::ifstream fd(file);

    for (std::string line; std::getline(fd, line); ) {
        ret.emplace_back();

        std::size_t index = 0;
        std::size_t prev = 0;
        std::size_t comma = line.find(',');
        while (comma != std::string::npos) {
            ret.back()[index++] = line.substr(prev, comma - prev);

            prev = comma + 1;
            comma = line.find(',', prev);
        }

        ret.back()[index++] = line.substr(prev);

        if (index != 11 && index != 10) {
            std::cerr << "Invalid line: index = " << index << '\n';
            ret.pop_back();
        }
    }

    fd.close();

    return ret;
}


Dataset::Dataset() : dataset(load_puzzles_csv("/home/shared/chess/lichess_db_puzzle.csv")),
                     dist(0, dataset.size() - 1) {}

void Dataset::load_from_bin(const std::string &file) {
    std::ifstream fd(file, std::ios::in | std::ios::binary);
    while (!fd.eof()) {
        StockfishEval ev{};
        char c;

        fd >> c;
        if (c != 'N') {
            std::cerr << "corrupt N\n";
            continue;
        }

        std::string fen;

        c = (char) fd.get();
        do {
            fen += c;
            c = (char) fd.get();
        } while (c != ',');

        fd.read((char *) &ev, sizeof(StockfishEval));

        gen[fen] = ev;

        accum += ev.eval;
        accum_w += ev.win;
        accum_l += ev.loss;
        avg_divisor++;
        med.emplace_back(ev.eval);
    }

    fd.close();
}

void Dataset::print() {
    std::sort(med.begin(), med.end());

    std::cout << "No. Puzzles = " << dataset.size() << '\n';
    std::cout << "Loaded " << gen.size() << " positions from previous runs\n";
    std::cout << "Average eval: " << accum / avg_divisor << '\n';

    if (!med.empty())
        std::cout << "Median eval: " << med.at(med.size() / 2) << '\n';
}

void Dataset::mm_print() {
    std::cout << "Histogram[{";
    for (auto &v: gen)
        std::cout << std::round(v.second.win * 1000) / 1000.0 << ", ";
    std::cout << "}, 200, Ticks -> {Table[-5000 + 200 i, {i, 0, 100}], Automatic}]" << std::endl;

//    for (auto &v : gen)
//        std::cout << v.first << '\n';
}
