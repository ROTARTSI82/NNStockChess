#pragma once

#include <string>
#include <vector>

#include "trainer.hpp"

struct Puzzle {
    std::string Id,FEN,Moves,Rating,RatingDev,Popularity,NbPlays,Themes,GameUrl,OpeningFamily,OpeningVariation;

    inline constexpr std::string &operator[](std::size_t i) { return (&Id)[i]; }
    inline constexpr const std::string &operator[](std::size_t i) const { return (&Id)[i]; }
};

std::vector<Puzzle> load_puzzles_csv(const std::string &file);

struct Dataset {
    std::unordered_map<std::string, StockfishEval> gen;

    std::vector<Value> med;
    std::int64_t accum = 0;
    std::int64_t avg_divisor = 1;

    std::vector<Puzzle> dataset;
    std::uniform_int_distribution<std::size_t> dist;

    Dataset();

    void load_from_bin(const std::string &file);

    void print();

    void mm_print();
};
