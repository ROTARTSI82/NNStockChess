#pragma once

#include "position.h"
#include "uci.h"

#include "nn_linalg.hpp"

using namespace Stockfish;


/**
 * Number of input layers:
 * 64 boardsize * 6 piece types * 2 sides
 *      PLUS
 * 64 possible en passant squares
 *      PLUS
 * 4 castling rights
 *      PLUS
 * 1 halfmove clock percentage (scales from 0 to 1, hits 1 at 50 halfmoves)
 */
constexpr auto INP_SIZE = 64*6*2 + 64 + 4 + 1;
constexpr auto L2_SIZE = 32768; // 16384;


/**
 * Output format:
 * [win %, loss %]
 *
 * Draw % is implied
 */

class Network {
public:
    NumericT err = 0;
    unsigned epoch = 0, num_samples = 0;

    Vec<INP_SIZE> inp{};
    Layer<INP_SIZE, INP_SIZE> hid1;
    Layer<INP_SIZE, L2_SIZE> hid2;
    Layer<L2_SIZE, 512> hid3;
    Layer<512, 64> hid4;
    Layer<64, 2> out;

    Network() {
        std::cout << "NET CTOR\n";
        hid1.input = &inp;
        hid2.input = &hid1.activation;
        hid3.input = &hid2.activation;
        hid4.input = &hid3.activation;
        out.input = &hid4.activation;

        hid1.randomize();
        hid2.randomize();
        hid3.randomize();
        hid4.randomize();
        out.randomize();
    }

    void save(const std::string &file = "net2.nn");
    void load(const std::string &file = "net2.nn");

    void apply_backprop();
};

struct StockfishEval {
    double win, loss;
    Stockfish::Value eval;
};

inline std::string clean_fen(Position &p) {
    std::string cleanFen = p.fen();
    return cleanFen.substr(0, cleanFen.rfind(' ', cleanFen.rfind(' ') - 1));
}

class Trainer {
public:
    std::unique_ptr<Network> net; // = std::make_unique<Network>();

    Position pos{};
    StateListPtr states{new std::deque<StateInfo>(1)};

    int depth = 0;

    Trainer();
    ~Trainer();

    void position_fen(const std::string &fen, const std::string &moves, const std::function<void()> &callback);

    void train_line_here();

    void train_this_position(const std::unordered_map<std::string, StockfishEval> *dataset = nullptr);

    StockfishEval stockfish_eval();

    void eval_forward() const;

    void position_fen(const std::string &fen);
};

// The win rate model returns the probability of winning (in per mille units) given an
// eval and a game ply. It fits the LTC fishtest statistics rather accurately.
double win_rate_model(Value v, int ply);

