//
// Created by grant on 7/15/22.
//

#include "trainer.hpp"

#include "thread.h"
#include "types.h"
#include "timeman.h"
#include "search.h"

#include <sstream>

static inline constexpr NumericT bb_bool_to_numeric(Bitboard bb) {
    return bb != 0 ? 1.0 : 0.0;
}

void Network::save(const std::string &file) {
    std::cout << "SAVE\t";
    std::ofstream fd{file, std::ios::out | std::ios::binary};
    hid1.save(fd);
    hid2.save(fd);
    hid3.save(fd);
    hid4.save(fd);
    out.save(fd);
    fd.close();
}

void Network::load(const std::string &file) {
    std::cout << "LOAD\t";
    std::ifstream fd{file, std::ios::in | std::ios::binary};
    hid1.load(fd);
    hid2.load(fd);
    hid3.load(fd);
    hid4.load(fd);
    out.load(fd);
    fd.close();
}

void Network::apply_backprop() {
    std::cout << "EPOCH " << epoch++ << " DONE: " << num_samples << " samples, error = " << err / num_samples << '\n';
    err = 0;
    num_samples = 0;

    hid1.apply_backprop();
    hid2.apply_backprop();
    hid3.apply_backprop();
    hid4.apply_backprop();
    out.apply_backprop();

    save();
}

void Trainer::position_fen(const std::string &fen, const std::string &moves,
                           const std::function<void()> &callback) {

    states = StateListPtr(new std::deque<StateInfo>(1)); // Drop the old state and create a new one
    pos.set(fen, Options["UCI_Chess960"], &states->back(), nullptr);

    if (!moves.empty()) {
        std::string token;
        Move m;

        auto is = std::istringstream(moves);
        while (is >> token && (m = UCI::to_move(pos, token)) != MOVE_NONE) {
            callback();

            states->emplace_back();
            pos.do_move(m, states->back());
        }
    }

    callback();
}

void Trainer::train_line_here() {
    depth++;
    auto moveList = MoveList<LEGAL>(pos);
    if (moveList.size() == 0) {
        std::cout << "DEADEND " << pos.fen() << '\n';
        depth--;
        return; // no-op on all leaf positions
    }

    train_this_position();

    {
        StateInfo st{};
        ASSERT_ALIGNED(&st, Eval::NNUE::CacheLineSize);

        NumericT bestEval = std::numeric_limits<NumericT>::min();

        Move bestMove = *moveList.begin();

        for (const auto &mov: moveList) {
//        auto givesCheck = pos.gives_check(mov);
            pos.do_move(mov, st);


            if (MoveList<LEGAL>(pos).size() > 0) {
                train_this_position();

                NumericT netEval = net->out.activation[0][0] - net->out.activation[1][0];
                if (netEval > bestEval) {
                    bestEval = netEval;
                    bestMove = mov;
                }
            } else {
                std::cout << "DEADEND-INSEARCH " << pos.fen() << '\n';
            }

            pos.undo_move(mov);
        }

        if (bestEval != std::numeric_limits<NumericT>::min()) {
//        auto givesCheck = pos.gives_check(bestMove);
            std::cout << "Plays " << UCI::move(bestMove, false) << " in " << pos.fen() << '\n';
            pos.do_move(bestMove, st);
            train_line_here();
            pos.undo_move(bestMove);
        } else {
            std::cout << "DEADEND-NO_SELECTION " << pos.fen() << '\n';
        }
    }

    depth--;
}

void Trainer::train_this_position(const std::unordered_map<std::string, StockfishEval> *dataset) {
    eval_forward();

    std::string fen = clean_fen(pos);
    StockfishEval ev{};
    if (dataset != nullptr && dataset->count(fen) > 0)
        ev = dataset->at(fen);
    else {
        std::cout << "Cache miss\n";
        ev = stockfish_eval();
    }

    Vec<2> expected = Vec<2>{{{(NumericT)ev.win}, {NumericT(ev.loss)}}};

    net->hid1.backward(net->hid2.backward(net->hid3.backward(net->hid4.backward(net->out.backward(net->out.init_backwards(expected))))));

    auto outW = net->out.activation[0][0];
    auto outL = net->out.activation[1][0];

    std::cout << "d = " << depth << ", s = " << net->num_samples << "; outp = " << outW << ' ' << outL
              << ", real = " << ev.win << ' ' << ev.loss << '\n';

    net->num_samples++;
    auto err = std::pow(ev.win - outW, 2) + std::pow(ev.loss - outL, 2);
    net->err += err;

//    if (net->num_samples > 64)
//        net->apply_backprop();
}

StockfishEval Trainer::stockfish_eval() {
    Search::LimitsType limits;

    limits.startTime = now(); // The search starts as early as possible
     limits.depth = 16;
//    limits.nodes = 40000000;

    Threads.stop = true;
    Threads.main()->CUSTOM_done.store(false);

    Position cpy{};
    auto stateCpy = StateListPtr(new std::deque<StateInfo>(1)); // Drop the old state and create a new one
    cpy.set(pos.fen(), false, &stateCpy->back(), Threads.main());

    Threads.start_thinking(cpy, stateCpy, limits, false);

    {
        std::unique_lock<std::mutex> lg(Threads.main()->CUSTOM_mtx);
        Threads.main()->CUSTOM_cv.wait(lg, []{ return Threads.main()->CUSTOM_done.load(); });
    }

    Threads.stop = true;

    auto v = Threads.main()->CUSTOM_final_eval.load();
    auto ply = Threads.main()->CUSTOM_games_ply.load();
    auto wdl_w = win_rate_model( v, ply);
    auto wdl_l = win_rate_model(-v, ply);

    return StockfishEval{wdl_w, wdl_l, v};
}

void Trainer::eval_forward() const {
    std::size_t i = 0;

    for (Color col : {pos.side_to_move(), ~pos.side_to_move()}) {
        for (PieceType pt: {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}) {
            Bitboard mask = 1;

            for (int iter = 0; iter < 64; iter++) {
                net->inp[i++][0] = bb_bool_to_numeric(pos.pieces(pt) & pos.pieces(col) & mask);
                mask <<= 1;
            }
        }

        for (CastlingRights mask : {QUEEN_SIDE, KING_SIDE}) {
            net->inp[i++][0] = bb_bool_to_numeric(pos.castling_rights(col) & mask);
        }
    }

    Bitboard mask = 1, trans = pos.ep_square() != SQ_NONE ? square_bb(pos.ep_square()) : 0;
    for (int iter = 0; iter < 64; iter++) {
        net->inp[i++][0] = bb_bool_to_numeric(trans & mask);
        mask <<= 1;
    }

    net->inp[i++][0] = pos.rule50_count() / 50.0;

    net->hid1.forward();
    net->hid2.forward();
    net->hid3.forward();
    net->hid4.forward();
    net->out.forward();

    assert(i == INP_SIZE);
}

Trainer::Trainer() {
    states = StateListPtr(new std::deque<StateInfo>(1));
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false, &states->back(), nullptr);
}

Trainer::~Trainer() {
//    std::cout << pos << '\n'; // dump final position
}

void Trainer::position_fen(const std::string &fen) {
    states = StateListPtr(new std::deque<StateInfo>(1)); // Drop the old state and create a new one
    pos.set(fen, false, &states->back(), nullptr);
}

double win_rate_model(Value v, int ply) {
    // The model only captures up to 240 plies, so limit the input and then rescale
    double m = std::min(240, ply) / 64.0;

    // The coefficients of a third-order polynomial fit is based on the fishtest data
    // for two parameters that need to transform eval to the argument of a logistic
    // function.
    double as[] = {-1.17202460e-01, 5.94729104e-01, 1.12065546e+01, 1.22606222e+02};
    double bs[] = {-1.79066759,  11.30759193, -17.43677612,  36.47147479};
    double a = (((as[0] * m + as[1]) * m + as[2]) * m) + as[3];
    double b = (((bs[0] * m + bs[1]) * m + bs[2]) * m) + bs[3];

    // Transform the eval to centipawns with limited range
    double x = std::clamp(double(100 * v) / static_cast<int>(PawnValueEg), -2000.0, 2000.0);

    // Return the win rate in per mille units rounded to the nearest value
    return 1.0 / (1 + std::exp((a - x) / b));
}
