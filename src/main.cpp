#include <iostream>


#include "bitboard.h"
#include "endgame.h"
#include "position.h"
#include "psqt.h"
#include "search.h"
#include "syzygy/tbprobe.h"
#include "thread.h"
#include "tt.h"
#include "uci.h"

#include "traindata.hpp"
#include "trainer.hpp"
#include <cfenv>

#include <random>

using namespace Stockfish;

std::random_device dev{};
std::mt19937_64 mt64(dev());


inline std::string clean_fen(Position &p) {
    std::string cleanFen = p.fen();
    return cleanFen.substr(0, cleanFen.rfind(' ', cleanFen.rfind(' ') - 1));
}


void generate_training_data() {
    Dataset set;
    set.load_from_bin("traindata.bin");
    set.print();
    set.mm_print();

    Trainer trainer{};

    int counter = 0;
    auto eval = [&]() {
        std::string cleanFen = clean_fen(trainer.pos);
        if (MoveList<LEGAL>(trainer.pos).size() == 0 || set.gen.count(cleanFen) > 0) return;


        auto start = std::chrono::high_resolution_clock::now();
        StockfishEval ev = trainer.stockfish_eval();
        auto diff = std::chrono::high_resolution_clock::now() - start;
        auto sec = std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() / 1000.0;

        double failChance = ev.eval > 0 ? 80 : 10; // change of adding position anyway despite being wrong direction
        if (ev.eval > 0)
            failChance *= std::min(1 - std::abs(ev.eval) / 1600.0, 1.0); // prefer low positive values
        else
            failChance *= std::min(std::abs(ev.eval) / 1600.0, 1.0); // prefer high negative values

        std::uniform_real_distribution<double> chanceIgn(0, 100);
        if (chanceIgn(mt64) > failChance && std::signbit<int>(ev.eval) == std::signbit(set.accum)) return;

        double rejectionChance = ev.eval > 0 ? 5 : 10; // chance of rejecting a perfectly good candidate
        if (ev.eval > 0)
            rejectionChance *= std::clamp(std::abs(ev.eval) / 1600.0, 0.0, 1.0); // prefer low positive values
        else
            rejectionChance *= std::clamp(1 - std::abs(ev.eval) / 3200.0, 0.0, 1.0); // prefer high negative values
        if (chanceIgn(mt64) < rejectionChance) return;

        set.gen[cleanFen] = ev;

        set.accum += ev.eval;
        set.avg_divisor++;

        std::cout << counter++ << "\t" << cleanFen << '\t' << ev.eval << "\twlr " << ev.win << ' '
                  << ev.loss << '\t' << sec << "sec" << "\t avg " << set.accum / set.avg_divisor << std::endl;
    };


    while (true) {
        const Puzzle *p;
        do {
            p = &set.dataset.at(set.dist(mt64));
        } while (!(p->Themes.contains("opening")));

        trainer.position_fen(p->FEN, p->Moves, [&]() {
            eval();
//                StateInfo st{};
//
//                for (auto m :  MoveList<LEGAL>(trainer.pos)) {
//                    trainer.pos.do_move(m, st);
//                    eval();
//                    trainer.pos.undo_move(m);
//                }
        });

        if (counter > 72) {
            std::ofstream fd("traindata.bin", std::ios::out | std::ios::binary);
            for (const auto &v: set.gen) {
                fd << 'N' << v.first << ',';
                fd.write((const char *) &v.second, sizeof(StockfishEval));
            }

            fd.close();
            counter = 0;
//                break;
        }
    }
}


void train_network() {
    Trainer trainer{};
    trainer.net = std::make_unique<Network>();

    Dataset set;
    set.load_from_bin("puzzles.bin");

    set.print();

    while (true) {
        auto randPos = set.gen.begin();
        std::advance(randPos, set.dist(mt64));

        trainer.states = StateListPtr(new std::deque<StateInfo>(1));
//        trainer.pos.unhash_from(randPos->first, false, &trainer.states->back(), nullptr);

        StateInfo *si = &trainer.states->back();
    }
}

int main(int argc, char* argv[]) {
//    feenableexcept(FE_INVALID | FE_OVERFLOW);

    std::cout << "Hello, World!" << std::endl;
    std::cout << engine_info() << std::endl;

    CommandLine::init(argc, argv);
    UCI::init(Options);
    Tune::init();
    PSQT::init();
    Bitboards::init();
    Position::init();
    Bitbases::init();
    Endgames::init();
    Threads.set(size_t(Options["Threads"]));
    Search::clear(); // After threads are up
    Eval::NNUE::init();

    generate_training_data();

//    UCI::loop(argc, argv);

    Threads.set(0);
    return 0;
}
