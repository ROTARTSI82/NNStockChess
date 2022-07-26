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



void generate_training_data() {
    Dataset set;
    set.load_from_bin("traindata.bin");
    set.print();
//    set.mm_print();

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

        NumericT avgW = set.accum_w / set.avg_divisor;
        NumericT avgL = set.accum_l / set.avg_divisor;

//        NumericT dw = avgW - 0.5;
//        NumericT dl = avgL - 0.5;

        if (chanceIgn(mt64) > failChance && std::signbit<int>(ev.eval) == std::signbit(set.accum)) return;

//        double rejectionChance = ev.eval > 0 ? 5 : 10; // chance of rejecting a perfectly good candidate
//        if (ev.eval > 0)
//            rejectionChance *= std::clamp(std::abs(ev.eval) / 1600.0, 0.0, 1.0); // prefer low positive values
//        else
//            rejectionChance *= std::clamp(1 - std::abs(ev.eval) / 3200.0, 0.0, 1.0); // prefer high negative values
//        if (chanceIgn(mt64) < rejectionChance) return;

        set.gen[cleanFen] = ev;

        set.accum += ev.eval;
        set.accum_w += ev.win;
        set.accum_l += ev.loss;
        set.avg_divisor++;

        std::cout << counter++ << "\t" << cleanFen << '\t' << ev.eval << "\twlr " << ev.win << ' '
                  << ev.loss << '\t' << sec << "sec" << "\t avg " << set.accum / set.avg_divisor
                  << "\tavgwdl " << avgW << " " << avgL << std::endl;
    };


    while (true) {
        const Puzzle *p;
        do {
            p = &set.dataset.at(set.dist(mt64));
        } while (!(p->Themes.contains("equality")));

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

        if (counter > 256) {
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
    trainer.net->load();

    Dataset set;
    set.load_from_bin("traindata.bin");
   

    set.print();

    int num = 0; // 10240
    while (set.gen.size() > 4) {
        auto it = set.gen.begin();

        auto dist = std::uniform_int_distribution<std::size_t>(0, set.gen.size() - 1);
        std::advance(it, dist(mt64));

        trainer.position_fen(it->first);
        trainer.train_this_position(&set.gen);

        if (num++ % 512 == 511) {
            trainer.net->apply_backprop();
            trainer.net->save();
            std::cout << " ================================ [ NETWORK SAVED! num = " << num
                      << ", left = " << set.gen.size() << " ] ================================\n";
        }

        set.gen.erase(it);
    }

    trainer.net->apply_backprop();
    trainer.net->save();
}

int main(int argc, char* argv[]) {
//    feenableexcept(FE_INVALID | FE_OVERFLOW);
// feenableexcept(FE_INVALID);

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

//    generate_training_data();
    train_network();

//    UCI::loop(argc, argv);

    Threads.set(0);
    return 0;
}
