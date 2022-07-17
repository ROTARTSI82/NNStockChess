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

    std::random_device dev;
    std::mt19937_64 rng(dev());

    {
        std::unordered_map<HashedPosition, StockfishEval> gen;

        std::vector<Value> med;
        std::int64_t accum = 0;
        std::int64_t avg_divisor = 0;

        {


            std::ifstream fd("traindata.bin", std::ios::in | std::ios::binary);
            while (!fd.eof()) {
                HashedPosition p;
                StockfishEval ev{};
                char c;

                fd >> c;
                if (c != 'N') {
                    std::cerr << "corrupt N\n";
                    continue;
                }

                fd.read((char *) &p, sizeof(HashedPosition));

                fd >> c;
                if (c != ',') {
                    std::cerr << "corrupt\n";
                    continue;
                }

                fd.read((char *) &ev, sizeof(StockfishEval));

                gen[p] = ev;

                accum += ev.eval;
                avg_divisor++;
                med.emplace_back(ev.eval);
            }

            fd.close();
        }


        std::sort(med.begin(), med.end());
        for (auto v : med)
            std::cout << '\t' << v;

        Trainer trainer{};
        std::vector<Puzzle> dataset = load_puzzles_csv("/home/shared/chess/lichess_db_puzzle.csv");
        std::cout << "No. Puzzles = " << dataset.size() << '\n';
        std::cout << "Loaded " << gen.size() << " positions from previous runs\n";
        std::cout << "Average eval: " << accum / avg_divisor << '\n';
        std::cout << "Median eval: " << med.at(med.size() / 2) << '\n';

        std::uniform_int_distribution<std::size_t> dist(0, dataset.size() - 1);

        int counter = 0;
        auto eval = [&]() {
            auto hp = HashedPosition(trainer.pos);
            if (MoveList<LEGAL>(trainer.pos).size() == 0 || gen.count(hp) > 0) return;


            auto start = std::chrono::high_resolution_clock::now();
            StockfishEval ev = trainer.stockfish_eval();
            auto diff = std::chrono::high_resolution_clock::now() - start;
            auto sec = std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() / 1000.0;

//            if (ev.eval > 0) return;

            gen[hp] = ev;

            accum += ev.eval;
            avg_divisor++;

            std::cout << counter++ << "\t" << trainer.pos.fen() << '\t' << ev.eval << "\twlr " << ev.win << ' ' << ev.loss << '\t' << sec << "sec" << "\t avg " << accum / avg_divisor << std::endl;
        };


        while (true) {
            const Puzzle *p;
            do {
                p = &dataset.at(dist(rng));
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

            if (counter > 64) {
                std::ofstream fd("traindata.bin", std::ios::out | std::ios::binary);
                for (const auto &v: gen) {
                    fd << 'N';
                    fd.write((const char *) &v.first, sizeof(HashedPosition));
                    fd << ',';
                    fd.write((const char *) &v.second, sizeof(StockfishEval));
                }

                fd.close();
                counter = 0;
                break;
            }
        }
    }

//    UCI::loop(argc, argv);

    Threads.set(0);
    return 0;
}
