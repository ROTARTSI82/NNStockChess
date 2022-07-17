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

    {
        Trainer trainer{};
        trainer.net->load();
        trainer.train_line_here();
        trainer.net->apply_backprop();
        trainer.net->save();
    }

    UCI::loop(argc, argv);



    Threads.set(0);
    return 0;
}
