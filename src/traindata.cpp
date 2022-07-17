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

HashedPosition::HashedPosition(Position &pos) : white(pos.castling_rights(WHITE)), black(pos.castling_rights(BLACK)) {
    constexpr auto EP_SQ_VALUE = 244;
    for (int i = 0; i < 64; i++) {
        Square sq = static_cast<Square>(SQ_A1 + i);
        board[i] = pos.piece_on(sq);
        if (sq == pos.ep_square() && pos.ep_square() != SQ_NONE) {
            board[i] = EP_SQ_VALUE;
        }
    }
}
