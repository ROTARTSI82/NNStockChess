#pragma once

#include <string>
#include <vector>

#include "trainer.hpp"
#include "murmurhash/MurmurHash3.hpp"

struct Puzzle {
    std::string Id,FEN,Moves,Rating,RatingDev,Popularity,NbPlays,Themes,GameUrl,OpeningFamily,OpeningVariation;

    inline constexpr std::string &operator[](std::size_t i) { return (&Id)[i]; }
    inline constexpr const std::string &operator[](std::size_t i) const { return (&Id)[i]; }
};

std::vector<Puzzle> load_puzzles_csv(const std::string &file);


struct HashedPosition {
    HashedPosition() = default;
    explicit HashedPosition(Position &pos);

    CastlingRights white, black;
    uint8_t board[64];

    constexpr bool operator==(const HashedPosition &rhs) const = default;
};

inline static uint64_t boost_hash_combine(uint64_t h, uint64_t k) {
    const uint64_t m = (uint64_t(0xc6a4a793) << 32) + 0x5bd1e995;
    const int r = 47;

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;

    // Completely arbitrary number, to prevent 0's
    // from hashing to 0.
    h += 0xe6546b64;

    return h;
}

namespace std {
    template <>
    struct hash<HashedPosition> {
        std::size_t operator()(const HashedPosition& k) const {
            uint64_t out[2];
            MurmurHash3_x86_128(&k, sizeof(HashedPosition), 0x2069d6d7, &out);
            return boost_hash_combine(out[0], out[1]);
        }
    };
}
