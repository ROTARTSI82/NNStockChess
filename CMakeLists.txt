cmake_minimum_required(VERSION 3.22)

set(CMAKE_C_COMPILER /usr/bin/clang)
set(CMAKE_CXX_COMPILER /usr/bin/clang++)
set(CMAKE_CXX_STANDARD 23)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    message("-- Using ccache")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache) # Less useful to do it for linking, see edit2
endif(CCACHE_FOUND)

project(NNStockChess)

file(GLOB_RECURSE STOCKFISH_SOURCES CONFIGURE_DEPENDS dep/Stockfish/src/*.cpp)
list(FILTER STOCKFISH_SOURCES EXCLUDE REGEX ".*main\\.cpp$")

file(GLOB_RECURSE MY_SOURCES CONFIGURE_DEPENDS src/*.cpp)

add_executable(NNStockChess ${MY_SOURCES} ${STOCKFISH_SOURCES})

target_include_directories(NNStockChess PUBLIC dep/Stockfish/src src)

message("-- Ofast build")
set(STOCK_COMP_FLAGS -Wall -Wcast-qual -fno-exceptions -std=c++17  -pedantic -Wextra -Wshadow -m64 -DUSE_PTHREADS -DNDEBUG -Ofast -fexperimental-new-pass-manager -DIS_64BIT -msse -msse3 -mpopcnt -DUSE_POPCNT -DUSE_AVX2 -mavx2 -DUSE_SSE41 -msse4.1 -DUSE_SSSE3 -mssse3 -DUSE_SSE2 -msse2 -DUSE_PEXT -mbmi2 -flto)
set(STOCK_LINK_FLAGS -latomic -m64 -lpthread  -Wall -Wcast-qual -fno-exceptions -std=c++17  -pedantic -Wextra -Wshadow -m64 -DUSE_PTHREADS -DNDEBUG -Ofast -fexperimental-new-pass-manager -DIS_64BIT -msse -msse3 -mpopcnt -DUSE_POPCNT -DUSE_AVX2 -mavx2 -DUSE_SSE41 -msse4.1 -DUSE_SSSE3 -mssse3 -DUSE_SSE2 -msse2 -DUSE_PEXT -mbmi2 -flto)

#set(STOCK_COMP_FLAGS -Wall -Wcast-qual -std=c++17  -pedantic -Wextra -Wshadow -m64 -DUSE_PTHREADS -Og -g3 -fexperimental-new-pass-manager -DIS_64BIT -msse -msse3 -mpopcnt -DUSE_POPCNT -DUSE_AVX2 -mavx2 -DUSE_SSE41 -msse4.1 -DUSE_SSSE3 -mssse3 -DUSE_SSE2 -msse2 -DUSE_PEXT -mbmi2) # -flto)
#set(STOCK_LINK_FLAGS -latomic -m64 -lpthread  -Wall -Wcast-qual -std=c++17  -pedantic -Wextra -Wshadow -m64 -DUSE_PTHREADS -Og -g3 -fexperimental-new-pass-manager -DIS_64BIT -msse -msse3 -mpopcnt -DUSE_POPCNT -DUSE_AVX2 -mavx2 -DUSE_SSE41 -msse4.1 -DUSE_SSSE3 -mssse3 -DUSE_SSE2 -msse2 -DUSE_PEXT -mbmi2) # -flto)

target_compile_options(NNStockChess PUBLIC ${STOCK_COMP_FLAGS})
target_link_options(NNStockChess PUBLIC ${STOCK_LINK_FLAGS})

add_compile_definitions(DISABLE_PV_OUTP)
