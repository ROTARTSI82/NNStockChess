#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>

template <typename T>
struct SigmoidActivation {
    static inline constexpr T activate(T in) {
        return 1 / (1 + std::exp((double) -in));
    }

    static inline constexpr T activate_prime(T in) {
        return std::exp((double) -in) / std::pow(1 + std::exp((double) -in), 2);
    }
};

template <typename T>
struct IdentityActivation {
    static inline constexpr T activate(T in) {
        return in;
    }

    static inline constexpr T activate_prime(T) {
        return 1;
    }
};

template <typename T>
inline constexpr T ReLU(T in) {
    return std::max(0.0, in);
}

template <typename T>
inline constexpr T ReLU_prime(T in) {
    return in < 0 ? 0 : 1;
}

template <typename T>
inline constexpr T GELU(T x) {
    return x * (tanh(sqrt(2 / std::numbers::pi) * (x + 0.044715 * x * x * x))) / 2.0;
}

template <typename T>
inline constexpr T GELU_prime(T x) {
    return x * (1.0 + 0.134145 * x * x) * std::pow(1.0 / cosh(sqrt(2.0 / std::numbers::pi) * (x + 0.044715 * x * x * x)), 2.0) / sqrt(2 * std::numbers::pi) + 0.5 * (1.0 + tanh(sqrt(2.0 / std::numbers::pi) * (x + 0.044715 * x * x * x)));
}

inline double rng(double lo = -1, double hi = 1) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<double> dist(lo, hi);
    return dist(gen);
}

template <typename T, int R, int C>
class Matrix {
    static_assert(R > 0);
    static_assert(C > 0);

public:
    typedef T type;
    typedef Matrix<T,R,C> self;
    constexpr static int rows = R;
    constexpr static int cols = C;

    T dat[R][C];

    // constexpr inline Matrix() = default;

    constexpr inline T *operator[](std::size_t r) {
        return dat[r];
    }

    constexpr inline T &operator()(std::size_t r, std::size_t c) {
        return dat[r][c];
    }

    constexpr inline const T *operator[](std::size_t r) const {
        return &dat[r][0];
    }

    constexpr inline const T &operator()(std::size_t r, std::size_t c) const {
        return dat[r][c];
    }

    constexpr inline self operator+(const self &rhs) const {
        return self{*this} += rhs;
    }

    constexpr inline self &operator+=(const self &rhs) {
        for (auto r = 0; r < rows; r++)
            for (auto c = 0; c < cols; c++)
                dat[r][c] += rhs.dat[r][c];
        return *this;
    }

    constexpr inline self operator*(T rhs) const {
        return self{*this} *= rhs;
    }

    constexpr inline self &operator/=(T rhs) {
        for (auto r = 0; r < rows; r++)
            for (auto c = 0; c< cols; c++)
                dat[r][c] /= rhs;
        return *this;
    }

    constexpr inline self operator/(T rhs) const {
        return self{*this} /= rhs;
    }

    constexpr inline self &operator*=(T rhs) {
        for (auto r = 0; r < rows; r++)
            for (auto c = 0; c < cols; c++)
                dat[r][c] *= rhs;
        return *this;
    }

    template <int O>
    constexpr inline Matrix<T, R, O> operator*(const Matrix<T, C, O> &rhs) const {
        Matrix<T, R, O> x;
        for (auto r = 0; r < R; r++)
            for (auto c = 0; c < O; c++)
                x.dat[r][c] = 0;

        for (auto r = 0; r < R; r++)
            for (auto c = 0; c < O; c++)
                for (auto mr = 0; mr < C; mr++)
                    x.dat[r][c] += rhs.dat[mr][c] * dat[r][mr];
        return x;
    }

    constexpr inline Matrix<T, C, R> transpose() const {
        Matrix<T, C, R> x;
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                x[c][r] = dat[r][c];
        return x;
    }

    [[nodiscard]] std::string to_string() const {
        std::string ret;
        for (auto r = 0; r < rows; r++) {
            for (auto c = 0; c < cols; c++)
                ret += std::to_string(dat[r][c]) + "\t";
            ret += "\n";
        }

        return ret;
    }

    constexpr static inline self zeroed() {
        self ret;
        for (auto r = 0; r < rows; r++)
            for (auto c = 0; c < cols; c++)
                ret.dat[r][c] = 0;
        return ret;
    }

    constexpr inline self &func_map_ip(const std::function<T(T)> &func) {
        for (auto r = 0; r < rows; r++)
            for (auto c = 0; c < cols; c++)
                dat[r][c] = func(dat[r][c]);
        return *this;
    }

    constexpr inline self func_map(const std::function<T(T)> &func) {
        return self{*this}.func_map_ip(func);
    }

    void randomize(T lo = -1, T hi = 1) {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_real_distribution<T> dist(lo, hi);

        for (auto r = 0; r < rows; r++)
            for (auto c = 0; c < cols; c++)
                dat[r][c] = dist(gen);
    }
};

using NumericT = float;

template <int R, int C>
using Mat = Matrix<NumericT, R, C>;

template <int D>
using Vec = Mat<D, 1>;


template <int InpNo, int OutNo, typename Activation = SigmoidActivation<NumericT>>
class Layer {
public:
    Mat<OutNo, InpNo> weights;
    Vec<OutNo> biases;

    Mat<OutNo, InpNo> weight_step_acc = Mat<OutNo, InpNo>::zeroed();
    Vec<OutNo> bias_step_acc = Vec<OutNo>::zeroed();
    NumericT num_backprops = 0;
    NumericT learn_rate = 0.1, bias_learn = 1;

    Vec<OutNo> z_act;
    Vec<OutNo> activation;
    Vec<InpNo> *input;

    inline constexpr void apply_backprop() {
        if (num_backprops <= 0) return;

        biases += (bias_step_acc *= (bias_learn * learn_rate / num_backprops));
        weights += (weight_step_acc *= (learn_rate / num_backprops));

        num_backprops = 0;
        weight_step_acc = Mat<OutNo, InpNo>::zeroed();
        bias_step_acc = Vec<OutNo>::zeroed();
    }


    inline constexpr Vec<InpNo> backward(const Vec<OutNo> &dCost_dAct) {
        // dAct is the derivative of the cost with respect to our activation
        num_backprops++;

        // derivative of the cost with respect to the activation of the previous layer
        Vec<InpNo> dCost_dActPrev = Vec<InpNo>::zeroed();

        for (int i = 0; i < OutNo; i++) {
            // derivative of activation with respect to Z
            const NumericT dAct_dZ = Activation::activate_prime(z_act[i][0]);

            // derivative of cost with respect to bias;
            const NumericT dCost_dZ = dCost_dAct[i][0] * dAct_dZ;

            // no term for derivative of Z with respect to bias since dZ_dBias = 1
            const NumericT dCost_dBias = dCost_dZ;
            bias_step_acc[i][0] += dCost_dBias;

            for (int j = 0; j < InpNo; j++) {
                const NumericT dZ_dWeight = input->dat[j][0]; // equal to the activation of the previous layer!
                const NumericT dCost_dWeight = dCost_dZ * dZ_dWeight;
                weight_step_acc[i][j] += dCost_dWeight;

                const NumericT dZ_dActPrev = weights[i][j]; // equal to the weight connecting that neuron to us
                dCost_dActPrev[j][0] += dCost_dZ * dZ_dActPrev;
            }
        }

        return dCost_dActPrev;
    }

    inline constexpr void forward() {
        z_act = weights * (*input) + biases;
        for (auto i = 0; i < OutNo; i++)
            activation[i][0] = Activation::activate(z_act[i][0]);
    }

    constexpr Vec<OutNo> init_backwards(const Vec<OutNo> &desired) {
        Vec<OutNo> dCdA;
        for (int i = 0; i < OutNo; i++)
            dCdA[i][0] = 2 * (desired[i][0] - activation[i][0]);
        return dCdA;
    }

    void randomize(NumericT lo = -1, NumericT hi = 1) {
        weights.randomize(lo, hi);
        biases.randomize(lo, hi);
    }

    void load(std::ifstream &stream) {
        stream.read((char *) &weights.dat[0][0], (InpNo * OutNo + OutNo) * sizeof(NumericT));
    }

    void save(std::ofstream &stream) {
        stream.write((char *) &weights.dat[0][0], (InpNo * OutNo + OutNo) * sizeof(NumericT));
    }
};




