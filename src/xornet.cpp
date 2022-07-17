//
// Created by grant on 7/15/22.
//

#include "nn_linalg.hpp"

#include <iostream>

void xornet() {
    Vec<2> input;
    Layer<2, 2> hidden1;
    Layer<2, 1> output;

    hidden1.input = &input;
    output.input = &hidden1.activation;

    hidden1.weights.randomize();
    hidden1.biases.randomize();
    output.weights.randomize();
    output.biases.randomize();

    for (int iter = 0; iter < 4096; iter++) {
        hidden1.learn_rate = output.learn_rate = 1 - (iter / 4096.0);
        NumericT err = 0;
        for (int a = 0; a < 2; a++) {
            for (int b = 0; b < 2; b++) {
                if (iter < 4000 && rng(0) < 0.25) {
                    err += 1;
                    continue;
                }

                auto ddes = static_cast<NumericT>(a ^ b);
                auto desired = Vec<1>{{{ddes}}};

                input = {{{(NumericT) a}, {(NumericT) b}}};
                hidden1.forward();
                output.forward();

                hidden1.backward(output.backward(output.init_backwards(desired)));

                err += std::pow(output.activation[0][0] - ddes, 2);

                if (iter > 4040) {
                    std::cout << a << " ^ " << b << " = " << output.activation[0][0] << '\n';
                }
            }
        }

        std::cout << "Error: " << err / 4 << '\n';

        if (err < std::numeric_limits<NumericT>::epsilon())
            break;

        if (rng(0) > 0.125)
            output.apply_backprop();

        if (rng(0) > 0.125)
            hidden1.apply_backprop();
    }

    std::cout << output.weights.to_string() << '\n';
    std::cout << output.biases.to_string() << '\n';

    std::cout << hidden1.weights.to_string() << '\n';
    std::cout << hidden1.biases.to_string() << '\n';
}