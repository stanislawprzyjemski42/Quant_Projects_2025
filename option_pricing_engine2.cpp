#include <iostream>
#include <cmath>
#include <iomanip>
#include <string>
#include <limits>

/**
 * @brief A humble implementation of the Black-Scholes model.
 * Focused on practicing Object-Oriented Programming (OOP) and 
 * safe user-input handling in C++.
 */
class EuropeanOption {
private:
    double S;     // Asset Price
    double K;     // Strike Price
    double T;     // Time to Maturity (Years)
    double r;     // Risk-free Rate
    double sigma; // Volatility

    // Cumulative Normal Distribution Function (Standard Approximation)
    double cumulativeNormal(double x) const {
        return 0.5 * erfc(-x * M_SQRT1_2);
    }

public:
    // Constructor
    EuropeanOption(double s, double k, double t, double rate, double vol)
        : S(s), K(k), T(t), r(rate), sigma(vol) {}

    // Method to calculate Call price
    double calculateCall() const {
        double d1 = (log(S / K) + (r + 0.5 * pow(sigma, 2)) * T) / (sigma * sqrt(T));
        double d2 = d1 - sigma * sqrt(T);
        return S * cumulativeNormal(d1) - K * exp(-r * T) * cumulativeNormal(d2);
    }

    // Method to calculate Put price
    double calculatePut() const {
        double d1 = (log(S / K) + (r + 0.5 * pow(sigma, 2)) * T) / (sigma * sqrt(T));
        double d2 = d1 - sigma * sqrt(T);
        return K * exp(-r * T) * cumulativeNormal(-d2) - S * cumulativeNormal(-d1);
    }
};

/**
 * @brief Utility to ensure inputs are valid numbers.
 * Demonstrates basic defensive programming for CS applications.
 */
double validateInput(const std::string& label) {
    double value;
    while (true) {
        std::cout << label;
        if (std::cin >> value && value >= 0) {
            return value;
        } else {
            std::cout << "  [Note] Please enter a valid positive number." << std::endl;
            std::cin.clear(); 
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
    }
}

int main() {
    std::cout << "--- Option Pricing Exploration (2025) ---" << std::endl;
    std::cout << "This tool is a student project for learning C++ logic." << std::endl << std::endl;

    // Collect validated inputs
    double s     = validateInput("Current Stock Price: ");
    double k     = validateInput("Strike Price:        ");
    double t     = validateInput("Years to Maturity:   ");
    double r     = validateInput("Risk-free Rate:      ");
    double sigma = validateInput("Volatility (sigma):  ");

    // Initialize the model
    EuropeanOption userOption(s, k, t, r, sigma);

    // Output formatted results
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n--- Calculated Theoretical Values ---" << std::endl;
    std::cout << "Call Price: " << userOption.calculateCall() << std::endl;
    std::cout << "Put Price:  " << userOption.calculatePut() << std::endl;
    std::cout << "-------------------------------------" << std::endl;

    return 0;
}
