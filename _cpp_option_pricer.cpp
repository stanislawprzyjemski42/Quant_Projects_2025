#include <iostream>
#include <cmath>
#include <algorithm>

/**
 * A humble implementation of the Black-Scholes formula.
 * This project was created to explore numerical approximations 
 * of the Cumulative Normal Distribution and their application in finance.
 */

// Standard Normal Cumulative Distribution Function (CDF)
// Using the Hart's approximation for simplicity and efficiency
double normalCDF(double x) {
    return 0.5 * erfc(-x * M_SQRT1_2);
}

// Core Pricing Function
void calculateOptionPrice(double S, double K, double T, double r, double sigma) {
    
    // Calculate d1 and d2 components
    double d1 = (log(S / K) + (r + 0.5 * pow(sigma, 2)) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);

    // Calculate Call and Put prices
    double callPrice = S * normalCDF(d1) - K * exp(-r * T) * normalCDF(d2);
    double putPrice = K * exp(-r * T) * normalCDF(-d2) - S * normalCDF(-d1);

    std::cout << "--- Option Pricing Results ---" << std::endl;
    std::cout << "Underlying Price (S): " << S << std::endl;
    std::cout << "Strike Price (K):     " << K << std::endl;
    std::cout << "Time to Maturity (T): " << T << " years" << std::endl;
    std::cout << "Risk-Free Rate (r):   " << r << std::endl;
    std::cout << "Volatility (sigma):   " << sigma << std::endl;
    std::cout << "------------------------------" << std::endl;
    std::cout << "Call Option Price:    " << callPrice << std::endl;
    std::cout << "Put Option Price:     " << putPrice << std::endl;
}

int main() {
    // Example parameters:
    // S = 100 (Current Price), K = 100 (Strike), T = 1 (1 Year)
    // r = 0.05 (5% Rate), sigma = 0.2 (20% Volatility)
    
    double S = 100.0, K = 100.0, T = 1.0, r = 0.05, sigma = 0.2;

    try {
        calculateOptionPrice(S, K, T, r, sigma);
    } catch (...) {
        std::cerr << "An error occurred during calculation." << std::endl;
    }

    return 0;
}
