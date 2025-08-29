#pragma once
#ifndef COMPUTATION_CONTEXT_HPP
#define COMPUTATION_CONTEXT_HPP

#include <Eigen/Dense>
#include "Activation.hpp"


// Abstract base class defining the computation interface for neural network layers
class ComputationContext {
public:
    // Virtual destructor to ensure proper cleanup in derived classes
    virtual ~ComputationContext() = default;

    virtual void fuckingHell() = 0;
}; 


#endif // COMPUTATION_CONTEXT_HPP
