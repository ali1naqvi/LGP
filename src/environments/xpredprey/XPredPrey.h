/*
 * This file belong to https://github.com/snolfi/evorobotpy
 * Author: Stefano Nolfi, stefano.nolfi@istc.cnr.it

 * Native C++ interface for the xpredprey environment.

 * The original Python/Cython wrapper has been removed; this environment is
 * built directly by the repository's CMake project.
 */

#ifndef LGP_XPREDPREY_H
#define LGP_XPREDPREY_H

#include "Utilities.h"

#include <string>


class XPredPrey
{

public:
	// Void constructor
	explicit XPredPrey(int max_steps = 2000,
                      const std::string& asset_directory = ".");
	// Destructor
	~XPredPrey();
	// Set the seed
	void seed(int s);
	// Reset episode
	void reset();
	// Perform a step
	double step();
	// Close
	void close();
	// Render the robot and the environment
	void render();
	// Copy the observations
	void copyObs(float* observation);
	// Copy the action
	void copyAct(float* action);
	// Copy the termination flag
	void copyDone(int* done);
	// Copy the pointer to the vector of objects to be displayed
	void copyDobj(double* objs);
	// Check whether the episode terminated
	int isDone();
	bool isCaptured() const;
    // number of inputs
    int ninputs;
    // number of outputs
    int noutputs;

private:
	// create the environment
    void initEnvironment();
    // compute the state of the observation
	void getObs();
	// Random generator
	RandomGenerator* rng = nullptr;
	bool captured_ = false;

};

#endif
