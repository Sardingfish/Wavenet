#include "Wavenet/Coach.h"
#include "Wavenet/Generators.h" /* To determine whether generator has natural epochs. */

namespace wavenet {
    
 // Set method(s).
// -------------------------------------------------------------------

void Coach::setBasedir (const std::string& basedir) {
    _basedir = basedir;
    if (strcmp(&_basedir.back(), "/") == 0) { _basedir.append("/"); }
    return;
}

void Coach::setNumEvents (const int& numEvents) {
    if (numEvents < 0 and numEvents != -1) {
        WARNING("Input number of events (%d) not supported.", numEvents);
        return;
    }
    _numEvents = numEvents;
    return;
}

void Coach::setNumCoeffs (const unsigned& numCoeffs) {
    if (!isRadix2(numCoeffs)) {
        WARNING("Input number of coefficients (%d) is not radix 2.", numCoeffs);
        return;
    }
    _numCoeffs = numCoeffs;
    return;
}

void Coach::setTargetPrecision (const double& targetPrecision) {
    if (targetPrecision != -1 && targetPrecision <= 0) {
        WARNING("Requested target precision (%f) is no good. Exiting.", targetPrecision);
        return;
    }
    if (!useAdaptiveLearningRate()) {
        WARNING("Target precision is to be used in conjuction with 'adaptive learning rate'.");
        WARNING("Remember to set it using 'Coach::setUseAdaptiveLearningRate()'. Going to continue,");
        WARNING("but the set value of target precision won't have any effect on its own.");

    }
    _targetPrecision = targetPrecision;
    return;
}


 // High-level management info.
// -------------------------------------------------------------------

bool Coach::run () {
    DEBUG("Entering.");

    // Performing checks.
    if (!_wavenet) {
        ERROR("WaveletML object not set.Exiting.");
        return false;
    }

    if (!_generator) {
        ERROR("Input generator not set. Exiting.");
        return false;
    }

    if (!_name.size()) {
        ERROR("Coach name not set. Exiting.");
        return false;
    }
    
    if (_numEvents < 0) {
        if ((dynamic_cast<NeedleGenerator*>  (_generator) != nullptr ||
             dynamic_cast<UniformGenerator*> (_generator) != nullptr ||
             dynamic_cast<GaussianGenerator*>(_generator) != nullptr) && 
            !(useAdaptiveLearningRate() && targetPrecision() > 0.)) {
            WARNING("The number of events is set to %d while using", _numEvents);
            WARNING(".. a generator with no natural epochs and with");
            WARNING(".. no target precision set. Etiher choose a ");
            WARNING(".. different generator or use");
            WARNING(".. 'Coach::setUseAdaptiveLearningRate()' and");
            WARNING("'Coach::setTargetPrecision(someValue)'.");
            WARNING("Exiting.");
            return false;
        }
    }
    
    
    // Run.
    INFO("Start training, using coach '%s'.", _name.c_str());
    
    // -- Save base snapshot, to step back from adaptive learning.
    _wavenet->save(outdir() + "snapshots/.tmp.snap");
    
    // -- Definitions for daptive learning rate.
    const unsigned int useLastN = 10;
    
    // Loop initialisations.
    for (unsigned init = 0; init < _numInits; init++) {

        // -- Print.
        if (_printLevel > 0) {
            INFO("Initialisation %d/%d", init + 1, _numInits);
        }

        // -- Load starting snapshot.
        _wavenet->load(outdir() + "snapshots/.tmp.snap");
        _wavenet->clear();

        // -- Generate initial coefficient configuration on random point on unit N-sphere.
        _wavenet->setFilter( PointOnNSphere(_numCoeffs) );
        
        // -- Definitions for simulated annealing.
        const double lambdaBare = _wavenet->lambda();
        
        // -- Definitions for adaptive learning.
        bool done = false;
        unsigned int tail = 0;
        unsigned int currentCostLogSize  = 0;
        unsigned int previousCostLogSize = 0;
        
        // Loop epochs.
        for (unsigned epoch = 0; epoch < _numEpochs; epoch++) {

            // -- Reset (re-open) generator.
            _generator->reset();

            // -- Print.
            if (_printLevel > 1) {
                INFO("  Epoch %d/%d", epoch + 1, _numEpochs);
            }

            // -- Reset lambda (if using simulated annealing).
            if (useSimulatedAnnealing()) {
                _wavenet->setLambda(lambdaBare);
            }

            // Loop events.
            int event = 0;
            int eventPrint = 100;
            do {
                // -- Print.
                if (_printLevel > 2 && (event + 1) % eventPrint == 0) {
                    if (_numEvents == -1) { INFO("    Event %d/-",  event + 1); }
                    else                  { INFO("    Event %d/%d", event + 1, _numEvents); }
                    if ((event + 1) == 10 * eventPrint) { eventPrint *= 10; }
                }

                // -- Simulated annealing.
                if (useSimulatedAnnealing()) {
                    /**
                     * As ievent -> 0,       lambda -> 0
                     * As ievent -> Nevents, lambda -> lambdaBare
                     * Effective weight: f / (2 - f)
                     *  -- f = 0:   0   / (2 - 0)^2   = 0   / 4    = 0
                     *  -- f = 0.5: 0.5 / (2 - 0.5)^2 = 0.5 / 2.25 = 0.22...
                     *  -- f = 1:   1   / (2 - 1)^2   = 1   / 1    = 1
                    **/

                     const double f = event/float(numEvents());
                     const double effectiveLambda = lambdaBare * f / sq(2 - f);
                     _wavenet->setLambda(effectiveLambda);
                }
                
                // -- Check initialisation.
                if (!_generator->initialised()) {
                    ERROR("Generator was not properly initialised. Did you remember to specify a valid shape? Exiting.");
                    return false;
                }

                // -- Main training call.
                _wavenet->batchTrain( _generator->next() );
                

                // -- Adaptive learning rate.
                if (useAdaptiveLearningRate()) {

                    // -- Determine whether a batch upate took place, by checking whether the size of the cost log changed.
                    previousCostLogSize = currentCostLogSize;
                    currentCostLogSize  = _wavenet->costLog().size();
                    bool changed = (currentCostLogSize != previousCostLogSize);
                    
                    // -- If it changed and the tail (number of updates since last learning rate adaptation) is sufficiently large, initiate adaptation.
                    if (changed && ++tail > useLastN) {
                        
                        const unsigned int filterLogSize = _wavenet->filterLog().size();

                        // Compute the (vector) size of the last N steps in the SGD, as well as the mean (scalar) size of these.
                        std::vector< arma::Col<double> > lastNsteps(useLastN);
                        double meanStepSize  = 0;
                        for (unsigned i = 0; i < useLastN; i++) {
                            lastNsteps.at(i) = _wavenet->filterLog().at(filterLogSize - useLastN + i) - _wavenet->filterLog().at(filterLogSize - useLastN + i - 1);
                            meanStepSize += arma::norm(lastNsteps.at(i));
                        }
                        meanStepSize /= float(useLastN);
                        
                        // Compute the total (vector) size of the last N steps in the SGD combined, as well as the (scalar) size.
                        arma::Col<double> totalStep = _wavenet->filterLog().at(filterLogSize - 1) - _wavenet->filterLog().at(filterLogSize - 1 - useLastN);
                        double totalStepSize = arma::norm(totalStep);
                       
                        // Check whether to perform adaptive learning rate update. 
                        // Check whether we have reached target precision.
                        if (targetPrecision() != -1 && meanStepSize < targetPrecision()) {
                            INFO("[Adaptive learning rate] The mean step size over the last %d updates (%f)", useLastN, meanStepSize);
                            INFO("[Adaptive learning rate] is smaller than the target precision (%f). Done.", targetPrecision());
                            done = true;
                        } else if (totalStepSize < meanStepSize) {
                            INFO("[Adaptive learning rate] Total step size (%f) is smaller than mean step size (%f).", totalStepSize, meanStepSize);
                            if (totalStepSize > 1e-07) {
                                INFO("[Adaptive learning rate]   Increasing batch size from %d to %d.", _wavenet->batchSize(), 2 * _wavenet->batchSize());
                                INFO("[Adaptive learning rate]   Reducing learning rate (alpha) from %f to %f.", _wavenet->alpha(), (1./2.) * _wavenet->alpha() * (totalStepSize/meanStepSize));
                                _wavenet->setBatchSize(  2     * _wavenet->batchSize() );
                                _wavenet->setAlpha    ( (1./2.) * _wavenet->alpha() * (totalStepSize/meanStepSize));
                                tail = 0;
                            } else if (!useSimulatedAnnealing()) {
                                // If using simulated annealing we disable early breaking, since then otherwise the minimisation would have been performed with a lower value of the regularisation factor than intended.
                                INFO("[Adaptive learning rate] Step size is smaller than 1e-07. Done.");
                                done = true;
                            }
                        }
                        
                        
                    }
                    
                } // adaptive learning rate
                

                // -- Increment.
                ++event;

            } while (!done && (_numEvents < 0 || (event < _numEvents && _generator->good())));
            
            if (done) { break; }
        }
        
        
        // Saving snapshot to file.
        char buff[100];
        snprintf(buff, sizeof(buff), "%s.%06u.snap", _name.c_str(), init);
        std::string filename = buff;
        _wavenet->save(outdir() + "snapshots/" + filename);
    }
    
    // Writing setup to run-specific README file.
    INFO("Writing run configuration to '%s'.", (outdir() + "README").c_str());
    std::ofstream outFileStream (outdir() + "README");
    
    outFileStream << "_numEvents: " << _numEvents << "\n";
    outFileStream << "_numEpochs: " << _numEpochs << "\n";
    outFileStream << "_numInits: "  << _numInits  << "\n";
    outFileStream << "_numCoeffs: " << _numCoeffs << "\n";
    
    outFileStream.close();
    
    // Clean up.
    _wavenet->costLog().pop_back(); // Remove the last entry in the cost log, which isn't properly scaled to batch size since the batch queue hasn't been flushed, and therefore might bias result.
    //_wavenet->clear();
    
    return true;
    
}

} // namespace
