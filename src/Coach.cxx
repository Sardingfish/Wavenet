#include "Wavenet/Coach.h"
#include "Wavenet/Generators.h" /* To determine whether generator has natural epochs. */

namespace wavenet {
    
void Coach::setBasedir (const std::string& basedir) {
    m_basedir = basedir;
    if (strcmp(&m_basedir.back(), "/") == 0) { m_basedir.append("/"); }
    return;
}

void Coach::setNumEvents (const int& numEvents) {
    if (numEvents < 0 and numEvents != -1) {
        WARNING("Input number of events (%d) not supported.", numEvents);
        return;
    }
    m_numEvents = numEvents;
    return;
}

void Coach::setNumCoeffs (const unsigned& numCoeffs) {
    if (!isRadix2(numCoeffs)) {
        WARNING("Input number of coefficients (%d) is not radix 2.", numCoeffs);
        return;
    }
    m_numCoeffs = numCoeffs;
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
    m_targetPrecision = targetPrecision;
    return;
}

bool Coach::run () {
    
    // Perform checks.
    if (!m_wavenet) {
        ERROR("WaveletML object not set.Exiting.");
        return false;
    }

    if (!m_generator) {
        ERROR("Input generator not set. Exiting.");
        return false;
    }

    if (!m_generator->initialised()) {
        ERROR("Generator was not properly initialised. Did you remember to specify a valid shape? Exiting.");
        return false;
    }

    if (!m_name.size()) {
        ERROR("Coach name not set. Exiting.");
        return false;
    }
    
    if (m_numEvents < 0) {
        if ((dynamic_cast<NeedleGenerator*>  (m_generator) != nullptr ||
             dynamic_cast<UniformGenerator*> (m_generator) != nullptr ||
             dynamic_cast<GaussianGenerator*>(m_generator) != nullptr) && 
            !(useAdaptiveLearningRate() && targetPrecision() > 0.)) {
            WARNING("The number of events is set to %d while using", m_numEvents);
            WARNING(".. a generator with no natural epochs and with");
            WARNING(".. no target precision set. Etiher choose a ");
            WARNING(".. different generator or use");
            WARNING(".. 'Coach::setUseAdaptiveLearningRate()' and");
            WARNING("'Coach::setTargetPrecision(someValue)'.");
            WARNING("Exiting.");
            return false;
        }
    }
    
    INFO("Start training, using coach '%s'.", m_name.c_str());
    
    // Save base snapshot of initial condition, so as to be able to restore same 
    // configuration for each intitialisation (in particular, to roll back 
    //changes made by adaptive learning methods.)
    Snapshot baseSnap (outdir() + "snapshots/.tmp.snap");
    m_wavenet->save(baseSnap);
    
    // Define number of trailing steps, for use with adaptive learning rate.
    const unsigned useLastN = 10;
    // Definition bare, specified regularsation constant, for use with simulated
    // annealing.
    const double lambdaBare = m_wavenet->lambda(); /* @TODO: Used? */
    
    // Define snapshot object, for saving the final configuration for each 
    // initialisation.
    Snapshot snap (outdir() + "snapshots/" + m_name + ".%06u.snap", 0);

    // Loop initialisations.
    for (unsigned init = 0; init < m_numInits; init++) {

        // Print progress.
        if (m_printLevel > 0) {
            INFO("Initialisation %d/%d", init + 1, m_numInits);
        }

        // Load base snapshot.
        m_wavenet->load(baseSnap);
        m_wavenet->clear();

        // Generate initial coefficient configuration as random point on unit 
        // N-sphere. In this way we immediately fullfill one out of the (at 
        // most) four (non-trivial) conditions on the filter coefficients.
        m_wavenet->setFilter( PointOnNSphere(m_numCoeffs) );
        
        // Definitions for adaptive learning.
        bool done = false; // Whether the training is done, i.e. whether to 
                           // break training early
        unsigned tail = 0; // The number of updates since beginning of training 
                           // or last update of the learning rate, whichever is 
                           // latest.
        unsigned currentCostLogSize  = 0; // Number of entries in cost log, now 
        unsigned previousCostLogSize = 0; // and at previous step in the loop. 
                                          // Used to determine whether a batch
                                          // update occurred.
        
        // Loop epochs.
        for (unsigned epoch = 0; epoch < m_numEpochs; epoch++) {

            // Reset (re-open) generator.
            m_generator->reset();

            // Print progress.
            if (m_printLevel > 1) {
                INFO("  Epoch %d/%d", epoch + 1, m_numEpochs);
            }

            // Reset lambda (if using simulated annealing).
            /* Don't reset for each epoch. Should only be reset for each initialisation.
            if (useSimulatedAnnealing()) {
                m_wavenet->setLambda(lambdaBare);
            }
            */

            // Loop events.
            int event = 0;
            int eventPrint = 100;
            do {
                // Print progress.
                if (m_printLevel > 2 && (event + 1) % eventPrint == 0) {
                    if (m_numEvents == -1) { INFO("    Event %d/-",  event + 1); }
                    else                   { INFO("    Event %d/%d", event + 1, m_numEvents); }
                    if ((event + 1) == 10 * eventPrint) { eventPrint *= 10; }
                }

                // Simulated annealing.
                if (useSimulatedAnnealing()) {
                     const double f = (event + epoch * numEvents())/float(numEvents() * numEpochs());
                     const double effectiveLambda = lambdaBare * f / sq(2 - f);
                     m_wavenet->setLambda(effectiveLambda);
                }

                // Main training call.
                m_wavenet->train( m_generator->next() );

                // Adaptive learning rate.
                if (useAdaptiveLearningRate()) {

                    // Determine whether a batch upate took place, by checking 
                    // whether the size of the cost log changed.
                    previousCostLogSize = currentCostLogSize;
                    currentCostLogSize  = m_wavenet->costLog().size();
                    bool changed = (currentCostLogSize != previousCostLogSize);
                    

                    // If it changed and the tail (number of updates since last 
                    // learning rate update) is sufficiently large, initiate
                    // adaptation.
                    if (changed && ++tail > useLastN) {
                        
                        const unsigned filterLogSize = m_wavenet->filterLog().size();

                        // Compute the (vector) size of the last N steps in the 
                        // SGD, as well as the mean (scalar) size of these.
                        std::vector< arma::Col<double> > lastNsteps(useLastN);
                        double meanStepSize  = 0;
                        for (unsigned i = 0; i < useLastN; i++) {
                            lastNsteps.at(i) = m_wavenet->filterLog().at(filterLogSize - useLastN + i) - m_wavenet->filterLog().at(filterLogSize - useLastN + i - 1);
                            meanStepSize += arma::norm(lastNsteps.at(i));
                        }
                        meanStepSize /= float(useLastN);
                        
                        // Compute the total (vector) size of the last N steps 
                        // in the SGD combined, as well as the (scalar) size.
                        arma::Col<double> totalStep = m_wavenet->filterLog().at(filterLogSize - 1) - m_wavenet->filterLog().at(filterLogSize - 1 - useLastN);
                        double totalStepSize = arma::norm(totalStep);
                       
                        // Check whether we have reached target precision or 
                        // whether to perform adaptive learning rate update. 
                        if (targetPrecision() != -1 && meanStepSize < targetPrecision() && !useSimulatedAnnealing()) {
                            INFO("[Adaptive learning rate] The mean step size over the last %d updates (%f)", useLastN, meanStepSize);
                            INFO("[Adaptive learning rate] is smaller than the target precision (%f). Done.", targetPrecision());
                            done = true;
                        } else if (totalStepSize < meanStepSize) {
                            INFO("[Adaptive learning rate] Total step size (%f) is smaller than mean step size (%f).", totalStepSize, meanStepSize);
                            INFO("[Adaptive learning rate]   Increasing batch size from %d to %d.", m_wavenet->batchSize(), 2 * m_wavenet->batchSize());
                            INFO("[Adaptive learning rate]   Reducing learning rate (alpha) from %f to %f.", m_wavenet->alpha(), (1./2.) * m_wavenet->alpha() * (totalStepSize/meanStepSize));
                            m_wavenet->setBatchSize(  2     * m_wavenet->batchSize() );
                            m_wavenet->setAlpha    ( (1./2.) * m_wavenet->alpha() * (totalStepSize/meanStepSize));
                            tail = 0;
                        }
                        
                        
                    }
                    
                } 

                // Increment event number. (Only level not in a for-loop, since
                // the number of events may be unspecified, i.e. be -1.)
                ++event;

                // If the generator is not in a good condition, break.
                if (!m_generator->good()) { break; }

            } while (!done && (event < m_numEvents || m_numEvents < 0 ));
            
            if (done) { break; }
        }
        
        // Saving snapshot to file.
        m_wavenet->save(snap++);
    }
    
    // Writing setup to run-specific README file.
    INFO("Writing run configuration to '%s'.", (outdir() + "README").c_str());
    std::ofstream outFileStream (outdir() + "README");
    
    outFileStream << "m_numEvents: " << m_numEvents << "\n";
    outFileStream << "m_numEpochs: " << m_numEpochs << "\n";
    outFileStream << "m_numInits: "  << m_numInits  << "\n";
    outFileStream << "m_numCoeffs: " << m_numCoeffs << "\n";
    
    outFileStream.close();

    // Clean up, byremove the last entry in the cost log, which isn't properly 
    // scaled to batch size since the batch queue hasn't been flushed, and 
    // therefore might bias result.
    m_wavenet->costLog().pop_back(); 
    
    return true;   
}

} // namespace
