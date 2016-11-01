#include "Wavenet/Coach.h"

 // Set method(s).
// -------------------------------------------------------------------

void Coach::setBasedir (const string& basedir) {
    _basedir = basedir;
    if (strcmp(&_basedir.back(), "/") == 0) { _basedir.append("/"); }
    return;
}

void Coach::setNevents (const int& Nevents) {
    if (Nevents < 0 and Nevents != -1) {
        WARNING("Input number of events (%d) not supported.", Nevents);
        return;
    }
    _Nevents = Nevents;
    return;
}

void Coach::setNcoeffs (const unsigned& Ncoeffs) {
    if (!isRadix2(Ncoeffs)) {
        WARNING("Input number of coefficients (%d) is not radix 2.", Ncoeffs);
        return;
    }
    _Ncoeffs = Ncoeffs;
    return;
}


 // High-level management info.
// -------------------------------------------------------------------

void Coach::run () {
    DEBUG("Entering.");

    // Performing checks.
    if (!_wavenet) {
        ERROR("WaveletML object not set.Exiting.");
        return;
    }

    if (!_generator) {
        ERROR("Input generator not set. Exiting.");
        return;
    }

    if (!_name.size()) {
        ERROR("Coach name not set. Exiting.");
        return;
    }
    
    // Run.
    INFO("Start training, using coach '%s'.", _name.c_str());
    
    // -- Save base snapshot, to step back from adaptive learning.
    _wavenet->save(_basedir + _name + "/snapshots/tmp.snap");
    
    // -- Stuff for adaptive learning.
    /*
    const unsigned Nstrats = 2;
    const unsigned maxAdaptiveSteps = 5;
     */
    const unsigned useLastN = 10;
    
    double rho = (_wavenet->lambda() > 0 ? 1. / (_wavenet->lambda()) : -1);
    for (unsigned init = 0; init < _Ninits; init++) {
        if (_printLevel > 0) {
            INFO("Initialisation %d/%d.", init + 1, _Ninits);
        }
        _wavenet->load(_basedir + _name + "/snapshots/tmp.snap");
        _wavenet->clear();
        arma_rng::set_seed_random();
        if (rho >= 0) {
            _wavenet->setFilter( PointOnNSphere(_Ncoeffs) );//, rho, true) );
        } else {
            _wavenet->setFilter( (arma::Col<double> ().randu(_Ncoeffs) * 2 - 1 )* 1.2 );
        }
        
        const double lambdaBare = _wavenet->lambda();
        
        // -- Stuff for adapetive learning.
        bool done = false;
        int tail = 0;
        /*
        int strat = 0;
        vector<unsigned> adaptiveSteps (Nstrats, 0);
        vector<double> lastNcosts (0);
        lastNcosts.clear();
         */
        
        unsigned currentCostLogSize  = 0;
        unsigned previousCostLogSize = 0;
        
        // * Perform training.
        for (unsigned epoch = 0; epoch < _Nepochs; epoch++) {
            _generator->reset();
            
            if (_printLevel > 1) {
                INFO("  Epoch %d/%d.", epoch + 1, _Nepochs);
            }

            int event = 0;
            int eventPrint = 100;
            do {
                if (_printLevel > 2 || (event + 1) % eventPrint == 0) {
                    INFO("    Event %d/%d.", event + 1, _Nevents);
                    if ((event + 1) == 10 * eventPrint) {
                        eventPrint *= 10;
                    }
                }
                /* ~ Simulated annealing.
                double frac = (event / float(_Nevents)) / float(_Nepochs - 1) + epoch / float(_Nepochs - 1); / * Current fraction of events processed * /
                _wavenet->setLambda( exp(log(lambdaBare) * (frac - 0.1) / frac) );
                */
                /*
                cout << "  frac:       " << frac << endl;
                cout << "  lambdaBare: " << lambdaBare << endl;
                cout << "  lambda:     " << exp(log(lambdaBare) * (frac - 0.1) / frac) << endl;
                */
                
                /* Main call. */
                _wavenet->batchTrain( _generator->next() );
                

                /* [BEGIN] Adaptive learning. */
                if (_useAdaptiveLearning) {
                    previousCostLogSize = currentCostLogSize;
                    currentCostLogSize  = _wavenet->costLog().size();
                    bool changed = (currentCostLogSize != previousCostLogSize);
                    
                    if (changed && ++tail > useLastN) {
                        
                        vector< Col<double> > lastNsteps(useLastN);
                        unsigned filterLogSize = _wavenet->filterLog().size();
                        for (unsigned i = 0; i < useLastN; i++) {
                            lastNsteps.at(i) = _wavenet->filterLog().at(filterLogSize - useLastN + i) - _wavenet->filterLog().at(filterLogSize - useLastN + i - 1);
                        }
                        
                        Col<double> totalStep (size(lastNsteps.at(0)), fill::zeros);
                        
                        double meanStepSize  = 0;
                        double totalStepSize = 0;
                        
                        for (unsigned i = 0; i < useLastN; i++) {
                            totalStep    +=            lastNsteps.at(i);
                            meanStepSize += arma::norm(lastNsteps.at(i));
                        }
                        
                        totalStepSize  = arma::norm(totalStep);
                        meanStepSize  /= (double) useLastN;
                        
                        if (totalStepSize < meanStepSize) {
                            cout << "[Adaptive learning]   Total step size (" << totalStepSize << ") is smaller than mean step size (" << meanStepSize << ")." << endl;
                            if (totalStepSize > 1e-07) {
                                cout << "[Adaptive learning]     Increasing batch size: " << _wavenet->batchSize() << " -> " << 2 * _wavenet->batchSize() << endl;
                                _wavenet->setBatchSize(  2     * _wavenet->batchSize() );
                                _wavenet->setAlpha    ( (2/3.) * _wavenet->alpha() * (totalStepSize/meanStepSize));
                                // _wavenet->setInertia  ( (2/3.) * _wavenet->inertia() );
                                if (meanStepSize < 1.0e-03) {
                                    _wavenet->setInertia ( 0.9 ); // << TEST
                                }
                                tail = 0;
                            } else {
                                cout << "[Adaptive learning] Done." << endl;
                                done = true;
                            }
                        }
                        
                        
                    }
                    
                    
                    /*
                    if (changed && currentCostLogSize == 20) {
                        //_wavenet->setInertia(0.99);
                        _wavenet->setBatchSize(10*_wavenet->batchSize());
                        //cout << "[Adaptive learning] Setting inertia to " << _wavenet->inertia() << endl;
                    }
                    */
                    /*
                    if (changed && currentCostLogSize % 200 == 0) {
                        _wavenet->setBatchSize( 2 * _wavenet->batchSize() );
                        //cout << "[Adaptive learning] Setting batch size to " << _wavenet->batchSize() << endl;
                    }
                    */
                    
                    /* ----------------------------------- */
                    /*
                    // Get slope, intercept, mean cost, and norm "errors" (NE).
                    double slope, intercept, meanCost, NE;
                    bool adapt = false;
                    
                    // If the number of costs in the log has changed...
                    if (changed && currentCostLogSize > 1) {
                        
                        // ... append the new cost to the buffer of last N costs...
                        double lastCost = _wavenet->costLog().at(previousCostLogSize - 1);
                        lastNcosts.push_back( lastCost );
                        
                        // ... and make sure that we only have at most N.
                        while (lastNcosts.size() > useLastNcosts) {
                            lastNcosts.erase(lastNcosts.begin());
                        }
                        
                        // If we do have exactly N costs in the buffer...
                        if (lastNcosts.size() == useLastNcosts) {
                            
                            // ... then perform a linear fit...
                            Col<double> X = linspace(1, useLastNcosts, useLastNcosts);
                            Col<double> Y (lastNcosts);
                            
                            slope = arma::as_scalar( arma::cov(X,Y) / arma::var(X) );
                            meanCost = mean(Y);
                            intercept = meanCost - slope * mean(X);
                            NE = arma::norm((X * slope + intercept) - Y);
                            
                            // ... and decide whether to change the learning parameters...

                            / **
                            * END GOAL:
                            *  - Data is correlated and decreasing very slowly
                            *
                            * Rules:
                            *  (1) If average slope is positive, decrease alpha.
                            *  (2) If consecutive costs are not correlated (coeff < 0.5), increase batch size.
                            *  (3) If consecutive costs are correlated, and slope is moderately negative ( [-0.01, -0.001]), increase alpha.
                            ** /
                            
                            cout << "  -- " << endl;
                            cout << "  NE/meanCost: " << NE/meanCost << endl;
                            cout << "  slope: " << slope << endl;
                            
                            if (slope > 0.0005) {
                                
                                cout << "   -> Reducing alpha by factor 2." << endl;
                                _wavenet->setAlpha( _wavenet->alpha() / 2.0 );
                                adapt |= true;
                                
                            } else if (NE/meanCost > 0.2) {
                                
                                cout << "   -> Increasing batch size by factor 2." << endl;
                                _wavenet->setBatchSize( _wavenet->batchSize() * 2.0 );
                                adapt |= true;
                                
                            } else if (slope > -0.01 && slope < -0.000001) {
                                
                                cout << "   -> Increasing alpha by factor 1.2." << endl;
                                _wavenet->setAlpha( _wavenet->alpha() * 1.2 );
                                adapt |= true;
                                
                            } else if (slope < 0 && slope > -0.000001) {
                                    
                                if (NE/meanCost < 0.1) {
                                    cout << "   -> Done!" << endl;
                                    done = true;
                                } else {
                                    cout << "   -> Increasing batch size by factor 2." << endl;
                                    _wavenet->setBatchSize( _wavenet->batchSize() * 2.0 );
                                    adapt |= true;
                                }
                                
                            } else {
                                
                                cout << "   -> Doing nothing." << endl;
                                
                            }
                            
                            // ... or whether to terminate the run as succesful.

                            //if (fabs(corr) > 0.8 && slope < 0 && slope > -0.0001) {
                            //    cout << "[Adaptive learning] Terminating succesfull run, with slope: " << slope << " (> -0.0001) and |corr|: " << fabs(corr) << " (> 0.8)." << endl;
                            //    done = true;
                            //}
                            
                        }
                        
                        if (adapt) { lastNcosts.clear(); }
                        
                    } // end: if cost log length has changed.
                    */
                    /* -----------------------------------
                    / * [BEGIN] Determine whether to adapt. * /
                    bool adapt = false;
                    double slope, corr;
                    
                    previousCostLogSize = currentCostLogSize;
                    currentCostLogSize  = _wavenet->costLog().size();
                    if (currentCostLogSize > previousCostLogSize && currectCostLogSize > 1) {
                        
                        double lastCost = _wavenet->costLog().at(previousCostLogSize - 1);
                        lastNcosts.push_back( lastCost );
                        
                        while (lastNcosts.size() > useLastNcosts) {
                            lastNcosts.erase(lastNcosts.begin());
                        }
                        if (lastNcosts.size() == useLastNcosts) {
                            Col<double> X = linspace(1, useLastNcosts, useLastNcosts);
                            Col<double> Y (lastNcosts);
                            
                            slope = arma::as_scalar( arma::cov(X,Y) / arma::var(X) );
                            corr  = arma::as_scalar( arma::cov(X,Y) / (arma::stddev(X) * arma::stddev(Y)) );
                            
                            / **
                             * END GOAL:
                             *  - Data is correlated and decreasing very slowly
                             *
                             * Rules:
                             *  (1) If average slope is positive, decrease alpha.
                             *  (2) If consecutive costs are not correlated (coeff < 0.5), increase batch size.
                             *  (3) If consecutive costs are correlated, and slope is moderately negative ( [-0.01, -0.001]), increase alpha.
                             ** /
                            if ((slope > -0.01  && slope < -0.001) || slope > 0 || fabs(corr) < 0.5) { // -0.001, 0.8
                                if (fabs(corr) < 0.5) {
                                    adapt = true;
                                    strat = 0;
                                }
                                / *
                                 if (slope > 0) {
                                 //strat = -1;
                                 } else if (fabs(corr) < 0.5) { // 0.8
                                 strat = 0;
                                 } else if (slope < -0.001){
                                 //strat =  1;
                                 }
                                 * /
                            }
                            
                            // Terminate succesful run.
                            / *
                             if (fabs(corr) > 0.8 && slope < 0 && slope > -0.0001) {
                             cout << "[Adaptive learning] Terminating succesfull run, with slope: " << slope << " (> -0.0001) and |corr|: " << fabs(corr) << " (> 0.8)." << endl;
                             done = true;
                             }
                             * /
                            
                            
                        }
                    } / * [END] Determine whether to adapt. * /
                    
                    
                    / * [BEGIN] If do adapt. * /
                    if (adapt) {
                        
                        // ----------------------------------------------------------------
                        // Adaptive learning.
                        
                        / **
                         * To do: Make different logics for when to update alpha (small slope, small variations), batchSize (large variations)
                         *
                         ** /
                        
                        if (adaptiveSteps.at(abs(strat)) > maxAdaptiveSteps) {
                            cout << "[Adaptive learning] Have reaching maximum number of adaptive steps (" << maxAdaptiveSteps << "). Exiting." << endl;
                            done = true;
                            break; // ?
                        };
                        
                        cout << "[Adaptive learning] Changing parameters with strategy " << strat << " (event: " << event << ", slope: " << slope << ", corr: " << corr << ")." << endl;
                        switch (abs(strat)) {
                            case 0:
                                _wavenet->setBatchSize( 2 * _wavenet->batchSize() );
                                break;
                            case 1:
                                if (strat > 0) {
                                    _wavenet->setAlpha( 2 * _wavenet->alpha() );
                                } else {
                                    _wavenet->setAlpha( 0.5 * _wavenet->alpha() );
                                }
                                break;
                            default:
                                cout << "Didn't recognise adative learning strategy." << endl;
                                break;
                        }
                        
                        adaptiveSteps.at(abs(strat))++;
                        //strat = (strat + 1) % Nstrats;
                        
                        // ----------------------------------------------------------------
                        
                        lastNcosts.clear();
                        
                    } / * [END] If do adapt. * /
                    ----------------------------------- */
                    
                } /* [END] Adaptive learning. */
                     
                
            } while (!done && (_Nevents < 0 || (++event < _Nevents && _generator->good())));
            
            if (done) { break; }
        }
        
        //_wavenet->flushBatchQueue(); // <<< Might bias!
        
        // * Saving snapshot to file.
        char buff[100];
        snprintf(buff, sizeof(buff), "%s.%06u.snap", _name.c_str(), init + 1);
        std::string filename = buff;
        _wavenet->save(_basedir + _name + "/snapshots/" + filename);
    }
    
    std::ofstream outFileStream (_basedir + _name + "/README" );
    
    outFileStream << "Nevents: " << _Nevents << "\n";
    outFileStream << "_Nepochs: " << _Nepochs << "\n";
    outFileStream << "_Ninits: " << _Ninits << "\n";
    outFileStream << "_Ncoeffs: " << _Ncoeffs << "\n";
    
    outFileStream.close();
    
    _wavenet->clear();
    
    return;
    
}