#ifndef WAVENET_COACH_H
#define WAVENET_COACH_H

/**
 * @file   Coach.h
 * @author Andreas Sogaard
 * @date   15 November 2016
 * @brief  Class for managing the training of Wavenet objects.
 */

// STL include(s).
#include <iostream> /* std::cout */
#include <string> /* std::string */
#include <fstream> /* std::ofstream */
#include <cmath> /* log10 */
#include <cstdlib> /* system */

// Wavenet include(s).
#include "Wavenet/Utilities.h"
#include "Wavenet/Logger.h"
#include "Wavenet/Wavenet.h"
#include "Wavenet/GeneratorBase.h"


namespace wavenet {

/**
 * Class for managing the training of Wavenet objects.
 *
 * This class requires (1) a Wavenet object and a (2) generator-type object 
 * (inheriting from GeneratorBase) as members. The Wavenet is the object being
 * trained, and the generator provides the input upon which to train.
 *
 * A few adaptive learning methods may be enabled in order to improve 
 * optimisation speed and/or quality.
 */
class Coach : Logger  {

    /**
     * Implementation comment:
     *
     * Cf. the ATLAS C++ coding guidelines v0.2:
     *
     *   "If a class owns memory via a pointer data member, then the copy 
     *    constructor, the assignment operator, and the destructor should all be 
     *    implemented. [define-copy-and-assignment ]"
     *   [http://atlas-computing.web.cern.ch/atlas-computing/projects/qa/draft_\
     *    guidelines-0.2.html]
     *
     * However, since the Coach class doesn't assume ownership of the data to 
     * which the pointer-type memeber variables point (specifically, the wavenet 
     * and generator objects), the default implementation of these operators 
     * will suffice.
     */

public:
    
/// Constructor(s).
    Coach (const std::string& name) :
        m_name(name)
    {};

   
    /// Destructor(s).
    ~Coach () {
        // Don't delete pointer-type member variables m_wavenet and m_generator.
        // The Coach object doesn't own these; it simply wraps around them. 
        // Therefore the lifetime of these should be handled externaly.
    };
        

/// Set method(s).
    // Set the name.
    inline void setName (const std::string& name) { m_name = name; return; }
    // Set the base directory.
    inline void setBasedir (const std::string& basedir);

    // Specify wavenet instance to be trained.
    inline void setWavenet (Wavenet* wavenet) { m_wavenet = wavenet; return; }
    // Specify generator instance to provide training data.
    inline void setGenerator (GeneratorBase* generator) { m_generator = generator; return; }
    
    // Set the number of events.
    void setNumEvents (const int& numEvents);
    // Set the number of epochs.
    inline void setNumEpochs (const unsigned& numEpochs) { m_numEpochs = numEpochs; return; }
    // Set the number of initialisation.
    inline void setNumInits  (const unsigned& numInits) { m_numInits = numInits; return; }
    // Set the number of filter coefficients.
    void setNumCoeffs (const unsigned& numCoeffs);

    // Specify whether to use adaptive learning rate.
    inline void setUseAdaptiveLearningRate (const unsigned& useAdaptiveLearningRate = true) { m_useAdaptiveLearningRate = useAdaptiveLearningRate; return; }
    // Specify whether to use adaptive batch size
    inline void setUseAdaptiveBatchSize (const unsigned& useAdaptiveBatchSize = true) { m_useAdaptiveBatchSize = useAdaptiveBatchSize; return; }
    // Specify whether to use simulated annealing
    inline void setUseSimulatedAnnealing (const unsigned& useSimulatedAnnealing = true) { m_useSimulatedAnnealing = useSimulatedAnnealing; return; }
    // Set the target filter coefficient space precision.
    void setTargetPrecision (const double& );
    
    // Set the print level.
    inline void setPrintLevel (const bool& printLevel) { m_printLevel = printLevel; return; }
    

/// Get method(s).
    // Returns the name.
    inline std::string name () const { return m_name; }
    // Returns the base directory.
    inline std::string basedir () const { return m_basedir; }
    // Returns the full output directory.
    inline std::string outdir () const { return m_basedir + m_name + "/"; }
    // If the output directory, and optional subdirectory, does't exist, create 
    // it.
    void checkMakeOutdir (const std::string& subdir = "") const;
    
    // Returns the member wavenet instance.
    inline Wavenet* wavenet () const { return m_wavenet; }
    // Returns the member generator instance.
    inline GeneratorBase* generator () const { return m_generator; }

    // Returns the number of events.
    inline int numEvents () const { return m_numEvents; }
    // Returns the number of epochs.
    inline unsigned numEpochs () const { return m_numEpochs; }
    // Returns the number of initialisations.
    inline unsigned numInits () const { return m_numInits; }
    // Returns the number of filter coefficients.
    inline unsigned numCoeffs () const { return m_numCoeffs; }
    
    // Returns whether the instance is configured to use adaptive learning rate.
    inline bool useAdaptiveLearningRate () const { return m_useAdaptiveLearningRate; }
    // Returns whether the instance is configured to use adaptive learning rate.
    inline bool useAdaptiveBatchSize () const { return m_useAdaptiveBatchSize; }
    // Returns whether the instance is configured to use simulated annealing.
    inline bool useSimulatedAnnealing () const { return m_useSimulatedAnnealing; }
    // Returns the filtee coefficient space target precision.
    inline double targetPrecision () const { return m_targetPrecision; }
    
    // Returns the print level.
    inline unsigned printLevel () const { return m_printLevel; }
    
    
/// High-level training method(s).
    /**
     * Main method for performing the training of the member wavenet object, 
     * according to internal configuration of the current instance. Return true 
     * if the training was completed succesfully.
     */
    bool run ();
    

private:

/// Data member(s).
    // Directory structure member(s).
    /**
     * Unique name of the Coach instance.
     *
     * All output is saved under this name, in the 'outdir' which is given as 
     *   m_basedir/m_name/<here>
     */
    std::string m_name = "";
    
    /**
     * Base directory, in which to save the output
     *
     * It should usually be fine to leave this at the default value, to avoid
     * clutter.
     */
    std::string m_basedir = "./output/";
    

    // Main object member(s).
    /**
     * Pointer to the wavenet object to train.
     */
    Wavenet* m_wavenet = nullptr;

    /**
     * Pointer to the generator object providing the input for the training.
     */
    GeneratorBase* m_generator = nullptr;
    
    // Training schedule member(s).
    /**
     * Number of events for each epoch. If set to -1, the training will keep 
     * going until the generator is no longer in a good condition
     */
    int m_numEvents = 1000;
    
    /**
     * Number of epochs for each initialisation.
     * 
     * That is, the number of repeated loops over the same set of data. Only 
     * makes sense/a difference for input of fixed, limited size, but may be 
     * used for all generators.
     */
     unsigned m_numEpochs = 1;
    
    /**
     * Number of initialisations.
     *
     * That is, the number of independent runs with different initial conditions
     * but otherwise with the same configuration. Since the filter coefficient 
     * optimisation, for more than two filter coefficieints, is a non-convex 
     * problem, performing for training with several different, random 
     * initialisation may be vital to find a good/globally minimal solution 
     * (which can, however, not be guaranteed).
     */
    unsigned m_numInits = 1;
    
    /**
     * Number of wavelet filter coefficients to use in the training.
     *
     * Used for generating the random initial conditions.
     */
    unsigned m_numCoeffs = 2;

    // Adaptive learning member(s).
    /**
     * Whether to make the learning rate (alpha) adaptive.
     *
     * If the learning rate is adaptive, the Coach keeps track of the mean and 
     * total learning step size during the last N ('useLastN' in Coach.cxx) 
     * update steps. If the total step size is smaller than the mean step size, 
     * the learning rate is reduced by 0.5. This allows for finding a more 
     * precise minimum.
     *
     * If a target precision is set (@see m_targetPrecision), the training may
     * break early if the mean step size is smaller than the target precision. 
     * This may reduce training time. However, if simulated annealing is also 
     * enabled the training won't break early, since then the minimum would have
     * been found for a regularisation constant (lambda) which is smaller than
     * the specified value which will make results irreproducible.
     */
    bool m_useAdaptiveLearningRate = false;

    /**
     * Whether to make the batch size adaptive.
     *
     * If the learning rate is adaptive, the Coach keeps track of the mean and 
     * total learning step size during the last N ('useLastN' in Coach.cxx) 
     * update steps. If the total step size is smaller than the mean step size, 
     * the batch size is increase by a factor of two. This allows for finding a
     * more precise minimum.
     *
     * If a target precision is set (@see m_targetPrecision), the training may
     * break early if the mean step size is smaller than the target precision. 
     * This may reduce training time. However, if simulated annealing is also 
     * enabled the training won't break early, since then the minimum would have
     * been found for a regularisation constant (lambda) which is smaller than
     * the specified value which will make results irreproducible.
     */
    bool m_useAdaptiveBatchSize = false;
    
    /**
     * Whether to use simulated annealing.
     *
     * A variant of simulated annealing can be enabled, in which case the 
     * regularisation constant (lambda) starts out as zero and then grows as
     *   \lambda = \lambda_{0} * \frac{f}{(2 - f)^{2}}
     * where \lambda_{0} is the specified regularisation parameter, and f is 
     * the fraction of events, in the range [0, 1], processed for current
     * initialisation. When f #rightarrow 1, \lambda \rightarrow \lambda_{0}, 
     * and the final set of filter coefficients are found as minima for the 
     * requested value of lambda.
     *
     * The use of simulated annealing is intended to overcome the steep const
     * contours arising regularsation according to the large number (up to 5) 
     * constraints on the filter coefficients. This means that the coefficients 
     * will initially be optimised according to the sparsity criterion alone, 
     * and only gradually with the regularisation be imposed. This should also 
     * allow to use larger learning rates, since wavenet will have moved out of 
     * the steepest-cost regions of the filter coefficient space before lambda
     * becomes too large to cause divergences that would otherwise have occured 
     * had \lambda_{0} been used all along.
     * 
     * Since the regularisation constant is gradually increased, so is the total
     * (sparsity + regularisation) cost for identical filter coefficient
     * configurations. Therefore, the cost of the wavenet object will _not_ be 
     * monotonically decreasing with training time as it would have otherwise 
     * been (at least approximately, or in the limit of infinitely large batch
     * size).
     *
     * If simulated annealing is used, the training will not break early.
     */
    bool m_useSimulatedAnnealing = false;
    
    /**
     * (Optional) target precision of the filter coefficient optimisation.
     *
     * If a target precision is specified, and adaptive learning rate is 
     * enabled, the training may break early once a sufficiently good solution 
     * has been found. This early breaking may reduce overall training time, but
     * is disabled if simulated annealing is also enabled.
     */
    double m_targetPrecision = -1;
    
    // Printing member(s).
    /**
     * The depth to which progress information should be printed.
     * 
     * If m_printLevel:
     * .. <= 0, no information regarding the progress of the training is printed.
     * .. == 1, progress at the level of initialisations is printed.
     * .. == 2,    -     -   -     -  -  epochs is printed.
     * .. >= 3,    -     -   -     -  -  events is printed.
     */
    unsigned m_printLevel = 3;
    
};

} // namespace

#endif // WAVENET_COACH_H
