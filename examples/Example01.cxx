// STL include(s).
#include <string> /* std::string */

// Wavenet include(s).
#include "Wavenet/Logger.h" /* FCTINFO */
#include "Wavenet/Generators.h" /* wavenet::NeedleGenerator */
#include "Wavenet/Wavenet.h" /* wavenet::Wavenet */
#include "Wavenet/Coach.h" /* wavevent::Coach */

/**
 * Example01: Find the best wavelet basis with 2 filter coefficients for needle-like input.
 *
 * This first example shows a minimal setup, where ...
 *
 *
 */
int main (int argc, char* argv[]) {

    
    FCTINFO("===========================================================");
    FCTINFO(" Running Wavenet Example01.");
    FCTINFO("-----------------------------------------------------------");

    // Set the number of filter coefficients to use in the training.
    // This is the dimension of the parameter space, in which the optimisation is performed
    const unsigned int Ncoeffs = 2;

    // Create an instance of the basic 'NeedleGenerator', and specify the shape of the data to generate.
    // The NeedleGenerator produces input matrices with all but a few entries set equal to zero. 
    // Note that, due to the dydic structure of the wavelet transform, the shape dimension(s) has to be radix 2, i.e. be equal to 2 to some integer power (1, 2, 4, 8, etc.).
    wavenet::NeedleGenerator ng;
    ng.setShape({16,16});
    
    // Create a 'Wavenet' instance.
    // For this example we're just using default settings. For more advances tuning, see some of the later examples.
    wavenet::Wavenet wn;

    // (Optional) Print it to show its internal configuration.
    //wn.print();
    
    // Create a 'Coach' instance. This object takes care of the training for us.
    // The 'Coach' needs:
    // -- a unique name (to create a unique directory to which to save the training history; by default it saves to './output/*name*/'),
    // -- a number of coefficients (optional; default value is 2, so setting it to 2 as we're doing here is actually a bit redundant),
    // -- a generator instance, to produce the data on which to train, and
    // -- a wavenet instance, which is the thing that we actually train.
    // The rest of the settings are left at their default values. For more advances tuning, see some of the later examples.
    wavenet::Coach coach ("Example01");

    coach.setNcoeffs  (Ncoeffs);
    coach.setGenerator(&ng);
    coach.setWavenet  (&wn);
    
    // Run the training.
    bool good = coach.run();

    if (good) {
        // Print initial and final configurations, with cost:
        FCTINFO("");
        FCTINFO("Number of updates: %d", wn.filterLog().size() - 1); // We subtract one, because the initial condition, stored in the filter log, doesn't count as an update.
        FCTINFO("Initial filter coefficients were:");
        FCTINFO("  [%-4.3f, %-4.3f] (cost: %4.2f)", wn.filterLog()[0][0], wn.filterLog()[0][1], wn.costLog()[0]);
        FCTINFO("Final filter coefficients found were:");
        FCTINFO("  [%-4.3f, %-4.3f] (cost: %4.2f)", wn.filter()[0], wn.filter()[1], wn.lastCost());
    } else {
        FCTWARNING("Uhh-oh!");
    }
    
    // Information about the Coach configuation can be found in './output/Example01/README', and the final snapshot of the trained wavenet can be found in './output/Example01/snapshots/'.

    // Try running a few different times. The initial condition for the filter coefficients is random, so you should see that the initial filter changes between rounds. But (hopefully!) you should also see that the final filter configuration is the same, namely the global minimum. That means that the optimisation worked! :)

    FCTINFO("-----------------------------------------------------------");
    FCTINFO(" Done.");
    FCTINFO("===========================================================");
    
    return 1;

}

