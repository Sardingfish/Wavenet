/**
 * @file   Example01.cxx
 * @author Andreas Sogaard
 * @date   24 November 2016
 * @brief  Annotated minimal working example.
 */

// STL include(s).
#include <string> /* std::string */
#include <cmath> /* sqrt */

// Wavenet include(s).
#include "Wavenet/Logger.h" /* FCTINFO, FCTWARNING */
#include "Wavenet/Generators.h" /* wavenet::NeedleGenerator */
#include "Wavenet/Wavenet.h" /* wavenet::Wavenet */
#include "Wavenet/Coach.h" /* wavevent::Coach */

/**
 * Example01: Find optimal wavelet basis with 2 filter coefficients for needle-
 *            like input.
 *
 * Requirements: None
 *
 * This first example shows a minimal setup, identical to that in Example00, but
 * with added context. This example (and the previous one) shows how it is
 * possible to learn an wavelet basis given a certain class of training data.
 * Here, we're using the NeedleGenerator, but several other classes can be used
 * out of the box, just as new, specialised generators can be defined by the
 * user in 'Wavenet/Generators.h' and then used as in this example.
 *
 * In this example, we're performing an optimisation in two-dimensional filter
 * coefficient space. Due to the regularisation conditions (C1-5) on the filter
 * coefficients, detailied in the companion note, there is actually only a
 * single configuration of the filter coefficients which corresponds to an
 * actual wavelet basis. This means that the optimisation in this example is
 * solely with respect to the regularisation terms; the sparsity term does not
 * contribute at the lowest dimension. However, for larger number of filter
 * coefficients (4 and upwards), the region of filter coefficient space which
 * satisfies all five regularisation conditions has dimension > 0, which means
 * that _there_ we will in fact perform an optimisation of the sparisty term as
 * well.
 *
 * The example is concluded by showing the initial and final filter coefficient
 * configurations, and their cost (which is also shown during training). The
 * final configuration is compared to the only mathematically allowed one.
 * Hopefully we're not too far off!
 */
int main (int argc, char* argv[]) {

    FCTINFO("===========================================================");
    FCTINFO("Running Wavenet Example01.");
    FCTINFO("-----------------------------------------------------------");

    // Set the number of filter coefficients to use in the training. This is the
    // dimension of the parameter space, in which the optimisation is performed.
    const unsigned int numCoeffs = 2;

    // Create an instance of the basic 'NeedleGenerator' class, and specify the
    // shape of the data to generate. The NeedleGenerator produces input 
    // matrices with all but a few entries set equal to zero, which then stand 
    // out as isolated 'needles'. Note that, due to the dydic structure of the 
    // wavelet transform, the shape dimension(s) have to be radix 2, i.e. be 
    // equal to 2 to some integer power (1, 2, 4, 8, etc.).
    wavenet::NeedleGenerator ng;
    ng.setShape({16,16});
    
    // Create a 'Wavenet' instance. For this example we're just using default 
    // settings. For more advances tuning, see some of the later examples.
    wavenet::Wavenet wn;

    // (Optional) Print it to show its internal configuration.
    //wn.print();
    
    // Create a 'Coach' instance. This object takes care of the training for us.
    // The 'Coach' needs:
    // -- a unique name (to create a unique directory to which to save the 
    //    training history; by default it saves to './output/*name*/'),
    // -- a number of coefficients (optional; default value is 2, so setting it 
    //    to 2 as we're doing here is actually a bit silly, but we're doing it 
    //    anyway just to be completely clear),
    // -- a generator instance, to produce the data on which to train, and
    // -- a wavenet instance, which is the thing that we actually train.
    // The rest of the settings are left at their default values. For more 
    // advances tuning, see some of the later examples.
    wavenet::Coach coach ("Example01");

    coach.setNumCoeffs(numCoeffs);
    coach.setGenerator(&ng);
    coach.setWavenet  (&wn);
    
    // Run the training.
    bool good = coach.run();

    // Check whether an error occured.
    if (!good) {
        FCTWARNING("Uhh-oh! Something went wrong.");
        return 0;
    }

    // Print initial and final configurations, with costs:
    FCTINFO("");
    // We subtract one, because the initial condition, stored in the filter log,
    // doesn't _really_ count as an update.
    FCTINFO("Number of updates: %d", wn.filterLog().size() - 1); 
    FCTINFO("Initial filter coefficients were:");
    FCTINFO("  [%-4.3f, %-4.3f] (cost: %4.2f)", wn.filterLog()[0][0], wn.filterLog()[0][1], wn.costLog()[0]);
    FCTINFO("Final filter coefficients found were:");
    FCTINFO("  [%-4.3f, %-4.3f] (cost: %4.2f)", wn.filter()[0], wn.filter()[1], wn.lastCost());
    FCTINFO("");
    FCTINFO("The only configuration of two filter coefficients allowed");
    FCTINFO("by the regularisation conditions is: ");
    FCTINFO("  [1/sqrt(2), 1/sqrt(2)] = [%-4.3f, %-4.3f]", 1./sqrt(2.), 1./sqrt(2));
    FCTINFO("Did we get close?");
   
    // Information about the Coach configuation used in the training can be 
    // found in './output/Example01/README', and the final snapshot of the 
    // trained wavenet can be found in './output/Example01/snapshots/'.

    // Try running a few different times. The initial condition for the filter 
    // coefficients is random, so you should see that the initial filter changes 
    // between initialisations. But (hopefully!) you should also see that the 
    // final filter configuration is the same, namely the global minimum. 
    // That means that the optimisation worked! :)

    FCTINFO("-----------------------------------------------------------");
    FCTINFO("Done.");
    FCTINFO("===========================================================");
    
    return 1;

}
