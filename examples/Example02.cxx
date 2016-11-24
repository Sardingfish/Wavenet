/**
 * @file   Example02.cxx
 * @author Andreas Sogaard
 * @date   24 November 2016
 * @brief  Tweaking Wavenet and Coach configurations; computing cost maps.
 */

 // STL include(s).
#include <string> /* std::string */
#include <vector> /* std::vector */

#ifdef USE_ROOT
// ROOT include(s).
#include "TStyle.h"
#include "TCanvas.h"
#include "TH1.h"
#endif // USE_ROOT

// Wavenet include(s).
#include "Wavenet/Logger.h" /* FCTINFO, FCTERROR */
#include "Wavenet/Generators.h" /* wavenet::NeedleGenerator */
#include "Wavenet/Wavenet.h" /* wavenet::Wavenet */
#include "Wavenet/Coach.h" /* wavevent::Coach */

/**
 * Example02: Tweaking Wavenet and Coach configurations; computing cost maps.
 *
 * Requirements: ROOT (non-essential)
 *
 * The first part of this example looks suspiciously like Example00 and 
 * Example01, except that we're tweaking the default values slightly (mainly to 
 * show of the syntax).
 *
 * The second part is devoted to computing cost maps for the chosen class of 
 * training examples. These cost maps are used in Example03 (requires ROOT) to 
 * display the combined cost contours in filter coefficient space, as well as 
 * the paths of the filter coefficient configurations through this space during 
 * training.
 *
 * If ROOT is installed, the generator training examples used for computing the 
 * cost map are saved as PDF files.
 */
int main (int argc, char* argv[]) {

    FCTINFO("===========================================================");
    FCTINFO("Running Wavenet Example02.");
    FCTINFO("-----------------------------------------------------------");

    #ifndef USE_ROOT
    FCTINFO("* - - - - - - - - - - - - - - - - - - - - - - -*");
    FCTINFO("| This example works best with ROOT installed! |");
    FCTINFO("* - - - - - - - - - - - - - - - - - - - - - - -*");
    #endif // USE_ROOT

    // Specify the unique name of the current project.
    const std::string name = "Example02";

    // Set the number of filter coefficients.
    const unsigned int numCoeffs = 2;

    // Specify the number of initialisations, i.e. runs with different initial 
    // conditions.
    const unsigned int numInits = 10;

    // Create a Generator instance, here 'NeedleGenerator', and specify shape.
    wavenet::NeedleGenerator ng;
    ng.setShape({32, 32});
    
    // Create a 'Wavenet' instance. For this example, we're tuning the settings 
    // slightly cf. the default values.
    wavenet::Wavenet wn;
    wn.setAlpha(0.0005); // Learning rate. Default value is 0.001
    wn.setLambda(20.);   // Regularisation constant. Default value is 10.
    wn.setBatchSize(10); // Batch size. Default value is 1.
    
    // Create a 'Coach' instance. For this example, we're also tuning the 
    // settings slightly.
    wavenet::Coach coach (name);
    coach.setNumCoeffs(numCoeffs);
    coach.setGenerator(&ng);
    coach.setWavenet  (&wn);

    coach.setNumEvents(5000);     // Number of events for each epoch. Default value is 1000.
    coach.setNumEpochs(2);        // Number of epochs for each initialisation. Default value is 1.
    coach.setNumInits (numInits); // Number of initialisations. Default value is 1.
    
    coach.setUseAdaptiveLearningRate(true); // Use adaptive learning rate.
    coach.setUseAdaptiveBatchSize   (true); // Use adaptive batch size.
    coach.setTargetPrecision(0.00001);      // Stop early if reaching target precision.
    
    // Run the training.
    bool good = coach.run();

    // Check whether an error occurred.
    if (!good) {
        FCTERROR("Uhh-oh! Something went wrong.");
        return 0;
    }

    // The last bit of this example focuses on creating a _cost map_: a map of
    // the cost (regularisation and sparsity, or just one of these) for a small
    // set of training examples, computed as by scanning over two filter
    // coefficients. In this, the lowest possible dimension of phase space, it
    // is possible to perform an exhaustive search of the filter coefficient
    // space and in this way study to contours of the cost space in which our
    // learning takes places. Looking at the cost map and how filter coefficient
    // configurations traverse it during learning can help us confirm that the
    // wavenet behaves the way it should. However, computing the cost for each
    // point in a, say, 300 x 300 grid is very time consuming, and for more than
    // two filter coefficients this becomes practically impossible (which is why
    // we need the wavenet in the first place!). This example computes the cost
    // maps for the regularisation alone, the sparsity alone, and the combined
    // regularisation and sparsity cost, and saves them to file as Armadillo
    // matrices. We can then, in a later example, read these cost maps at try to
    // make some nice plots.
     
    // Whether we should overwrite an existing cost map. This is useful when you
    // change the configuration behind it, e.g. the regularisation constant
    // lambda or the class of training data, and want to update it. However,
    // computing a cost map with a reasonable degree of granularity (say,
    // 300 x 300 filter coefficient resolution) can take some time, so for your
    // own sake try to do this sparingly.
    const bool overwrite = false;

    // Initialise the names of the cost map files.
    std::string costMapName               = coach.outdir() + "costMap.mat";
    std::string costMapSparsityName       = coach.outdir() + "costMapSparsity.mat";
    std::string costMapRegularisationName = coach.outdir() + "costMapRegularisation.mat";
    
    // Determine whether we should perform the actual calculation.
    if (!wavenet::fileExists(costMapName) or overwrite) {

        // If the cost map files do not already exist, we need to produce them.
        
        // Specify the number of training examples to be used when computing the
        // cost maps. More examples given smoother sparsity contours, but take 
        // longer to run.
        const unsigned int numExamples = 10;

        // Initialise a vector of matrices to hold the training examples.
        std::vector< arma::Mat<double> > examples (numExamples);

        // Loop and generated the necessary training examples.
        for (unsigned i = 0; i < numExamples; i++) {

            // Store the next examples to in the vector.
            examples.at(i) = ng.next();

            #ifdef USE_ROOT
            // If ROOT is enabled, we draw each of the used training examples as
            // save them to file. In this way, it is completely clear how our 
            // class of training data looks, and exactly which ones have gone 
            // into computing the cost map(s).

            // Remove default statistics box.
            gStyle->SetOptStat(0);

            // Initialise TCanvas in which to draw.
            TCanvas c ("c", "", 700, 600);

            // Convert the training example matrix to a ROOT histogram, either 
            // TH1F or TH2F depending on the generator shape.
            std::unique_ptr<TH1> exampleInput = wavenet::MatrixToHist(examples.at(i), 3.2);

            // Draw the current training example histogram.
            exampleInput->Draw("COL Z");

            // Save the plot as PDF file in the output directory of the Coach 
            // instance.
            c.SaveAs((coach.outdir() + "exampleInput." + std::to_string(i + 1) +".pdf").c_str());
            #endif // USE_ROOT
        }
    

        // Compute the cost maps by looping @c examples, scanning the two filter
        // coefficient across the range [-1.2, 1.2], with a resolution of 300 
        // along each axis.
        std::vector< arma::Mat<double> > costs = wn.costMap(examples, 1.2, 300);

        // Save each of the returned cost maps to file.
        costs.at(0).save(costMapName);
        costs.at(1).save(costMapSparsityName);
        costs.at(2).save(costMapRegularisationName);
    }

    // Now you have the cost maps for a given type of input (although the type 
    // of input is of secondary importance in two filter coefficient dimensions 
    // which has a single global minimum which is the same for all types of 
    // input). 

    // If you have ROOT installed, try taking a look at Example03 where we plot 
    // the cost map, the path of the filter coefficients in this map during the 
    // training in this example, as well as graphs of the costs during training.

    // Once you are confident with how things work, you can try changing the 
    // training parameters, the number of filter coefficients, the type of 
    // training data -- you name it!

    FCTINFO("-----------------------------------------------------------");
    FCTINFO("Done.");
    FCTINFO("===========================================================");
    
    return 1;

}
