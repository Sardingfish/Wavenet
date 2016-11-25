/**
 * @file   Example03.cxx
 * @author Andreas Sogaard
 * @date   25 November 2016
 * @brief  Plotting cost maps, filter-, and cost logs.
 */

 // STL include(s).
#include <string> /* std::string */
#include <vector> /* std::vector */
#include <utility> /* std::move */
#include <memory> /* std::unique_ptr */
#include <cassert> /* assert */

#ifdef USE_ROOT
// ROOT include(s).
#include "TGraph.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TStyle.h"
#include "TEllipse.h"
#include "TMarker.h"
#include "TLatex.h"
#include "TColor.h"
#include "TLine.h"
#endif // USE_ROOT

// Wavenet include(s).
#include "Wavenet/Utilities.h" /* wavenet::isRadix2 */
#include "Wavenet/Logger.h" /* FCTINFO, FCTWARNING */
#include "Wavenet/Generators.h" /* wavenet::NeedleGenerator */
#include "Wavenet/CostFunctions.h" /* wavenet::RegTerm */
#include "Wavenet/Wavenet.h" /* wavenet::Wavenet */
#include "Wavenet/Coach.h" /* wavevent::Coach */


/**
 * Example03: Plotting cost maps, filter-, and cost logs.
 *
 * Requirements: ROOT
 * 
 * This example shows how, using Snapshots, we can easily read information from
 * existing projects as well as how we can produce a collection of graphical
 * outputs to illustrate the learning process.
 *
 * In particular, using the cost maps computed in the previous example 
 * (Example02), we draw the evolutions of the filter coefficients in this space,
 * to illustrate the workings of the gradient descend algorithm; we plot graphs
 * of the combined cost for a number of different initialisations as a functions
 * of the number of training updates, showing how the cost decreases with
 * training until reaching a (global) minimum; we plot the distributions of
 * inner products between all function in the best learned basis, i.e. the one
 * with the lowest final cost, in order to illustrate the orthonormality of the
 * best solution; and finally we plot the basis functions (1- or 2D), throughout
 * the learning process, of the best final wavelet basis, in order to directly 
 * show the effect of the training and the properties of the optimal solution.
 *
 * This example can be modified by changing the shape (and dimension) of the
 * basis functions to be drawn. In addition, other projects can be specified,
 * from which to read training information. The _contents_ of the graphical
 * output depends on the configuration of the training, as performed in e.g.
 * Example02.
 */
int main (int argc, char* argv[]) {

    FCTINFO("===========================================================");
    FCTINFO("Running Wavenet Example03");
    FCTINFO("-----------------------------------------------------------");

    #ifndef USE_ROOT
    FCTINFO("* - - - - - - - - - - - - - - - - - - - - - - - - - - - *");
    FCTINFO("| This example only works with ROOT installed -- sorry! |");
    FCTINFO("* - - - - - - - - - - - - - - - - - - - - - - - - - - - *");
    #endif // USE_ROOT

    // Specify the unique name of the current project.
    const std::string name = "Example03";

    // Specify the unique name of the project from which to read snapshots and 
    // cost map(s).
    const std::string readFromName = "Example02";

    // Specify the number of initialisation to read. Make negative to read all
    // available. However, this latter option might land you in trouble if your
    // latest run was over fewer initialisations that some previous run,
    // possibly with another configuration. In that case you might be comparing
    // apples and oranges.
    const int numInits = 10;

    // Set the shape of the basis functions to be studied. Try experimenting
    // with shapes of different dimension, e.g. 16 x 16 and 16 x 1. The output
    // will take the difference in dimension into accout.
    const std::vector<unsigned> shape {16, 16};

    // Create a 'Wavenet' instance for reading snapshots
    wavenet::Wavenet wn;

    // Create dummy 'Coach' instances, for reading and saving.
    wavenet::Coach coach         (name);
    wavenet::Coach readFromCoach (readFromName);
    
    #ifdef USE_ROOT

    FCTINFO("Plotting the results of '%s'.", readFromName.c_str());

    // Initialise the 'Snapshot' object that we're going to use to read the
    // results of the optimisation. The snapshots for each initialisation above
    // are stored 'snapshots/' subdirectory in the base directory of the
    // responsible Coach instance (by default './output/'). If we specify a
    // pattern (by the %-guy in the string), the snapshot can automatically
    // iterate though successivle snapshots. The format chosen below is the one
    // used by default when the Coach writes the snapshots to file.
    const std::string pattern = readFromCoach.outdir() + "snapshots/" +  readFromName + ".%06u.snap";
    wavenet::Snapshot snap (pattern);

    if (!snap.exists()) {
        FCTERROR("Snapshot '%s' doesn't exist.", snap.file().c_str());
        return 0;
    }

    // Set up vector of TGraph to store and plot the cost graphs from each
    // initialisation.
    std::vector< TGraph > costGraphs;
    std::vector< TGraph > filterGraphs;

    // Initialise the minimum and maximum cost throughout training, for use with
    // plotting.
    double minCost = 1./wavenet::EPS;
    double maxCost = 0.;

    // Initialise the length of the longest cost log, for use when drawing axes.
    unsigned longestCostLog = 0;

    // Initialise index of the best basis, i.e. the one wit the lowest final
    // cost.
    unsigned bestBasis;

    // Loop through successive snapshots.
    while (snap.exists() && (numInits < 0 || snap.number() < numInits)) {
        
        // Print progress.
        FCTINFO("  Reading snapshot %d/%s.", snap.number() + 1, (numInits < 0 ? "-" : std::to_string(numInits)).c_str());

        // Load the wavenet snapshot.
        wn.load(snap);
        
        // Print the last entry in the cost log.
        FCTINFO("    Last cost: %.3f", wn.lastCost());

        // Get the filter- and cost logs.
        auto filterLog = wn.filterLog();
        auto costLog   = wn.costLog();
        
        // Get number of filter coefficients.
        const unsigned numCoeffs = filterLog.size();
    
        // Add the graph of the current cost log to the vector.
        costGraphs.push_back( wavenet::costGraph( costLog ) );

        // Get the final and initial (assumed maximal, if everything works)
        // costs in the current log.
        double tmpMin = wn.lastCost();
        double tmpMax = wn.costLog()[0];

        // Update the best basis index, if the current last cost is the lowest
        // one encountered yet.
        if (tmpMin < minCost) {
            bestBasis = snap.number();
            minCost = tmpMin;
        }

        // Possibly update maximum cost.        
        maxCost = (tmpMax > maxCost && tmpMax > 0 ? tmpMax : maxCost);

        // Possibly update length of longest cost log    
        longestCostLog = (costLog.size() > longestCostLog ? costLog.size() : longestCostLog);
        
        // Create filter graph.
        const unsigned int numSteps = filterLog.size();
        double x[numSteps], y[numSteps];
        for (unsigned i = 0; i < numSteps; i++) {
            x[i] = arma::as_scalar(filterLog.at(i).row(0));
            y[i] = arma::as_scalar(filterLog.at(i).row(1));
        }
        
        // Store it in the vector.
        filterGraphs.push_back( TGraph(numSteps, x, y) );
        
        // Move to next snapshot.
        snap++;
        
    }

    // Print minimum and maximum cost found.
    FCTINFO("  Maximum cost: %.3f", maxCost);
    FCTINFO("  Minimum cost: %.3f", minCost);



    /// General, cosmetic stuff.
    // -------------------------------------------------------------------------

    // Remove default statistics box.
    gStyle->SetOptStat(0);

    // Define custom colours palette.
    int kMyRed = 1756; // Colour index.
    TColor *MyRed =  new TColor(kMyRed,  224./255.,   0./255.,  42./255.);
    int kMyBlue = 1757;
    TColor *MyBlue = new TColor(kMyBlue,   3./255.,  29./255.,  66./255.);



    /// Filter graphs.
    // -------------------------------------------------------------------------

    // In this section we (try to) read the cost maps compute in the previous
    // example, plot the corresponding contours, and overlay filter graphs, i.e.
    // the "trajectories" of the learning process in filter coefficient space
    // for each initialisations.

    { // Restricted scope.

        FCTINFO("Plotting filter graphs.");

        // Define path, where we should look for the cost maps to load.
        std::string costMapName               = readFromCoach.outdir() + "costMap.mat";
        std::string costMapSparsityName       = readFromCoach.outdir() + "costMapSparsity.mat";
        std::string costMapRegularisationName = readFromCoach.outdir() + "costMapRegularisation.mat";
        
        // Check whether a file exists at the specified location for the
        // combined cost map.
        if (!wavenet::fileExists(costMapName)) {
            FCTWARNING("Cost map '%s' doesn't exist. We need this the draw the filter graphs. Please make sure that '%s' produces these maps.", costMapName.c_str(), readFromName.c_str());
        } else {

            // If the cost maps have been produced, initialise matrices for each
            // type.
            arma::Mat<double> costMap;
            arma::Mat<double> costMapSparsity;
            arma::Mat<double> costMapRegularisation;

            // Load the cost maps from file.
            costMap.load(costMapName);
            //arma::Mat<double> costMapSparsity.load(costMapSparsityName); // Not used.
            //arma::Mat<double> costMapRegularisation.load(costMapRegularisationName); // Not used.

            // Define colour palette for cost contours (dark blue to white).
            const int Number = 2; 
            double Red[Number]    =  {   3./255., 0.98 }; 
            double Green[Number]  =  {  29./255., 0.98 }; 
            double Blue[Number]   =  {  66./255., 0.98 }; 
            double Length[Number] =  { 0.0, 1.00 }; 
            int nb = 104;
            TColor::CreateGradientColorTable(Number, Length, Red, Green, Blue, nb);

            // Define ROOT TCanvas.
            TCanvas c ("c", "", 700, 600);
            c.SetLogz(true);

            // Convert combined cost map to ROOT histogram.
            std::unique_ptr<TH1> J = wavenet::MatrixToHist(costMap, 1.2);

            // Set number of contours to draw.
            J->SetContour(nb); 
            
            // Set the maixmum (necessary?).
            J->SetMaximum(100.);

            // Perform styling.
            J->GetXaxis()->SetTitle("Filter coefficient a_{1}");
            J->GetYaxis()->SetTitle("Filter coefficient a_{2}");
            J->GetZaxis()->SetTitle("Cost (sparsity + regularisation) [a.u.]");

            J->GetXaxis()->SetTitleOffset(1.2);
            J->GetYaxis()->SetTitleOffset(1.3);
            J->GetZaxis()->SetTitleOffset(1.4);

            c.SetTopMargin   (0.09);
            c.SetBottomMargin(0.11);
            c.SetLeftMargin  (0.10 + (1/3.)*(1/7.));
            c.SetRightMargin (0.10 + (2/3.)*(1/7.));

            c.SetTickx();
            c.SetTicky();

            // Draw the cost map.
            J->Draw("CONT1 Z");
            c.Update();

            // Draw unit circle (one of the five conditions on the filter coefficients).
            TEllipse normBoundary;
            normBoundary.SetFillStyle(0);
            normBoundary.SetLineStyle(2);
            normBoundary.SetLineColor(kMyRed);
            normBoundary.DrawEllipse(0., 0., 1., 1., 0., 360., 0.);

            // Draw the filter evolution graphs, and markers at each end.
            for (unsigned m = 0; m < filterGraphs.size(); m++) {

                // Filter evolution graph.
                filterGraphs.at(m).Draw("L same");

                // Initialise marker in filter coefficient space.
                TMarker marker;
                double x, y;

                // Draw initial marker (red).
                marker.SetMarkerColor(kRed);
                marker.SetMarkerStyle(8);
                marker.SetMarkerSize (0.3);
                filterGraphs.at(m).GetPoint(0, x, y);
                marker.DrawMarker(x, y);
                
                //Draw final marker (blue).
                marker.SetMarkerColor(kBlue);
                marker.SetMarkerStyle(19);
                marker.SetMarkerSize (0.3);
                filterGraphs.at(m).GetPoint(filterGraphs.at(m).GetN() - 1, x, y);
                marker.DrawMarker(x, y);
            }


            // Make sure that the output directory exists.
            coach.checkMakeOutdir();

            // Save the final cost map as PDF.
            c.SaveAs((coach.outdir() + "FilterGraph.pdf").c_str());

        } // end: cost map exists

    } // end: restricted scope



    /// Cost graphs.
    // -------------------------------------------------------------------------

    // In this section we draw graphs for the costs logged throughout the
    // training; one graph for each initialisation. Ideally, these graphs should
    // be monotonically decreasing, as functions of the number of training
    // updates, towards some (hopefully global) minimum. However, since we are
    // utilising stochastic gradient descend, the sparsity term will be
    // susceptible to fluctuations in the measured sparsity for different
    // training examples (the regularisation, which is common for all training
    // examples, should -- on its own -- result in such monotonically decreasing
    // cost graphs.) This means that noise will occur to obscure the descreasing
    // behaviour. By increasing the batch size this effect can be minimised, the
    // price being longer run times. In the limit of infinitely large batches 
    // the cost graphs will be monotonic. However, if inertia is used in the
    // training, this is not the case anymore: sufficiently large inertia may
    // lead to the wavenet actually "climbing up hills" in the combined cost
    // space.

    { // Restricted scope.

        FCTINFO("Plotting cost graphs.");

        // Initialise canvas.
        TCanvas c ("c", "", 700, 600);
        c.SetLogy(true);

        c.SetTickx();
        c.SetTicky();

        // Set axis ranges.
        double ymin = minCost / 2.;
        double ymax = maxCost / 2.;

        // Draw axis for longest cost log.
        for (unsigned m = 0; m < costGraphs.size(); m++) {

            if (costGraphs.at(m).GetN() == longestCostLog) {
                costGraphs.at(m).SetTitle("");
                costGraphs.at(m).GetXaxis()->SetTitle("Number of filter coefficient updates");
                costGraphs.at(m).GetYaxis()->SetTitle("Cost (sparsity + regularisation) [a.u.]");
                costGraphs.at(m).GetXaxis()->SetRangeUser(0, longestCostLog); 
                costGraphs.at(m).GetYaxis()->SetRangeUser(ymin, ymax); 
                costGraphs.at(m).Draw("LAXIS");
                c.Update();
                break;
            }
        }
        
        for (unsigned m = 0; m < costGraphs.size(); m++) {
            costGraphs.at(m).SetLineStyle(1);
            costGraphs.at(m).SetLineWidth(2);
            costGraphs.at(m).SetLineColor((m < 10 ? kRed : kBlue) - 10 + m % 15);

            costGraphs.at(m).Draw("L same");
        }

        // Update canvas.
        c.Update();

        // Make sure that the output directory exists.
        coach.checkMakeOutdir();

        // Save plot.
        c.SaveAs((coach.outdir() + "CostGraph.pdf").c_str());

    } // end: restricted scope



    /// Orthonormality of best basis.
    // -------------------------------------------------------------------------

    // The functional basis corresponding to the final, optimal filter 
    // coefficient configuration should at least be orthonormal, even if the
    // wavelet-specific regularisation conditions have been omitted -- provided 
    // that the training has had time to reach the sub-space where all 
    // regularisation conditions are met. A quick check of this is to compute 
    // and plot the inner product of all combininations of basis functions. This
    // should result in a small peak (with N entries) around 1, and a larger 
    // peak (with N(N - 1) entries) around 0. If this is the case, we know that
    // the found pseudo-basis is indeed sufficiently close to an actual, exact
    // basis to be used in practice.
    //
    // If the inner product are too far from 1 and 0, respectively, try
    // increasing the regularisation constant lambda of the Wavenet object being
    // trained.

    { // Restricted scope.

        FCTINFO("Studying orthonormality of the best basis found.");

        // Initialise canvas.
        TCanvas c ("c", "", 700, 600);
        c.SetLogy(true);

        c.SetTickx();
        c.SetTicky();

        // Load the snapshot of the wavenet with the lowest final cost.
        snap.setNumber(bestBasis);
        wn.load(snap);

        // Initialise histogram of inner products.
        TH1F innerProducts ("innerProducts", "", 200, -0.5, 1.5);

        // Perform styling.
        innerProducts.SetLineStyle(1);
        innerProducts.SetLineWidth(2);
        innerProducts.SetLineColor(kMyBlue);
        innerProducts.GetXaxis()->SetTitle("Basis function inner products");
        innerProducts.GetXaxis()->SetTitle("Number of basis function pairs");

        // Initialise number shape of position space in which to draw all basis
        // function.
        const unsigned int sizex = shape[0];
        const unsigned int sizey = shape[1];

        // Initialise basis function matrices.
        arma::Mat<double> f1;
        arma::Mat<double> f2;

        // Loop all basis functions to study.
        for (unsigned i = 0; i < wavenet::sq(sizex); i++) {
            for (unsigned j = 0; j < wavenet::sq(sizey); j++) {

                // Get the two basis function, the inner product of which to
                // take.
                f1 = wn.basisFunction(sizex, sizey, i / sizex, i % sizey);
                f2 = wn.basisFunction(sizex, sizey, j / sizex, j % sizey);

                // Take the inner product, realised as a Frobenius matrix inner
                // product.
                double innerProduct = wavenet::frobeniusProduct(f1,f2);

                // Fill the histogram of inner products, properly accounting for
                // under- and overflow.
                innerProducts.Fill( innerProduct < -0.5 ? -0.499 : (innerProduct > 1.5 ? 1.499 : innerProduct) );
            }
        }

        // Draw histogram.
        innerProducts.Draw("HIST");

        // Make sure that the output directory exists.
        coach.checkMakeOutdir();

        // Save plot.
        c.SaveAs((coach.outdir() + "Orthonormality.pdf").c_str());
    
    } // end: restricted scope



    /// Learning of best basis.
    // -------------------------------------------------------------------------

    // In this, final section we plot the wavelet basis functions, for the best
    // final filter coefficient configuration, as they evolved throughout the
    // learning process. This allows us to visualise the learning process
    // directly. Depending on the specified shape, 1- or 2D basis functions are
    // drawn. The set of output images can be used to make a movie (e.g. using
    // ffmpeg) showning the evolution of the best functional basis given a
    // certain class of training data.

    { // Restricted scope.

        FCTINFO("Plotting the learning process for the best basis found.");

        // Load the snapshot of the wavenet with the lowest final cost.
        snap.setNumber(bestBasis);
        wn.load(snap);

        // Specify the size of the grid on which to plot the basis functions.
        // If the requested shape is 2D, the grid is square. Otherwise the grid
        // is rectangular, with the number of rows corresponding to the number
        // of frequency scales, and the number of columns equal to the number of
        // basis functions at the highest frequency scale.
        const unsigned dim = 8;

        // The dimension has to be radix 2.
        assert(wavenet::isRadix2(dim));
        
        // Convert the dimension to a float, and define the margin between basis
        // function pads.
        const double dimf = double(dim);
        const double marg = 0.03;

        // Get the dimension along each axis. If the requested dimension is
        // smaller than the corresponding shape size specified above, the latter
        // is used.
        unsigned dimx = std::min(shape[0], dim);
        unsigned dimy = std::min(shape[1], dim);

        // If the input data is 1D, redefine the corresponding axis dimension in
        // order to make nice 1D basis function plots.
        if      (dimx == 1) { dimx = log2(dimy) + 2; }
        else if (dimy == 1) { dimy = log2(dimx) + 2; }

        // Convert dimensions to float.
        const double dimfx = double(dimx);
        const double dimfy = double(dimy);

        // Define colour palette for 2D basis functions (read over white to dark
        // blue).
        const int Number = 3; 
        double Red[Number]    =  { 224./255., 0.98,   3./255. }; 
        double Green[Number]  =  {   0./255., 0.98,  29./255. }; 
        double Blue[Number]   =  {  42./255., 0.98,  66./255. }; 
        double Length[Number] =  { 0.00, 0.50, 1.00 }; 
        int nb = 104;
        TColor::CreateGradientColorTable(Number, Length, Red, Green, Blue, nb);

        // Initialise vector of colours, for 1D basis function.
        std::vector<unsigned> colours = {kViolet + 7, kAzure  + 7, kTeal, 
                                         kSpring - 2, kOrange - 3, kPink};

        // Define the text size, to be used when writing information on the
        // output plots.
        const double textSize  = 0.035;

        // Comput the (relative) margin on the top of the canvas, where the test
        // is written. Corresponds to two lines at the specified text size.
        const double topMargin = 2 * textSize * 1.3 / std::max(dimfy/dimfx, 1.);

        // Initialise the canvas.
        TCanvas c ("c", "", 1200 * dimfx/dimf, 1200 * dimfy / dimf);
                
        // Initialise (nested) vector of ROOT TPads on which to draw the basis
        // functions.
        std::vector< std::vector< std::unique_ptr<TPad> > > pads (dimx);
        for (unsigned i = 0; i < dimx; i++) {
            pads.at(i).resize(dimy);
        }

        // Draw one TPad for each basis function.
        for (unsigned i = 0; i < dimx; i++) {
            for (unsigned j = 0; j < dimy; j++) {

                // Initialise the (unique) pad name.
                std::string padName = std::string("pad_") + std::to_string(i) + "_" + std::to_string(j);

                // Initialise the coordinates for the pad's corners, taking into
                // accoung the top margin for text.
                double x1 =  i   /dimfx;
                double x2 = (i+1)/dimfx;
                double y1 = (dimy - j - 1)/(dimfy * (1 + topMargin));
                double y2 = (dimy - j)    /(dimfy * (1 + topMargin));

                // Create the pad, and store its unique pointer.
                pads[i][j] = std::unique_ptr<TPad>(std::move(new TPad(padName.c_str(), "", x1, y1, x2, y2)));

                // Peform styling.
                pads[i][j]->SetMargin(marg, marg, marg, marg);
                pads[i][j]->SetTickx();
                pads[i][j]->SetTicky();

                // Draw the pad on the canvas.
                c.cd();
                pads[i][j]->Draw();
            }
        }
        
        // Get the filter log to use.
        auto filterLog = wn.filterLog();

        // Get the associated cost log.
        auto costLog = wn.costLog();

        // Get the number of learning steps (the cost log is/may be one entry
        // shorter than the filter log, which is why we're using it to get the
        // largest common number of steps).
        const unsigned numSteps = costLog.size();

        // Create vector containing the steps at which we want to draw the basis
        // functions. The first and last @c completeMargin entries in the filter
        // log are always plotted, and the intermediate steps are plotted in
        // steps of @ stride. For very long-running, high-dimensional
        // optimisations it might be too much to plot every frame.
        std::vector<unsigned> steps;
        const unsigned completeMargin = 20;
        const unsigned stride = 1;
        assert(stride > 0);
        for (unsigned i = 0; i < completeMargin; i++) {
            steps.push_back(i);
        }
        for (unsigned i = completeMargin; i < numSteps - completeMargin; i += stride) {
            steps.push_back(i);
        }
        for (unsigned i = numSteps - completeMargin; i < numSteps; i++) {
            steps.push_back(i);
        }

        // Set the z-axis range for 2D basis functions).
        const double zmax = 0.40;

        // Set the y-axis range for 1D basis functions).
        const double ymax = 0.80;

        // Set the x- (and y-) axis range for 1D (and 2D) basis functions.
        const double range = 1.0;

        // Determine dimension of basis, in order to draw functions nicely.
        const bool is1D = (shape[0] == 1 || shape[1] == 1);

        // Initialise the frame counter. This number labels the successive plots
        // which are saved and might be different from @c step for @c stride 
        // differente from 1. 
        unsigned iFrame = 0;

        // Loop the filter coefficient update steps.
        for (unsigned step : steps) {

            FCTINFO("Step: %d", step);

            // Get the filter coefficients at this step.
            auto filter = filterLog.at(step);
            wn.setFilter(filter);

            if (is1D) {
                // Plotting 1D basis functions.

                // Initialise number of functions to draw.
                const unsigned numFunctions = dim * 2;

                // Initialise TLine for drawing guidling lines.
                TLine line;
                line.SetLineColor(kGray + 2);
                line.SetLineStyle(3);
                
                // Loop basis functions to draw.
                int icol = 0;
                int irow = -1;
                for (unsigned i = 0; i < numFunctions; i++) {
                    
                    // Get the correct row and column indices.
                    if (icol + 1 == pow(2, std::max(irow - 1, 0))) {
                        icol = 0;
                        irow++;
                    } else {
                        icol++;
                    }

                    // Point to the correct pad.
                    pads[icol][irow]->cd();

                    // Initialise basis function histogram.
                    std::unique_ptr<TH1> basisFct = wavenet::MatrixToHist(wn.basisFunction(numFunctions, 1, i, 0), range);

                    // Perform styling.
                    basisFct->GetYaxis()->SetRangeUser(-ymax, ymax);

                    basisFct->SetLineColor(colours[irow]);
                    if      (dim <= 4) { basisFct->SetLineWidth(3); }
                    else if (dim <= 8) { basisFct->SetLineWidth(2); }

                    basisFct->GetXaxis()->SetTickLength(0.);
                    basisFct->GetYaxis()->SetTickLength(0.);
                    basisFct->GetXaxis()->SetTitleOffset(9999.);
                    basisFct->GetYaxis()->SetTitleOffset(9999.);
                    basisFct->GetXaxis()->SetLabelOffset(9999.);
                    basisFct->GetYaxis()->SetLabelOffset(9999.);

                    // Draw.
                    basisFct->DrawCopy("COL");
                    line.DrawLine(-range, 0, range, 0);

                }

            } else {
                // Plotting 2D basis functions.

                // Loop basis functions to draw.
                for (unsigned i = 0; i < dimx; i++) {
                    for (unsigned j = 0; j < dimy; j++) {
                        
                        // Point to the correct pad.
                        pads[i][j]->cd();

                        // Initialise basis function histogram.
                        std::unique_ptr<TH1> basisFct = wavenet::MatrixToHist(wn.basisFunction(shape[0], shape[1], i, j), range);
                        
                        // Perform styling.
                        basisFct->GetZaxis()->SetRangeUser(-zmax, zmax);
                        basisFct->SetContour(nb);
                        
                        basisFct->GetXaxis()->SetTickLength(0.);
                        basisFct->GetYaxis()->SetTickLength(0.);
                        basisFct->GetXaxis()->SetTitleOffset(9999.);
                        basisFct->GetYaxis()->SetTitleOffset(9999.);
                        basisFct->GetXaxis()->SetLabelOffset(9999.);
                        basisFct->GetYaxis()->SetLabelOffset(9999.);
                        
                        // Draw.
                        basisFct->DrawCopy("COL");
                        
                    }
                }

            } // end: basis dimension
            
            // Calculate proper offset and padding for text lines.
            double offset = 1. / (1./topMargin + 1.);
            double paddingx = marg / dimfx;
            double paddingy = marg / dimfy / 2. + 0.005;

            // Initialise TLatex object for writing text on canvas.
            TLatex text;
            text.SetTextFont(42);
            text.SetTextSize(textSize);
            c.cd();

            // Draw information about run configuration.
            if (iFrame == 0) {
                text.DrawLatexNDC(paddingx, 1 - offset * 0.5 + paddingy, "#font[62]{Wavenet}  |  NeedleGenerator  |  2 coeffs.");
            }

            // Create overlay TPad, to draw step-specific information.
            TPad textFrame ("textFrame", "", 0., 0., 1., 1.);
            textFrame.SetFillStyle(0);
            c.cd();
            textFrame.Draw();
            textFrame.cd();

            // Draw step progress information.
            text.SetTextAlign(31);
            text.DrawLatexNDC(1. - paddingx, 1 - offset * 0.5 + paddingy, ("Update " + std::to_string(step + 1) + "/" + std::to_string(numSteps)).c_str());

            // Compute regularisaton (R), sparsity (S), and combined (J) costs
            // at the current step.
            double R = wavenet::RegTerm(wn.filter());
            double J = costLog.at(step);
            double S = J - R;

            // Draw cost information.
            text.SetTextAlign(11);
            text.DrawLatexNDC(paddingx, 1 - offset + paddingy, ("Cost: Regularisation = " + wavenet::formatNumber(R,2,false,2) + ", sparsity = " + wavenet::formatNumber(S,2,false,2) + ", combined = " + wavenet::formatNumber(J,2,false,2)).c_str());
            textFrame.Update();

            // Make sure that the output directory exists.
            coach.checkMakeOutdir("movie");

            // Save plot.
            char buff[100];
            std::string savename;

            // Save plot as PDF.
            snprintf(buff, sizeof(buff), (coach.outdir() + "movie/bestBasis_%dD_%06d.pdf").c_str(), (is1D ? 1 : 2), iFrame); // iCoeff);
            savename = buff;
            c.SaveAs(savename.c_str());

            // Save plot as PNG.
            snprintf(buff, sizeof(buff), (coach.outdir() + "movie/bestBasis_%dD_%06d.png").c_str(), (is1D ? 1 : 2), iFrame); // iCoeff);
            savename = buff;
            c.SaveAs(savename.c_str());

            iFrame++;
        }
    
    } // end: restricted scope

    #endif // USE_ROOT

    FCTINFO("-----------------------------------------------------------");
    FCTINFO("Done.");
    FCTINFO("===========================================================");
    
    return 1;

}
