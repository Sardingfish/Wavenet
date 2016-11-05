// STL include(s).
#include <string> /* std::string */
#include <vector> /* std::vector */

#ifdef USE_ROOT
// ROOT include(s).
#include "TGraph.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TEllipse.h"
//#include "TLine.h"
#include "TMarker.h"
#include "TLatex.h"
#include "TColor.h"
#endif // USE_ROOT

// Wavenet include(s).
#include "Wavenet/Logger.h" /* FCTINFO, FCTWARNING */
#include "Wavenet/Generators.h" /* wavenet::NeedleGenerator */
#include "Wavenet/Wavenet.h" /* wavenet::Wavenet */
#include "Wavenet/Coach.h" /* wavevent::Coach */

/**
 * Example02: Find the best wavelet basis with 2 filter coefficients for needle-like input.
 *
 * Requirements: ROOT
 *
 * This first example shows a minimal setup, where ...
 *
 *
 */
int main (int argc, char* argv[]) {

    FCTINFO("===========================================================");
    FCTINFO("Running Wavenet Example01.");
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

    // Set number of initialisations, i.e. runs with different initial conditions. Default value is 1.
    const unsigned int numInits = 20;

    // Create a Generator instance, here 'NeedleGenerator', and specify shape.
    wavenet::NeedleGenerator ng;
    ng.setShape({16,16});
    
    // Create a 'Wavenet' instance.
    // For this example, we're tuning the settings slightly cf. the default values.
    wavenet::Wavenet wn;
    wn.setAlpha(0.001);  // Learning rate. Default value is 0.01
    wn.setBatchSize(10); // Batch size, i.e. the number of input examples used in each batch of the SGD. Default value is 1.
    

    // Create a 'Coach' instance.
    // For this example, we're also tuning the settings slightly.
    wavenet::Coach coach (name);
    coach.setNumCoeffs(numCoeffs);
    coach.setGenerator(&ng);
    coach.setWavenet  (&wn);

    coach.setNumEvents(2000); // Number of events used in each "epoch". Default value is 1000.
    coach.setNumInits (numInits);
    
    // Run the training.
    bool good = coach.run();

    if (!good) {
        FCTWARNING("Uhh-oh! Something went wrong.");
        return 0;
    }

    // Draw the cost map.
    
    // ...
    const bool overwrite = false;
    arma::Mat<double> costMap, costMapSparsity, costMapRegularisation;
    std::string costMapName               = coach.outdir() + "costMap.mat";
    std::string costMapSparsityName       = coach.outdir() + "costMapSparsity.mat";
    std::string costMapRegularisationName = coach.outdir() + "costMapRegularisation.mat";
    
    if (!wavenet::fileExists(costMapName) or overwrite) {
        // If the cost map files do not already exist, we need to produce them.
        // This is done by...

        // Examples...
        const unsigned int numExamples = 10;
        std::vector< arma::Mat<double> > examples;
        for (unsigned i = 0; i < numExamples; i++) {
            // Append example signal with which to compute cost map.
            examples.push_back( ng.next() );

            #ifdef USE_ROOT
            // Draw example inputs to PDF.
            TCanvas cExample ("cExample", "", 700, 600);
            std::unique_ptr<TH1> exampleInput = wavenet::MatrixToHist(examples.at(i), 3.2);
            exampleInput->Draw("COL Z");
            cExample.SaveAs((coach.outdir() + "exampleInput." + std::to_string(i + 1) +".pdf").c_str());
            #endif // USE_ROOT
        }
    

        // Define...
        arma::field< arma::Mat<double> > costs;
        
        costs = wn.costMap(examples, 1.2, 300);

        costs(0,0).save(costMapName);
        costs(1,0).save(costMapSparsityName);
        costs(2,0).save(costMapRegularisationName);
        
        costMap = costs(0,0);

    } else {
        // If the cost maps have already been produced, just load them from file.
        costMap              .load(costMapName);
        costMapSparsity      .load(costMapSparsityName);
        costMapRegularisation.load(costMapRegularisationName);
        
    }

    // Initialise the 'Snapshot' object that we're going to use to read the results of the optimisation.
    // The snapshots for each initialisation above is stored 'snapshots/' subdirectory in the base directory of the responsible Coach instance (by default './output/')
    // If we specify a pattern (by the %-bit in the string), the snapshot can automatically iterate though successivle snapshots.
    // The format chosen below is the one used by default when the Coach writes the snapshots to file.
    const std::string pattern = coach.outdir() + "snapshots/" +  name + ".%06u.snap";
    wavenet::Snapshot snap (pattern);

    #ifdef USE_ROOT

    // Create one ROOT TGraph, showing the evolution of the filter coefficients, for each initialisation.
    std::vector< TGraph > filterGraphs (numInits);
    while (snap.exists() and snap.number() < numInits) {
        
        // Load the Wavenet configuration from the current snapshot.
        wn.load(snap.file());

        // Get the filter log from training.
        auto filterLog = wn.filterLog();
        
        // Create filter graph for this initialisation.
        const unsigned int numSteps = filterLog.size();
        double x[numSteps], y[numSteps];
        for (unsigned i = 0; i < numSteps; i++) {
            x[i] = arma::as_scalar(filterLog.at(i).row(0));
            y[i] = arma::as_scalar(filterLog.at(i).row(1));
        }
        
        // Store it in the vector.
        filterGraphs.at(snap.number()) = TGraph(numSteps, x, y);
        
        // Next.
        snap++;   
    }

    // Define colour palette.
    int kMyRed = 1756; // color index
    TColor *MyRed =  new TColor(kMyRed,  224./255.,   0./255.,  42./255.);
    int kMyBlue = 1757;
    TColor *MyBlue = new TColor(kMyBlue,   3./255.,  29./255.,  66./255.);
    
    const int Number = 2; 
    double Red[Number]    =  {   3./255., 0.98 }; 
    double Green[Number]  =  {  29./255., 0.98 }; 
    double Blue[Number]   =  {  66./255., 0.98 }; 
    double Length[Number] =  { 0.50, 1.00 }; 
    int nb = 104;
    TColor::CreateGradientColorTable(Number, Length, Red, Green, Blue, nb);
    
    // Define ROOT TCanvas.
    TCanvas c ("c", "", 700, 600);
    c.SetLogz(true);
    std::unique_ptr<TH1> J = wavenet::MatrixToHist(costMap, 1.2);
    J->SetContour(104); 
    gStyle->SetOptStat(0);
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
    for (unsigned m = 0; m < numInits; m++) {
        // Filter evolution graph.
        filterGraphs.at(m).Draw("L same");

        // Draw markers: Red dots are the initial filter coefficients; blue dots are the final ones.
        TMarker marker;
        double x, y;

        // -- Initial marker.
        marker.SetMarkerColor(kRed);
        marker.SetMarkerStyle(8);
        marker.SetMarkerSize (0.3);
        filterGraphs.at(m).GetPoint(0, x, y);
        marker.DrawMarker(x, y);
        
        // -- Final marker.
        marker.SetMarkerColor(kBlue);
        marker.SetMarkerStyle(19);
        marker.SetMarkerSize (0.3);
        filterGraphs.at(m).GetPoint(filterGraphs.at(m).GetN() - 1, x, y);
        marker.DrawMarker(x, y);
    }
    
    
    // Save the final cost map as PDF.
    c.SaveAs((coach.outdir() + "CostMap.pdf").c_str());

    # endif // USE_ROOT

    FCTINFO("-----------------------------------------------------------");
    FCTINFO("Done.");
    FCTINFO("===========================================================");
    
    return 1;

}

