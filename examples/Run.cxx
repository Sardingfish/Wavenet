// STL include(s).
#include <string> /* std::string */
#include <vector> /* std::vector */
#include <cmath> /* log2 */
#include <chrono>

// ROOT include(s).
#include "TStyle.h"
#include "TCanvas.h"
#include "TH2.h"
#include "TGraph.h"
#include "TMarker.h"
#include "TEllipse.h"

// Wavenet include(s).
#include "Wavenet/Utils.h"
#include "Wavenet/Logger.h"
#include "Wavenet/Generators.h"
#include "Wavenet/Wavenet.h"
#include "Wavenet/Coach.h"


int main (int argc, char* argv[]) {
    FCTINFO("Running Wavenet study.");
    /*
    arma::Mat<double> M;

    //HepMCGenerator hg;
    GaussianGenerator hg;
    //hg.open("input/Pythia.WpT500._000001.hepmc");
    hg.setShape({64,16});
    

    M = hg.next();
 
    TCanvas c1 ("c1", "", 600, 600);
    TH1* h1 = MatrixToHist(M, 3.2);
    h1->Draw("COL Z");
    c1.SaveAs("TEMPTEMPTEMP.pdf");
    
    return 0;
    */



    Wavenet::GeneratorMode mode = Wavenet::GeneratorMode::Gaussian;
    int Nfilter = 8;
   
    /* ----- */
    
    // Variables.
    std::string outdir = "./output/";
    
    std::string project = "Run.";
    switch (mode) {
        case Wavenet::GeneratorMode::File:
            // ...
            project += "File";
            break;
            
        case Wavenet::GeneratorMode::Uniform:
            project += "Uniform";
            break;
            
        case Wavenet::GeneratorMode::Needle:
            project += "Needle";
            break;
            
        case Wavenet::GeneratorMode::Gaussian:
            project += "Gaussian";
            break;
            
        default:
            FCTINFO("Event mode not recognised. Exiting.");
            return 0;
            break;
    }
    project += ".N" + std::to_string(Nfilter);
    
    Wavenet::Wavenet wavenet;

    wavenet.setLambda(10.);
    wavenet.setAlpha(0.002); // 10 -> 0.01; 100 -> 0.02
    wavenet.setInertia(0.9);
    wavenet.setInertiaTimeScale(20.);
    wavenet.setBatchSize(20);
    wavenet.doWavelet(true); // >>> Default: true:
    
    wavenet.print();
    
    // * Get cost map(s).
    /*
    arma::Mat<double> costMap;
    std::string costMapName = "costMap.mat";
    if (!fileExists(costMapName)) {
        arma::field< arma::Mat<double> > costs = ML.costMap(X, 1.2, 300);
        costMap = costs(0,0);
        costMap.save(costMapName);
    } else {
        costMap.load(costMapName);
    }
    */
    
 
    // Coached training.
    /*
    Reader reader;
    reader.setGeneratorMode(mode);
    if (mode == GeneratorMode::File) {
        bool stat = reader.open("input/Pythia.WpT500._000001.hepmc");
        if (!stat) { return 1; }
    }
    reader.setSize(64);
    */
    
    //HepMCGenerator generator ("input/Pythia.WpT500._000001.hepmc");
    Wavenet::GaussianGenerator generator;
    generator.setShape({32,32});

    Wavenet::Coach  coach  (project); //("Pythia.WpT500.N16");
    coach.setNevents(10000); // (1000); // 25000
    coach.setNepochs(1); // 4
    coach.setNcoeffs(Nfilter);
    coach.setNinits (10); // (10);
    //coach.setUseAdaptiveLearning(true);
    coach.setUseAdaGrad(true);
    coach.setGenerator(&generator);
    coach.setWavenet(&wavenet);
    
    coach.run();
    
    generator.close();
    
    FCTINFO("Done.");
    
    return 1;
}
