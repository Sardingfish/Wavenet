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
#include "Wavenet/Utilities.h"
#include "Wavenet/Logger.h"
#include "Wavenet/Generators.h"
#include "Wavenet/Wavenet.h"
#include "Wavenet/Coach.h"


int main (int argc, char* argv[]) {

    
    FCTINFO("Running Wavenet study.");


    wavenet::GeneratorMode mode = wavenet::GeneratorMode::File;
    int Nfilter = 4;
   
    /* ----- */
    
    // Variables.
    std::string outdir = "./output/";
    
    std::string project = "Run.";
    switch (mode) {
        case wavenet::GeneratorMode::File:
            // ...
            project += "File";
            break;
            
        case wavenet::GeneratorMode::Uniform:
            project += "Uniform";
            break;
            
        case wavenet::GeneratorMode::Needle:
            project += "Needle";
            break;
            
        case wavenet::GeneratorMode::Gaussian:
            project += "Gaussian";
            break;
            
        default:
            FCTINFO("Event mode not recognised. Exiting.");
            return 0;
            break;
    }
    project += ".N" + std::to_string(Nfilter);
    
    wavenet::Wavenet wavenet;

    wavenet.setLambda(10.); // 10.
    wavenet.setAlpha(0.002); // 10 -> 0.01; 100 -> 0.02
    wavenet.setInertia(0.999); // 0.99
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
    
    wavenet::HepMCGenerator generator ("input/Pythia.WpT500._000001.hepmc");
    //wavenet::GaussianGenerator generator;
    //wavenet::NeedleGenerator generator;
    generator.setShape({32,32});

    wavenet::Coach  coach  (project); //("Pythia.WpT500.N16");
    coach.setNumEvents(100000); // (1000); // 25000
    coach.setNumEpochs(1); // 4
    coach.setNumCoeffs(Nfilter);
    coach.setNumInits (5); // (10);
    coach.setUseAdaptiveLearningRate();
    //coach.setUseSimulatedAnnealing();
    //coach.setTargetPrecision(0.0001);
    coach.setGenerator(&generator);
    coach.setWavenet(&wavenet);
    
    coach.run();
    
    generator.close();
    
    FCTINFO("Done.");
    
    return 1;

}

