// STL include(s).
#include <string>
#include <vector>

// ROOT include(s).
#include "TStyle.h"
#include "TCanvas.h"
#include "TH2.h"
#include "TGraph.h"
#include "TMarker.h"
#include "TEllipse.h"

// WaveletML include(s).
#include "Wavenet/WaveletML.h"
#include "Wavenet/Coach.h"
#include "Wavenet/Reader.h"

int main (int argc, char* argv[]) {
    cout << "Running WaveletML study." << endl;
    
    
    EventMode mode = EventMode::File;
    int Nfilter = 4;
   
    /* ----- */
    
    // Variables.
    std::string outdir = "./output/";
    
    string project = "Run.";
    switch (mode) {
        case EventMode::File:
            // ...
            project += "File";
            break;
            
        case EventMode::Uniform:
            project += "Uniform";
            break;
            
        case EventMode::Needle:
            project += "Needle";
            break;
            
        case EventMode::Gaussian:
            project += "Gaussian";
            break;
            
        default:
            cout << "Event mode not recognised." << endl;
            return 0;
            break;
    }
    project += ".N" + to_string(Nfilter);
    
    WaveletML ML;

    ML.setLambda(10.);
    ML.setAlpha(0.001); // 10 -> 0.01; 100 -> 0.02
    ML.setInertia(0.0);
    ML.setBatchSize(10);
    ML.doWavelet(true); // >>> Default: true:
    
    ML.print();
    
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
    Reader reader;
    reader.setEventMode(mode);
    if (mode == EventMode::File) {
        bool stat = reader.open("input/Pythia.WpT500._000001.hepmc");
        if (!stat) { return 1; }
    }
    reader.setSize(64);
    
    Coach  coach  (project); //("Pythia.WpT500.N16");
    coach.setNevents(100); // (1000); // 25000
    coach.setNepochs(5  ); // 4
    coach.setNcoeffs(Nfilter);
    coach.setNinits (2); // (10);
    coach.setUseAdaptiveLearning(true);
    coach.setReader(&reader);
    coach.setWaveletML(&ML);
    
    coach.run();
    
    reader.close();
    
    cout << "Done." << endl;
    
    return 1;
}
