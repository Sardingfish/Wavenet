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
#include "Wavenet/Wavenet.h"
#include "Wavenet/Coach.h"
#include "Wavenet/Reader.h"
#include "Wavenet/Generators.h"
#include "Wavenet/Utils.h"


int main (int argc, char* argv[]) {
    cout << "Running Wavenet study." << endl;    
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
    
    Wavenet ML;

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
    coach.setWavenet(&ML);
    
    coach.run();
    
    reader.close();
    
    cout << "Done." << endl;
    
    return 1;
}
