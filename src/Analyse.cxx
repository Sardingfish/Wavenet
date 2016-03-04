// STL include(s).
#include <string>
#include <vector>
#include <regex>

// ROOT include(s).
#include "TStyle.h"
#include "TCanvas.h"
#include "TH2.h"
#include "TGraph.h"
#include "TMarker.h"
#include "TEllipse.h"

// WaveletML include(s).
#include "WaveletML.h"
#include "Snapshot.h"
#include "Reader.h"

using namespace std;
using namespace arma;

int main (int argc, char* argv[]) {
    
    cout << "Running WaveletML analysis." << endl;

    EventMode mode = EventMode::Uniform;
    const int M = 10;
    int Nfilter = 4;
    
    /* --- */
    
    WaveletML ML;
    
    // Variables.
    string outdir  = "./output/";
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
    
    outdir += project + "/";
    
    string pattern = outdir + "snapshots/" +  project + ".%06u.snap";
    
    Snapshot snap (pattern);
    
    vector< TGraph > costGraphs (M);
    vector< TGraph > filterGraphs (M);
    TCanvas c ("c", "", 600, 600);
    
    ML.save(outdir + "tmp.snap");
    
    Reader reader;
    reader.setEventMode(mode);
    //reader.open("input/Pythia.WpT500._000001.hepmc");
    
    vector< Mat<double> > examples;
    for (unsigned i = 0; i < 50; i++) {
        examples.push_back( reader.next() );
        TH2F exampleSignal = MatrixToHist(examples.at(i), 3.2);
        exampleSignal.Draw("COL Z");
        c.SaveAs((outdir + "exampleSignal." + to_string(i + 1) +"." + project + ".pdf").c_str());
    }
    reader.close();


    // * Run    
    double minCost = 99999., maxCost = 0;
    
    int bestBasis = 0;
    
    unsigned longestCost = 0;
    while (snap.exists() && snap.number() <= M) {
        
        ML.load(snap.file());
        ML.print();
        
        auto filterLog = ML.filterLog();
        auto costLog   = ML.costLog();
        
        costLog.pop_back();
        
        if (costLog.size() > longestCost) {
            longestCost = costLog.size();
        }
        const unsigned Ncoeffs = filterLog.size();
        
        // Cost graphs.
        costGraphs.at(snap.number() - 1) = ML.getCostGraph( costLog );
        
        double tmpMin = costGraphs.at(snap.number() - 1).GetMinimum();
        double tmpMax = costGraphs.at(snap.number() - 1).GetMaximum();
        
        if (costLog.at(costLog.size() - 2) < minCost) {
            bestBasis = snap.number() - 1;
            minCost = costLog.at(costLog.size() - 2);
        }
        
        maxCost = (tmpMax > maxCost && tmpMax > 0 ? tmpMax : maxCost);
        //minCost = (tmpMin < minCost               ? tmpMin : minCost);
        
        
        // Filter graphs.
        double x[Ncoeffs], y[Ncoeffs];
        for (unsigned i = 0; i < Ncoeffs; i++) {
            x[i] = arma::as_scalar(filterLog.at(i).row(0));
            y[i] = arma::as_scalar(filterLog.at(i).row(1));
        }
        
        filterGraphs.at(snap.number() - 1) = TGraph(Ncoeffs, x, y);
        
        // Next.
        snap++;
        
    }
    
    cout << "Longest costlog = " << longestCost << endl;
    
    c.SetLogy(true);
    for (unsigned m = 0; m < M; m++) {
        if (costGraphs.at(m).GetN() == longestCost) {
            costGraphs.at(m).Draw("LAXIS");
            c.Update();
        }
    }
    
    for (unsigned m = 0; m < M; m++) {
        /*
        if (m == 0) {
            costGraphs.at(m).Draw("LAXIS");
            c.Update();
        }
         */
        //costGraphs.at(m).GetXaxis()->SetRangeUser(0, 200); //maxCost);
        costGraphs.at(m).GetYaxis()->SetRangeUser(0.3, 0.55); // Uniform : (0.4, 0.7) | Needle: (0.0, 0.5)
        costGraphs.at(m).SetLineColor(20 + m % 30);
        costGraphs.at(m).SetLineStyle(1);
        //costGraphs.at(m).Draw(m == 0 ? "LAXIS" : "L same");
        costGraphs.at(m).Draw("L same");
        c.Update();
    }
    c.RangeAxis(0,0.35,200,0.65);
    c.Update();
    c.SaveAs((outdir + "CostGraph" + ".pdf").c_str());
    
    
    // * Get cost map(s).
    c.SetLogy(false);
    arma::Mat<double> costMap;
    std::regex re (".N(\\d+)");
    std::string costMapName = "output/costMap." + std::regex_replace(project, re, "") + ".mat";

    if (!fileExists(costMapName)) {
        arma::field< arma::Mat<double> > costs;
        
        costs = ML.costMap(examples, 1.2, 300);
        costMap = costs(0,0);
        
        costMap.save(costMapName);
    } else {
        costMap.load(costMapName);
    }
    
    
    c.SetLogz(true);
    TH2F J = MatrixToHist(costMap, 1.2);
    J.SetContour(30); // (104);
    gStyle->SetOptStat(0);
    J.SetMaximum(1.);
    J.Draw("CONT1 Z"); // COL / CONT1
    c.Update();
    TMarker marker;
    
    for (unsigned m = 0; m < M; m++) {
        filterGraphs.at(m).Draw("L same");

        // * Marker
        double x, y;
        marker.SetMarkerColor(kRed);
        marker.SetMarkerStyle(8);
        marker.SetMarkerSize (0.3);
        filterGraphs.at(m).GetPoint(0, x, y);
        marker.DrawMarker(x, y);
        
        marker.SetMarkerColor(kBlue);
        marker.SetMarkerStyle(19);
        marker.SetMarkerSize (0.3);
        filterGraphs.at(m).GetPoint(filterGraphs.at(m).GetN() - 1, x, y);
        marker.DrawMarker(x, y);
    }

    c.SaveAs((outdir + "CostMap.pdf").c_str());
    c.SetLogz(false);
    
    // * Basis function.
    snap.setNumber(bestBasis + 1);
    ML.load(snap.file());
    TH2F basisFct;
    
    basisFct = MatrixToHist(ML.basisFct(64, 0, 0), 3.2);
    basisFct.Draw("COL Z");
    c.SaveAs((outdir + "BestBasisFunction-0-0.pdf").c_str());
    
    basisFct = MatrixToHist(ML.basisFct(64, 1, 1), 3.2);
    basisFct.Draw("COL Z");
    c.SaveAs((outdir + "BestBasisFunction-1-1.pdf").c_str());

    basisFct = MatrixToHist(ML.basisFct(64, 2, 2), 3.2);
    basisFct.Draw("COL Z");
    c.SaveAs((outdir + "BestBasisFunction-2-2.pdf").c_str());
    
    basisFct = MatrixToHist(ML.basisFct(64, 6, 6), 3.2);
    basisFct.Draw("COL Z");
    c.SaveAs((outdir + "BestBasisFunction-6-6.pdf").c_str());
    
    basisFct = MatrixToHist(ML.basisFct(64, 6, 40), 3.2);
    basisFct.Draw("COL Z");
    c.SaveAs((outdir + "BestBasisFunction-6-40.pdf").c_str());

    basisFct = MatrixToHist(ML.basisFct(64, 40, 40), 3.2);
    basisFct.Draw("COL Z");
    c.SaveAs((outdir + "BestBasisFunction-40-40.pdf").c_str());

    cout << "Done." << endl;
    
    cout << "Checking orthonormality (best snap):" << endl;
    TH1F norms ("norms", "", 200, -0.5, 1.5);
    const unsigned size = 16;
    for (unsigned i = 0; i < sq(size); i++) {
        for (unsigned j = 0; j < sq(size); j++) {
            Mat<double> f1 = ML.basisFct(size, i % size, i / size);
            Mat<double> f2 = ML.basisFct(size, j % size, j / size);
            double norm = trace(f1*f2.t());
            norms.Fill( norm < -0.5 ? -0.499 : (norm > 1.5 ? 1.499 : norm) );
        }
    }
    
    c.SetLogy(true);
    norms.Draw("HIST");
    c.SaveAs((outdir + "NormDistributions.pdf").c_str());
    
    return 1;
}

