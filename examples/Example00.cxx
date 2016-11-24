/**
 * @file   Example00.cxx
 * @author Andreas Sogaard
 * @date   24 November 2016
 * @brief  Minimal working example.
 */

 // Wavenet include(s).
#include "Wavenet/Generators.h" /* wavenet::NeedleGenerator */
#include "Wavenet/Wavenet.h" /* wavenet::Wavenet */
#include "Wavenet/Coach.h" /* wavevent::Coach */

/**
 * Example00: Minimal working example. See Example01 for more explanations.
 */
int main (int argc, char* argv[]) {

    // Create Generator instance; here NeedleGenerator.
    wavenet::NeedleGenerator ng;
    ng.setShape({16,16});
    
    // Create Wavenet instance.
    wavenet::Wavenet wn;

    // Create Coach instance.
    wavenet::Coach coach ("Example00");
    coach.setNumCoeffs(2);
    coach.setGenerator(&ng);
    coach.setWavenet  (&wn);
    
    // Run the training.
    coach.run();

    return 1;
}
