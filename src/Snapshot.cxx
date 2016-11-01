#include "Wavenet/Snapshot.h"

std::string Snapshot::file () {
    
    if (_pattern.find("%") == std::string::npos) { return _pattern; }
    
    char buff[100];
    snprintf(buff, sizeof(buff), _pattern.c_str(), _number);
    std::string filename = buff;
    return filename;
    
}

void Snapshot::save (Wavenet* wavenet) {
    
    INFO("Saving snapshot '%s'.", file().c_str());
    
    if (strcmp(file().substr(0,1).c_str(), "/") == 0) {
        WARNING("File '%s' not accepted. Only accepting realtive paths.", file().c_str());
        return;
    }
    
    if (exists()) {
        WARNING("File '%s' already exists. Overwriting.", file().c_str());
    }
    
    if (file().find("/") != std::string::npos) {
        std::string dir = file().substr(0,file().find_last_of("/")); // ...
        if (!dirExists(dir)) {
            WARNING("Directory '%s' does not exist. Creating it.", dir.c_str());
            system(("mkdir -p " + dir).c_str());
        }
    }
    
    ofstream outFileStream (file());
    
    outFileStream << wavenet->_lambda << "\n";
    outFileStream << wavenet->_alpha << "\n";
    outFileStream << wavenet->_inertia << "\n";
    outFileStream << wavenet->_filter << "\n#\n";
    outFileStream << wavenet->_momentum << "\n#\n";
    
    outFileStream << wavenet->_batchSize << "\n";
    outFileStream << "BATCHQUEUE" << "\n";
    for (const auto& q : wavenet->_batchQueue) {
        outFileStream << q << "\n#\n";
    }
    outFileStream << "FILTERLOG" << "\n";
    for (const auto& f : wavenet->_filterLog) {
        outFileStream << f << "\n#\n";
    }
    outFileStream << "COSTLOG" << "\n";
    for (const auto& f : wavenet->_costLog) {
        outFileStream << f << "\n";
    }
    
    outFileStream.close();
    
    return;
}

void Snapshot::load (Wavenet* wavenet) {
    
    INFO("Loading snapshot '%s'.", file().c_str())
    
    if (!fileExists(file())) {
        WARNING("File '%s' doesn't exists.", file().c_str());
        return;
    }
    
    ifstream inFileStream (file());
    std::string tmp; // To stream in values and check for delimeters.
    
    inFileStream >> wavenet->_lambda;
    inFileStream >> wavenet->_alpha;
    inFileStream >> wavenet->_inertia;
    
    // Read filter.
    std::vector<double> vec_filter;
    while (inFileStream >> tmp && tmp.find("#") == std::string::npos) {
        try {
            vec_filter.push_back( stod(tmp) );
        } catch (const std::invalid_argument& ia) {;}
    }
    wavenet->_filter = arma::conv_to< arma::Col<double> >::from(vec_filter);
    
    // Read momentum.
    std::vector<double> vec_momentum;
    while (inFileStream >> tmp && tmp.find("#") == std::string::npos) {
        try {
            vec_momentum.push_back( stod(tmp) );
        } catch (const std::invalid_argument& ia) {;}
    }
    wavenet->_momentum = arma::conv_to< arma::Col<double> >::from(vec_momentum);
    
    inFileStream >> wavenet->_batchSize;
    
    // Read batch queue.
    inFileStream >> tmp;
    wavenet->_batchQueue.clear();
    while (tmp.find("FILTERLOG") == std::string::npos) {
        std::vector<double> vec_momentum;
        while (inFileStream >> tmp) {
            try {
                vec_momentum.push_back( stod(tmp) );
            } catch (const std::invalid_argument& ia) { break; }
        }
        if (!vec_momentum.size()) { break; }
        wavenet->_batchQueue.push_back( arma::conv_to< arma::Col<double> >::from(vec_momentum) );
    }
    
    // Read filter log.
    wavenet->_filterLog.clear();
    while (tmp.find("COSTLOG") == std::string::npos) {
        std::vector<double> vec_filter;
        while (inFileStream >> tmp) {
            try {
                vec_filter.push_back( stod(tmp) );
            } catch (const std::invalid_argument& ia) { break; }
        }
        if (!vec_filter.size()) { break; }
        wavenet->_filterLog.push_back( arma::conv_to< arma::Col<double> >::from(vec_filter) );
    }
    
    // Read cost log.
    wavenet->_costLog.clear();
    while (!inFileStream.fail()) {
        while (inFileStream >> tmp) {
            try {
                wavenet->_costLog.push_back( stod(tmp) );
            } catch (const std::invalid_argument& ia) { break; }
        }
    }
    
    inFileStream.close();
    
    return;
}