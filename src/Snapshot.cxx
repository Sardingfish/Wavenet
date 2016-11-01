#include "Wavenet/Snapshot.h"

string Snapshot::file () {
    
    if (_pattern.find("%") == string::npos) { return _pattern; }
    
    char buff[100];
    snprintf(buff, sizeof(buff), _pattern.c_str(), _number);
    string filename = buff;
    return filename;
    
}

void Snapshot::save (WaveletML* ML) {
    
    cout << "Saving snapshot '" << file() << "'." << endl;
    
    if (strcmp(file().substr(0,1).c_str(), "/") == 0) {
        //cout << "WARNING: File '" << file() << "' not accepted. Only accepting realtive paths." << endl;
        WARNING("File '%s' not accepted. Only accepting realtive paths.", file().c_str());
        return;
    }
    
    if (exists()) {
        //cout << "WARNING: File '" << file() << "' already exists. Overwriting." << endl;
        WARNING("File '%s' already exists. Overwriting.", file().c_str());
    }
    
    if (file().find("/") != string::npos) {
        string dir = file().substr(0,file().find_last_of("/")); // ...
        if (!dirExists(dir)) {
            //cout << "WARNING: Directory '" << dir << "' does not exist. Creating it." << endl;
            WARNING("Directory '%s' does not exist. Creating it.", dir.c_str());
            system(("mkdir -p " + dir).c_str());
        }
    }
    
    ofstream outFileStream (file());
    
    outFileStream << ML->_lambda << "\n";
    outFileStream << ML->_alpha << "\n";
    outFileStream << ML->_inertia << "\n";
    outFileStream << ML->_filter << "\n#\n";
    outFileStream << ML->_momentum << "\n#\n";
    
    outFileStream << ML->_batchSize << "\n";
    outFileStream << "BATCHQUEUE" << "\n";
    for (const auto& q : ML->_batchQueue) {
        outFileStream << q << "\n#\n";
    }
    outFileStream << "FILTERLOG" << "\n";
    for (const auto& f : ML->_filterLog) {
        outFileStream << f << "\n#\n";
    }
    outFileStream << "COSTLOG" << "\n";
    for (const auto& f : ML->_costLog) {
        outFileStream << f << "\n";
    }
    
    outFileStream.close();
    
    return;
}

void Snapshot::load (WaveletML* ML) {
    
    INFO("Loading snapshot '%s'.", file().c_str())
    //cout << "Loading snapshot '" << file() << "'." << endl;
    
    if (!fileExists(file())) {
        cout << "WARNING: File '" << file() << "' doesn't exists." << endl;
        return;
    }
    
    ifstream inFileStream (file());
    std::string tmp; // To stream in values and check for delimeters.
    
    inFileStream >> ML->_lambda;
    inFileStream >> ML->_alpha;
    inFileStream >> ML->_inertia;
    
    // Read filter.
    vector<double> vec_filter;
    while (inFileStream >> tmp && tmp.find("#") == string::npos) {
        try {
            vec_filter.push_back( stod(tmp) );
        } catch (const std::invalid_argument& ia) {;}
    }
    ML->_filter = arma::conv_to< arma::Col<double> >::from(vec_filter);
    
    // Read momentum.
    vector<double> vec_momentum;
    while (inFileStream >> tmp && tmp.find("#") == string::npos) {
        try {
            vec_momentum.push_back( stod(tmp) );
        } catch (const std::invalid_argument& ia) {;}
    }
    ML->_momentum = arma::conv_to< arma::Col<double> >::from(vec_momentum);
    
    inFileStream >> ML->_batchSize;
    
    // Read batch queue.
    inFileStream >> tmp;
    ML->_batchQueue.clear();
    while (tmp.find("FILTERLOG") == string::npos) {
        vector<double> vec_momentum;
        while (inFileStream >> tmp) {
            try {
                vec_momentum.push_back( stod(tmp) );
            } catch (const std::invalid_argument& ia) { break; }
        }
        if (!vec_momentum.size()) { break; }
        ML->_batchQueue.push_back( arma::conv_to< arma::Col<double> >::from(vec_momentum) );
    }
    
    // Read filter log.
    ML->_filterLog.clear();
    while (tmp.find("COSTLOG") == string::npos) {
        vector<double> vec_filter;
        while (inFileStream >> tmp) {
            try {
                vec_filter.push_back( stod(tmp) );
            } catch (const std::invalid_argument& ia) { break; }
        }
        if (!vec_filter.size()) { break; }
        ML->_filterLog.push_back( arma::conv_to< arma::Col<double> >::from(vec_filter) );
    }
    
    // Read cost log.
    ML->_costLog.clear();
    while (!inFileStream.fail()) {
        while (inFileStream >> tmp) {
            try {
                ML->_costLog.push_back( stod(tmp) );
            } catch (const std::invalid_argument& ia) { break; }
        }
    }
    
    inFileStream.close();
    
    return;
}