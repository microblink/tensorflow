/**
 * \file
 *
 * ElapsedTimer.cpp
 *
 *  Created on: Feb 22, 2013
 *      Author: dodo
 *
 * Copyright (c)2015 MicroBlink Ltd. All rights reserved.
 *
 * ANY UNAUTHORIZED USE OR SALE, DUPLICATION, OR DISTRIBUTION
 * OF THIS PROGRAM OR ANY OF ITS PARTS, IN SOURCE OR BINARY FORMS,
 * WITH OR WITHOUT MODIFICATION, WITH THE PURPOSE OF ACQUIRING
 * UNLAWFUL MATERIAL OR ANY OTHER BENEFIT IS PROHIBITED!
 * THIS PROGRAM IS PROTECTED BY COPYRIGHT LAWS AND YOU MAY NOT
 * REVERSE ENGINEER, DECOMPILE, OR DISASSEMBLE IT.
 */

#include <math.h>      // for floor, modf
#include <cstdint>     // for timeval
#include <sstream>     // for basic_ostream, operator<<, stringstream, string

#include "ElapsedTimer.hpp"


ElapsedTimer::ElapsedTimer() {
    tic();
}

ElapsedTimer::~ElapsedTimer() {}

void ElapsedTimer::tic() {
    ref_time_ = std::chrono::high_resolution_clock::now();
}

double ElapsedTimer::toc() const {
    auto timestamp = std::chrono::high_resolution_clock::now();

    auto duration = timestamp - ref_time_;

    return static_cast< double >( std::chrono::duration_cast< std::chrono::nanoseconds >( duration ).count() ) / 1e6;
}

std::string ElapsedTimer::generateHumanReadableInterval(double ms) {
    double seconds = ms/1000.0;
    int iSec = (int)floor(seconds);
    int iMin = iSec / 60;
    int finSecs = iSec % 60;
    int finHrs = iMin / 60;
    int finMinutes = iMin % 60;
    
    std::stringstream ss;
    if (finHrs>0) {
    ss <<  finHrs << " h ";
    }
    if (finMinutes>0) {
    ss << finMinutes << " min ";
    }
    double intpart;
    double desSec = modf(seconds, &intpart);
    double finSecsWithMs = finSecs + desSec;
    ss << finSecsWithMs << " s";
    return ss.str();
}

