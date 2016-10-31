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

#include "ElapsedTimer.hpp"
#include <math.h>

#include <sstream>

#if !defined _WIN32 && !defined _WIN64
#include <sys/time.h>
#endif

#if (defined PLATFORM_IOS) || (defined TARGET_IPHONE_SIMULATOR && TARGET_IPHONE_SIMULATOR==1) || (defined TARGET_OS_IPHONE && TARGET_OS_IPHONE==1)
#include <mach/mach_time.h>
#endif

ElapsedTimer::ElapsedTimer() {
#if defined _WIN32 || defined _WIN64
    // get ticks per second
    QueryPerformanceFrequency(&frequency_);
#endif
    tic();
}

ElapsedTimer::~ElapsedTimer() {}

void ElapsedTimer::tic() {
#if (defined PLATFORM_IOS) || (defined TARGET_IPHONE_SIMULATOR && TARGET_IPHONE_SIMULATOR==1) || (defined TARGET_OS_IPHONE && TARGET_OS_IPHONE==1)
    refTime_ = mach_absolute_time();
#elif defined _WIN32 || defined _WIN64
    // start timer
    QueryPerformanceCounter(&refTime_);
#else
    struct timeval start;
    gettimeofday(&start, NULL);
    refTime_ = (double)start.tv_sec + ((double)start.tv_usec / 1000000.0);
#endif
}

double ElapsedTimer::toc() const {
    double duration;

#if (defined PLATFORM_IOS) || (defined TARGET_IPHONE_SIMULATOR && TARGET_IPHONE_SIMULATOR==1) || (defined TARGET_OS_IPHONE && TARGET_OS_IPHONE==1)
    /* Get the timebase info */
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);

    uint64_t timediff = mach_absolute_time() - refTime_;

    duration = timediff * info.numer;
    duration /= info.denom;
    duration /= 1000000;
#elif defined _WIN32 || defined _WIN64
    LARGE_INTEGER timestamp;
    QueryPerformanceCounter(&timestamp);
    duration = (timestamp.QuadPart - refTime_.QuadPart) * 1000.0 / frequency_.QuadPart;
#else
    struct timeval end;
    gettimeofday(&end, NULL);

    double startV, endV;
    startV = refTime_;
    endV = (double)end.tv_sec + ((double)end.tv_usec / 1000000.0);
    duration = (endV - startV) * 1000;
#endif
    return duration;
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

