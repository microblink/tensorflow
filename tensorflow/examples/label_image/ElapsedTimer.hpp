/**
 * \file
 *
 * ElapsedTimer.hpp
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

#pragma once

#include <string>

#if defined _WIN32 || defined _WIN64
#include <windows.h>
#endif

class ElapsedTimer {
public:
    ElapsedTimer();
    virtual ~ElapsedTimer();

    /**
     * Remembers the current time.
     */
    void tic();

    /**
     * Returns the time elapsed from last remembered time (last tic call).
     * @return Elapsed time in miliseconds.
     */
    double toc() const;
    
    static std::string generateHumanReadableInterval(double ms);
private:
#if (defined PLATFORM_IOS) || (defined TARGET_IPHONE_SIMULATOR && TARGET_IPHONE_SIMULATOR==1) || (defined TARGET_OS_IPHONE && TARGET_OS_IPHONE==1)
    uint64_t refTime_;
#elif defined _WIN32 || defined _WIN64
    LARGE_INTEGER refTime_; // ticks at reference time
    LARGE_INTEGER frequency_; // ticks per second
#else
    double refTime_;
#endif
};

