/**
 * \file
 *
 * Date.hpp
 *
 *  Created on: May 22, 2014
 *      Author: cerovec
 *
 * Copyright (c)20114 Racuni.hr d.o.o. All rights reserved.
 *
 * ANY UNAUTHORIZED USE OR SALE, DUPLICATION, OR DISTRIBUTION
 * OF THIS PROGRAM OR ANY OF ITS PARTS, IN SOURCE OR BINARY FORMS,
 * WITH OR WITHOUT MODIFICATION, WITH THE PURPOSE OF ACQUIRING
 * UNLAWFUL MATERIAL OR ANY OTHER BENEFIT IS PROHIBITED!
 * THIS PROGRAM IS PROTECTED BY COPYRIGHT LAWS AND YOU MAY NOT
 * REVERSE ENGINEER, DECOMPILE, OR DISASSEMBLE IT.
 */
#pragma once

#include <ctime>        // for tm
#include <string>       // for string

class DateTime {

private:

    /**
     * Exact time
     * @brief time_
     */
    tm time_;

    /**
     * Indicates whether this DateTime object is successfully parsed from string.
     * @brief successfullyParsed_
     */
    bool successfullyParsed_;

    /**
     * Original date time string
     */
    std::string originalString_;

    /**
     * Designated constructor.
     *
     * Private. Use factory method which returns a status if creation failed.
     */
    DateTime(tm time);

public:

    /**
     * Creates Now object
     */
    DateTime();

    /**
     * Copy constructor
     */
    DateTime(const DateTime& other);

    /**
     * Assignment operator
     */
    DateTime& operator=(const DateTime& other);

    /**
     * Equality operator
     */
    bool operator==(const DateTime& other) const;

    std::string toString() const;

    /**
     * @brief calculates number of days for specific month
     * @param month
     * @param year
     * @return days in month
     */
    static int daysInMonth(int month, int year);

    /**
     * @brief updates a date with a current day, month and year
     */
    void setToCurrentDate();

    /**
     * @brief updates a date to a given day, month and year
     * @param day
     * @param month
     * @param year
     * @return true if succeded, false otherwise.
     */
    bool setToDMY(int day, int month, int year);

    /**
     * Virtual destructor
     */
    virtual ~DateTime();

    /**
     * @brief getDay
     * @return day in month
     */
    int getDay() const {
        return time_.tm_mday;
    }

    /**
     * @brief getMonth
     * @return Month in year from 1 to 12
     */
    int getMonth() const {
        return time_.tm_mon + 1;
    }

    /**
     * @brief getYear
     * @return Year of the date
     */
    int getYear() const {
        return time_.tm_year + 1900;
    }

    /**
     * @brief getCentury
     * @return century in which the date is
     */
    int getCentury() const {
        return getYear() / 100;
    }

    /**
     * @brief isSuccessfullyParsed
     * @return if this DateTime object is successfully parsed from string returns true, false otherwise.
     */
    bool isSuccessfullyParsed() const {
        return successfullyParsed_;
    }

    /**
     * @brief setOriginalString
     * @param dateString original date string
     */
    void setOriginalString(const std::string& dateString) {
        originalString_ = dateString;
    }

    /**
     * @brief getOriginalString
     * @return Original date string, or empty string if original date string
     * is not set
     */
    const std::string& getOriginalString() const {
        return originalString_;
    }
};

/**
 * @brief Class responsible for parsing date time objects from string
 *
 * Currently supports following values:
 *
 * DD day in the month, two digit
 * MM month, two digits
 * YYYY year, four digits
 *
 * All combinations are allowed, e.g
 * DDMMYYYY
 * YYYYMMDD
 * etc.
 */
class DateTimeFormatter {
private:

    /** Format */
    std::string format_;

public:

    /**
     * @brief DateTimeFormatter
     * @param format
     */
    DateTimeFormatter(const std::string& format) :
            format_(format) {
    }

    /**
     * @brief formats the datetime object to string
     *
     * @param dateTime
     * @return
     */
    std::string format(const DateTime& dateTime);
};
