#include <string.h>         // for strlen, strncmp
#include <cstdint>          // for time_t
#include <ostream>          // for string, basic_ostream, stringstream, operator<<
#include <string>           // for basic_string<>::value_type
#include <sstream>

#include "DateTime.hpp"

#define MONTH_NUM 12

namespace {

/**
 * @brief FMT_SPECIAL special character in date format that is used to escape the next char and define day
 * month and year positions (eg. %d for day)
 */
const char FMT_SPECIAL = '%';

const char* const monthsShort[MONTH_NUM] {"jan", "feb", "mar", "apr", "may", "june", "july", "aug", "sep", "oct", "nov", "dec"};
const char* const months[MONTH_NUM] {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"};

/**
 * @brief extractNumFromStr extracts the number from the string at the given position
 * @param str dateString
 * @param beginPos position in dateString where number starts
 * @param maxChars max allowed chars for number
 * @param endPos end position of the extracted number in dateString (actually it is the position of the next char after last digit of the number)
 * @return extracted number or 0 if number can not be parsed
 */
int extractNumFromStr(const std::string& str, size_t beginPos, size_t maxChars, size_t& endPos) {
    int num = 0;
    for (endPos = beginPos; endPos < str.length() && endPos < beginPos + maxChars; endPos++) {
        char c = str[endPos];
        if (c >= '0' && c <= '9') {
            num = num * 10 + c - '0';
        } else {
            break;
        }
    }
    return num;
}

/**
 * @brief extractMonthFromStr extracts the month string and converts it to number (1..12) from the string at the given position
 * @param str dateString
 * @param beginPos position in dateString where month chars start
 * @param endPos end position of the extracted month in dateString (actually it is the position of the next char after last month char)
 * @return extracted month as number 1..12 or 0 if month can not be parsed
 */
int extractMonthFromStr(const std::string& str, size_t beginPos, size_t& endPos) {
    const char* orig = str.c_str() + beginPos;
    for (int i = 0; i < MONTH_NUM; i++) {
        if (strncmp(orig, monthsShort[i], strlen(monthsShort[i])) == 0) {
            if (strncmp(orig, months[i], strlen(months[i])) == 0) {
                endPos += strlen(months[i]);
            } else {
                endPos += strlen(monthsShort[i]);
            }
            return i + 1;
        }
    }
    return 0;
}

} // unnamed namespace

DateTime::DateTime(tm time) :
    time_(time),
    successfullyParsed_(false) {
    // nothing here
}

DateTime::DateTime() :
    successfullyParsed_(false) {
    time_t now;
    now = time(0);

    tm* timeInfo;
    timeInfo = localtime(&now);

    time_ = *timeInfo;
}

DateTime::DateTime(const DateTime& other) :
    time_(other.time_),
    successfullyParsed_(other.successfullyParsed_),
    originalString_(other.originalString_) {
    // nothing to do
}

DateTime& DateTime::operator=(const DateTime& other) {
    if (this != &other) {
        time_ = other.time_;
        successfullyParsed_ = other.successfullyParsed_;
        originalString_ = other.originalString_;
    }
    return *this;
}

bool DateTime::operator==(const DateTime& other) const {
    return time_.tm_mday == other.time_.tm_mday
           && time_.tm_mon == other.time_.tm_mon
           && time_.tm_year == other.time_.tm_year;
}

std::string DateTime::toString() const {
    std::stringstream ss;
    ss << getDay() << "/" << getMonth() << "/" << getYear();
    return ss.str();
}

DateTime::~DateTime() {
    // nothing here
}

void DateTime::setToCurrentDate() {
    time_t now(time(0));
    tm* timeInfo = localtime(&now);
    time_ = *timeInfo;
}

bool DateTime::setToDMY(int day, int month, int year) {
    time_t rawtime(time(0));
    tm* timeInfo = localtime(&rawtime);

    if (year < 0) {
        return false;
    }

    if (month < 1 || month > 12) {
        return false;
    }

    if (day < 1 || day > daysInMonth(month, year)) {
        return false;
    }

    timeInfo->tm_mday = day;
    timeInfo->tm_mon = month - 1;
    timeInfo->tm_year = year - 1900;

    // uncommented on 22.4.2015 by Zivac because it SEGFAULTs on Windows
    //rawtime = mktime(timeInfo);
    //timeInfo = localtime(&rawtime);

    time_ = *timeInfo;

    return true;
}

bool isLeapYear(int year) {
    return ((year % 4 == 0) && (year % 100 != 0)) || (year % 400 == 0);
}

int DateTime::daysInMonth(int month, int year) {
    switch (month) {
        case 1:
        case 3:
        case 5:
        case 7:
        case 8:
        case 10:
        case 12:
            return 31;
        case 2: {
            if (isLeapYear(year)) {
                return 29;
            } else {
                return 28;
            }
        }
        case 4:
        case 6:
        case 9:
        case 11:
        default:
            return 30;
    }
}

std::string DateTimeFormatter::format(const DateTime& dateTime) {
    std::size_t DDPos = format_.find("DD");
    std::size_t MMPos = format_.find("MM");
    std::size_t YYYYPos = format_.find("YYYY");

    std::string result = format_;

    std::stringstream ddss;
    ddss << dateTime.getDay() / 10 << dateTime.getDay() % 10;

    std::stringstream mmss;
    mmss << dateTime.getMonth() / 10 << dateTime.getMonth() % 10;

    std::stringstream yyyyss;
    yyyyss << dateTime.getCentury() / 10 << dateTime.getCentury() % 10
           << (dateTime.getYear() % 100) / 10 << dateTime.getYear() % 10;

    result.replace(DDPos, 2, ddss.str());
    result.replace(MMPos, 2, mmss.str());
    result.replace(YYYYPos, 4, yyyyss.str());

    return result;
}
