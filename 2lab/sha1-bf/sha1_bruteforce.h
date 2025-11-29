#ifndef SHA1_BRUTEFORCE_H
#define SHA1_BRUTEFORCE_H

#include <string>
#include <functional>
#include <vector>
#include <ctime>
#include <stdint.h>

class SHA1BruteForce {
public:
    SHA1BruteForce();
    ~SHA1BruteForce();

    void setTargetHash(const std::string& targetHash);
    void setCharset(const std::string& charset);
    void setLengthRange(int minLength, int maxLength);
    void setProgressCallback(std::function<void(int, long long, double)> callback);
    void setDictionaryFile(const std::string& dictionaryPath);
    void setDictionaryRules(const std::vector<std::string>& rules);
    
    bool bruteForce(std::string& result);
    bool dictionaryAttack(std::string& result);
    bool hybridAttack(std::string& result);
    bool bruteForceWithMask(std::string& result, const std::string& mask);
    
    long long getTotalAttempts();
    double getElapsedTime();
    double getSpeed();

private:
    std::string computeSHA1(const std::string& input);
    std::string generateCombination(long long index, int length) const;
    long long calculateTotalCombinations() const;
    std::vector<std::string> applyRules(const std::string& word);

    std::string targetHash;
    std::string charset;
    std::string dictionaryPath;
    std::vector<std::string> dictionaryRules;
    int minLength;
    int maxLength;
    long long totalAttempts;
    double elapsedTime;
    clock_t startTime;
    std::function<void(int, long long, double)> progressCallback;
    uint32_t leftRotate(uint32_t value, int shift);
    std::string toHexString(uint32_t value);
};

#endif