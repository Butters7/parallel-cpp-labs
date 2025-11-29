#include "sha1_bruteforce.h"
#include <openssl/sha.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <fstream>
#include <algorithm>

SHA1BruteForce::SHA1BruteForce() 
    : minLength(1), maxLength(6), totalAttempts(0), elapsedTime(0.0) {
    // Стандартный набор символов
    charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
}

SHA1BruteForce::~SHA1BruteForce() {
    // Деструктор
}

void SHA1BruteForce::setTargetHash(const std::string& targetHash) {
    this->targetHash = targetHash;
}

void SHA1BruteForce::setCharset(const std::string& charset) {
    this->charset = charset;
}

void SHA1BruteForce::setLengthRange(int minLength, int maxLength) {
    this->minLength = minLength;
    this->maxLength = maxLength;
}

void SHA1BruteForce::setProgressCallback(std::function<void(int, long long, double)> callback) {
    this->progressCallback = callback;
}

void SHA1BruteForce::setDictionaryFile(const std::string& dictionaryPath) {
    this->dictionaryPath = dictionaryPath;
}

void SHA1BruteForce::setDictionaryRules(const std::vector<std::string>& rules) {
    this->dictionaryRules = rules;
}

std::string SHA1BruteForce::computeSHA1(const std::string& input) {
    uint32_t h0 = 0x67452301;
    uint32_t h1 = 0xEFCDAB89;
    uint32_t h2 = 0x98BADCFE;
    uint32_t h3 = 0x10325476;
    uint32_t h4 = 0xC3D2E1F0;


    std::string msg = input;
    uint64_t original_bit_len = input.size() * 8;
        
    msg += static_cast<char>(0x80);
    while ((msg.size() * 8) % 512 != 448) {
        msg += static_cast<char>(0x00);
    }
    for (int i = 7; i >= 0; --i) {
        msg += static_cast<char>((original_bit_len >> (i * 8)) & 0xFF);
    }

    for (size_t i = 0; i < msg.size(); i += 64) {
        const char* block = msg.data() + i;
            
        uint32_t w[80];
        for (int j = 0; j < 16; ++j) {
            w[j] = (static_cast<uint32_t>(block[j*4] & 0xFF) << 24) |
                    (static_cast<uint32_t>(block[j*4+1] & 0xFF) << 16) |
                    (static_cast<uint32_t>(block[j*4+2] & 0xFF) << 8) |
                    (static_cast<uint32_t>(block[j*4+3] & 0xFF));
        }
            
        for (int j = 16; j < 80; ++j) {
            w[j] = leftRotate(w[j-3] ^ w[j-8] ^ w[j-14] ^ w[j-16], 1);
        }

        uint32_t a = h0;
        uint32_t b = h1;
        uint32_t c = h2;
        uint32_t d = h3;
        uint32_t e = h4;

        for (int j = 0; j < 80; ++j) {
            uint32_t f, k;

            if (j < 20) {
                f = (b & c) | ((~b) & d);
                k = 0x5A827999;
            } else if (j < 40) {
                f = b ^ c ^ d;
                k = 0x6ED9EBA1;
            } else if (j < 60) {
                f = (b & c) | (b & d) | (c & d);
                k = 0x8F1BBCDC;
            } else {
                f = b ^ c ^ d;
                k = 0xCA62C1D6;
            }

            uint32_t temp = leftRotate(a, 5) + f + e + k + w[j];
            e = d;
            d = c;
            c = leftRotate(b, 30);
            b = a;
            a = temp;
        }

        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
        h4 += e;
    }


    return toHexString(h0) + toHexString(h1) + toHexString(h2) + 
               toHexString(h3) + toHexString(h4);
}
uint32_t SHA1BruteForce::leftRotate(uint32_t value, int shift)
{
     return (value << shift) | (value >> (32 - shift));
}

std::string SHA1BruteForce::toHexString(uint32_t value)
{
    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(8) << value;
    return ss.str();
}

std::string SHA1BruteForce::generateCombination(long long index, int length) const {
    std::string result;
    long long temp = index;
    
    for (int i = 0; i < length; i++) {
        int charIndex = temp % charset.size();
        result.push_back(charset[charIndex]);
        temp /= charset.size();
    }
    
    return result;
}

long long SHA1BruteForce::calculateTotalCombinations() const {
    long long total = 0;
    for (int len = minLength; len <= maxLength; len++) {
        total += static_cast<long long>(std::pow(charset.size(), len));
    }
    return total;
}

// Функция для применения правил трансформации к словам
std::vector<std::string> SHA1BruteForce::applyRules(const std::string& word) {
    std::vector<std::string> transformations;
    transformations.push_back(word); // Оригинальное слово
    
    // Базовые трансформации
    std::string upper = word;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    transformations.push_back(upper);
    
    std::string lower = word;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    transformations.push_back(lower);
    
    // Capitalize first letter
    if (!word.empty()) {
        std::string capitalized = word;
        capitalized[0] = std::toupper(capitalized[0]);
        transformations.push_back(capitalized);
    }
    
    // Добавление чисел в конец (0-999)
    for (int i = 0; i <= 999; i++) {
        transformations.push_back(word + std::to_string(i));
        if (i <= 99) {
            transformations.push_back(word + (i < 10 ? "0" + std::to_string(i) : std::to_string(i)));
        }
    }
    
    // leet speak замены
    std::string leet = word;
    std::replace(leet.begin(), leet.end(), 'a', '@');
    std::replace(leet.begin(), leet.end(), 'e', '3');
    std::replace(leet.begin(), leet.end(), 'i', '1');
    std::replace(leet.begin(), leet.end(), 'o', '0');
    std::replace(leet.begin(), leet.end(), 's', '$');
    transformations.push_back(leet);
    
    // Дублирование слова
    transformations.push_back(word + word);
    
    // Реверс слова
    std::string reversed = word;
    std::reverse(reversed.begin(), reversed.end());
    transformations.push_back(reversed);
    
    // Удаление дубликатов
    std::sort(transformations.begin(), transformations.end());
    transformations.erase(std::unique(transformations.begin(), transformations.end()), transformations.end());
    
    return transformations;
}

bool SHA1BruteForce::dictionaryAttack(std::string& result) {
    if (dictionaryPath.empty()) {
        std::cerr << "Dictionary path not set!" << std::endl;
        return false;
    }
    
    std::ifstream file(dictionaryPath);
    if (!file.is_open()) {
        std::cerr << "Cannot open dictionary file: " << dictionaryPath << std::endl;
        return false;
    }
    
    bool found = false;
    totalAttempts = 0;
    startTime = clock();
    
    std::vector<std::string> dictionaryWords;
    std::string word;
    
    // Чтение словаря в память
    while (std::getline(file, word)) {
        if (!word.empty()) {
            dictionaryWords.push_back(word);
        }
    }
    file.close();
    
    std::cout << "Loaded " << dictionaryWords.size() << " words from dictionary" << std::endl;
    
    long long totalWords = dictionaryWords.size();
    long long processedWords = 0;
    
    // Обработка словаря
    for (size_t i = 0; i < dictionaryWords.size(); i++) {
        if (found) continue;
        
        std::vector<std::string> candidates = applyRules(dictionaryWords[i]);
        
        for (const auto& candidate : candidates) {
            if (found) break;
            
            std::string candidateHash = computeSHA1(candidate);
            
            if (candidateHash == targetHash) {
                found = true;
                result = candidate;
                std::cout << "Found in dictionary attack: " << result << std::endl;
                break;
            }
            
            totalAttempts++;
        }
        
        // Обновление прогресса
        processedWords++;
        
        if (progressCallback && processedWords % 1000 == 0) {
            double progress = static_cast<double>(processedWords) / totalWords * 100.0;
            progressCallback(0, processedWords, progress);
        }
    }
    
    elapsedTime = (clock() - startTime) / (double)CLOCKS_PER_SEC;
    return found;
}

bool SHA1BruteForce::hybridAttack(std::string& result) {
    // Сначала пробуем атаку по словарю
    std::cout << "Starting hybrid attack (dictionary + brute force)..." << std::endl;
    
    if (!dictionaryPath.empty()) {
        std::cout << "Trying dictionary attack first..." << std::endl;
        if (dictionaryAttack(result)) {
            return true;
        }
        std::cout << "Dictionary attack failed, switching to brute force..." << std::endl;
    }
    
    // Если словарная атака не сработала, переходим к brute force
    return bruteForce(result);
}

bool SHA1BruteForce::bruteForce(std::string& result) {
    bool found = false;
    totalAttempts = 0;
    startTime = clock();
    
    long long totalCombinations = calculateTotalCombinations();
    long long processedCombinations = 0;
    
    // Перебор по длине пароля
    for (int length = minLength; length <= maxLength && !found; length++) {
        long long combinationsForLength = static_cast<long long>(std::pow(charset.size(), length));
        
        for (long long i = 0; i < combinationsForLength; i++) {
            if (found) continue;
            
            std::string candidate = generateCombination(i, length);
            std::string candidateHash = computeSHA1(candidate);
            
            if (candidateHash == targetHash) {
                found = true;
                result = candidate;
            }
            
            totalAttempts++;
            processedCombinations++;
            
            if (progressCallback && processedCombinations % 10000 == 0) {
                double progress = static_cast<double>(processedCombinations) / totalCombinations * 100.0;
                progressCallback(length, processedCombinations, progress);
            }
        }
        
        if (progressCallback) {
            double progress = static_cast<double>(processedCombinations) / totalCombinations * 100.0;
            progressCallback(length, processedCombinations, progress);
        }
    }
    
    elapsedTime = (clock() - startTime) / (double)CLOCKS_PER_SEC;
    return found;
}

bool SHA1BruteForce::bruteForceWithMask(std::string& result, const std::string& mask) {
    // Простая реализация атаки по маске
    // Маска: "pass???" где ? - любой символ из charset
    bool found = false;
    totalAttempts = 0;
    startTime = clock();
    
    // Подсчет количества переменных символов в маске
    int variableCount = 0;
    for (char c : mask) {
        if (c == '?') variableCount++;
    }
    
    if (variableCount == 0) {
        // Проверяем саму маску
        std::string candidateHash = computeSHA1(mask);
        if (candidateHash == targetHash) {
            result = mask;
            found = true;
        }
        totalAttempts = 1;
        elapsedTime = (clock() - startTime) / (double)CLOCKS_PER_SEC;
        return found;
    }
    
    long long combinationsForMask = static_cast<long long>(std::pow(charset.size(), variableCount));
    
    for (long long i = 0; i < combinationsForMask; i++) {
        if (found) continue;
        
        std::string candidate = mask;
        long long temp = i;
        int pos = 0;
        
        // Заменяем '?' в маске символами из charset
        for (char& c : candidate) {
            if (c == '?') {
                int charIndex = temp % charset.size();
                c = charset[charIndex];
                temp /= charset.size();
                pos++;
            }
        }
        
        std::string candidateHash = computeSHA1(candidate);
        
        if (candidateHash == targetHash) {
            found = true;
            result = candidate;
        }
        
        totalAttempts++;
        
        if (progressCallback && totalAttempts % 10000 == 0) {
            double progress = static_cast<double>(totalAttempts) / combinationsForMask * 100.0;
            progressCallback(variableCount, totalAttempts, progress);
        }
    }
    
    elapsedTime = (clock() - startTime) / (double)CLOCKS_PER_SEC;
    return found;
}

long long SHA1BruteForce::getTotalAttempts() {
    return totalAttempts;
}

double SHA1BruteForce::getElapsedTime() {
    return elapsedTime;
}

double SHA1BruteForce::getSpeed() {
    return elapsedTime > 0 ? totalAttempts / elapsedTime : 0;
}