#include "sha1_bruteforce.h"
#include <iostream>
#include <iomanip>

// Функция для отображения прогресса
void progressCallback(int currentLength, long long attempts, double progress) {
    std::cout << "\rТекущая длина: " << currentLength 
              << " | Попыток: " << attempts 
              << " | Прогресс: " << std::fixed << std::setprecision(2) << progress << "%";
    std::cout.flush();
}

int main() {
    SHA1BruteForce bruteForce;
    
    // Установка целевого хеша (хеш для "test")
    bruteForce.setTargetHash("a94a8fe5ccb19ba61c4c0873d391e987982fbbd3");
    
    // Установка набора символов
    bruteForce.setCharset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");
    
    // Установка диапазона длин
    bruteForce.setLengthRange(1, 6);
    
    // Установка callback для отображения прогресса
    bruteForce.setProgressCallback(progressCallback);
    
    std::cout << "Начинаем перебор хеша SHA-1..." << std::endl;
    
    std::string result;
    if (bruteForce.bruteForce(result)) {
        std::cout << "\n\nНайдено соответствие: " << result << std::endl;
        std::cout << "Всего попыток: " << bruteForce.getTotalAttempts() << std::endl;
        std::cout << "Затраченное время: " << bruteForce.getElapsedTime() << " секунд" << std::endl;
        std::cout << "Скорость: " << bruteForce.getSpeed() << " попыток/секунду" << std::endl;
    } else {
        std::cout << "\nСоответствие не найдено" << std::endl;
    }
    
    return 0;
}