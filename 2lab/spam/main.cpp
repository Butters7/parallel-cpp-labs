#include "spam_filter.h"
#include <iostream>

int main() {
    SpamFilter filter;
    
    // Обучение на примерах
    filter.train("spam_emails.txt", "ham_emails.txt");
    
    // Сохранение модели
    filter.saveModel("spam_model.dat");
    
    // Загрузка модели (альтернатива обучению)
    // filter.loadModel("spam_model.dat");
    
    // Тестирование
    std::string testEmail = "Get cheap viagra now!!! Special offer!!!";
    bool isSpam = filter.classify(testEmail);
    
    std::cout << "Email: " << testEmail << std::endl;
    std::cout << "Classification: " << (isSpam ? "SPAM" : "HAM") << std::endl;
    
    return 0;
}